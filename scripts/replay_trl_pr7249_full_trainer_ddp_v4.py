#!/usr/bin/env python3
"""Run pinned AsyncGRPO normalization through Trainer.train on two CPU DDP ranks (v4)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


BASE_COMMIT = "a98fa6a4428f9aae58dfb26d729d7437f662f27a"
CANDIDATE_COMMIT = "5234eb7c70f4ca7eb92fe8e01611a33eaff17f40"
SOURCE_HASHES = {
    "base": "b84c4544a4ac2aac5b08681a93fb3d49d1f24ff2f2f9fe53a6c35ab5c8f0ea40",
    "candidate": "5c4fc857c563db0d9f937a2edf89dd74c4f218a31f3a05bf33bffbf5285749dc",
}


def sample(n_tokens: int, advantage: float, sample_id: int) -> dict:
    mask = [0, 0] + [1] * (n_tokens // 2) + [0] + [1] * (n_tokens - n_tokens // 2)
    return {
        "input_ids": [1] * len(mask),
        "completion_mask": mask,
        "old_log_probs": [-math.log(2)] * len(mask),
        "advantage": advantage,
        "group_id": sample_id,
        "prompt_id": sample_id,
        "metrics": {},
    }


def cases() -> dict[str, tuple[list[list[dict]], int]]:
    return {
        "unequal_rank_and_microbatch_counts": (
            [
                [[sample(2, -1.5, 40)], [sample(7, -0.4, 41)]],
                [[sample(11, -0.9, 42)], [sample(3, -1.7, 43)]],
            ],
            2,
        ),
        "equal_token_control": (
            [
                [[sample(5, -1.8, 50)], [sample(5, -0.7, 51)]],
                [[sample(5, -1.1, 52)], [sample(5, -1.4, 53)]],
            ],
            2,
        ),
        "zero_local_tokens_positive_global": (
            [
                [[sample(0, -1.8, 60)], [sample(6, -0.6, 61)]],
                [[sample(0, -0.7, 62)], [sample(3, -1.4, 63)]],
            ],
            2,
        ),
        "short_final_window": (
            [[[sample(2, -1.9, 70)], [sample(11, -0.4, 71)]]],
            2,
        ),
    }


def build_model():
    from types import SimpleNamespace
    from transformers import PretrainedConfig

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(2, 2)
            torch.nn.init.ones_(self.embed.weight)

        def forward(self, input_ids, position_ids, use_cache=False):
            return SimpleNamespace(last_hidden_state=self.embed(input_ids))

    class TokenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = PretrainedConfig()
            self.base_model = Backbone()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.zeros_(self.lm_head.weight)

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, position_ids, labels=None, completion_mask=None, use_cache=False):
            hidden = self.base_model(input_ids, position_ids).last_hidden_state
            log_probs = self.lm_head(hidden[:, :-1]).log_softmax(-1)
            return {
                "log_probs": log_probs.gather(-1, input_ids[:, 1:, None]).squeeze(-1),
                "entropy": -(log_probs.exp() * log_probs).sum(-1),
                "aux_loss": self.lm_head.weight.sum() + 2.0,
            }

    return TokenModel()


def collate_rank_rows(rank_rows: list[list[dict]]) -> dict[str, torch.Tensor]:
    from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout

    collator = DataCollatorForRollout(pad_token_id=0, num_processes=len(rank_rows))
    return collator([rank_rows])


def oracle_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        labels=batch["input_ids"],
        completion_mask=batch["completion_mask"],
        use_cache=False,
    )
    mask = batch["completion_mask"][:, 1:].float()
    old_log_probs = batch["old_log_probs"][:, 1:]
    advantages = batch["advantages"][:, 1:].float()
    ratio = torch.exp(outputs["log_probs"] - old_log_probs)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    per_token = -torch.minimum(ratio * advantages, clipped * advantages)
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def tensor_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def source_commit(repo: Path, arm: str) -> str:
    expected_commit = BASE_COMMIT if arm == "base" else CANDIDATE_COMMIT
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    if actual != expected_commit or status:
        raise RuntimeError(f"source checkout mismatch: commit={actual}; dirty={bool(status)}")
    rel = "trl/experimental/async_grpo/async_grpo_trainer.py"
    digest = hashlib.sha256((repo / rel).read_bytes()).hexdigest()
    if digest != SOURCE_HASHES[arm]:
        raise RuntimeError(f"source hash mismatch: {digest}")
    return actual


def train_case(AsyncGRPOTrainer, Trainer, TrainingArguments, rank_rows, requested_slots, rank, output):
    from trl.models.utils import _ForwardRedirection

    all_examples = [example for microbatch in rank_rows for row in microbatch for example in row]
    rank_batches = []
    global_counts = []
    for microbatch in rank_rows:
        batch = collate_rank_rows(microbatch)
        global_counts.append(int(batch["global_n_tokens"][0].item()))
        rank_batches.append({key: value[rank : rank + 1].contiguous() for key, value in batch.items()})

    torch.manual_seed(12345)
    model = build_model()
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    args = TrainingArguments(
        output_dir=str(output),
        use_cpu=True,
        report_to="none",
        logging_strategy="no",
        save_strategy="no",
        disable_tqdm=True,
        max_steps=1,
        gradient_accumulation_steps=requested_slots,
        per_device_train_batch_size=1,
        max_grad_norm=0.0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=12345,
    )
    trainer = AsyncGRPOTrainer.__new__(AsyncGRPOTrainer)
    Trainer.__init__(
        trainer,
        model=model,
        args=args,
        compute_loss_func="non-None value to disable Trainer auto-scaling",
        optimizers=(optimizer, scheduler),
    )
    trainer.model_accepts_loss_kwargs = False
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.aux_loss_enabled = False
    trainer.router_aux_loss_coef = 0.1
    trainer._metrics = {"train": defaultdict(list)}
    # Normally initialized by AsyncGRPOTrainer.__init__; this harness deliberately skips its worker/server setup.
    trainer._step_forward_tokens = trainer._step_trained_tokens = 0
    trainer._step_seq_len_weighted = trainer._step_samples = trainer._step_forward_s = 0
    trainer._step_microbatches = trainer._current_train_step_time = 0
    trainer.rollout_worker = None
    trainer._forward_redirection = _ForwardRedirection()
    trainer.args.beta = 0.0
    # Keep the actual Trainer optimizer loop but omit the asynchronous rollout-worker lifecycle.
    trainer._inner_training_loop = Trainer._inner_training_loop.__get__(trainer, AsyncGRPOTrainer)
    trainer.log = Trainer.log.__get__(trainer, AsyncGRPOTrainer)
    trainer.get_train_dataloader = lambda: DataLoader(rank_batches, batch_size=None)
    captured_grads = {}
    rank_step_losses = []
    original_training_step = trainer.training_step

    def capture_training_step(model_arg, inputs, num_items_in_batch):
        loss = original_training_step(model_arg, inputs, num_items_in_batch)
        unwrapped = trainer.accelerator.unwrap_model(model_arg)
        captured_grads.clear()
        captured_grads.update({
            name: param.grad.detach().clone()
            for name, param in unwrapped.named_parameters()
            if param.grad is not None
        })
        rank_step_losses.append(float(loss))
        return loss

    trainer.training_step = capture_training_step
    result = trainer.train()

    trained = trainer.accelerator.unwrap_model(trainer.model_wrapped)
    after = {name: value.detach().clone() for name, value in trained.state_dict().items()}
    distributed_step_loss = torch.tensor(sum(rank_step_losses), dtype=torch.float64)
    dist.all_reduce(distributed_step_loss, op=dist.ReduceOp.SUM)
    global_step_loss = float((distributed_step_loss / dist.get_world_size()).item())
    oracle = build_model()
    oracle.load_state_dict(initial)
    pooled = collate_rank_rows([all_examples])
    expected_loss = oracle_loss(oracle, pooled)
    expected_loss.backward()
    expected_grads = {name: p.grad.detach().clone() for name, p in oracle.named_parameters()}
    with torch.no_grad():
        for param in oracle.parameters():
            param.add_(param.grad, alpha=-0.1)
    expected_after = {name: value.detach().clone() for name, value in oracle.state_dict().items()}
    grad = captured_grads
    if set(grad) != set(expected_grads):
        raise RuntimeError(f"gradient capture mismatch: captured={sorted(grad)}, expected={sorted(expected_grads)}")
    grad_error = max(tensor_error(grad[name], expected_grads[name]) for name in grad)
    parameter_error = max(tensor_error(after[name], expected_after[name]) for name in after)
    loss_error = abs(global_step_loss - float(expected_loss.detach()))
    state_digest = hashlib.sha256(b"".join(
        name.encode() + after[name].contiguous().cpu().numpy().tobytes() for name in sorted(after)
    )).hexdigest()
    return {
        "rank": rank,
        "global_step": result.global_step,
        "trainer_training_loss_local_diagnostic": float(result.training_loss),
        "rank_mean_training_step_loss": global_step_loss,
        "oracle_loss": float(expected_loss.detach()),
        "loss_abs_error": loss_error,
        "max_gradient_abs_error": grad_error,
        "max_sgd_parameter_abs_error": parameter_error,
        "global_tokens_per_microbatch": global_counts,
        "post_step_state_sha256": state_digest,
        "finite": all(math.isfinite(v) for v in [loss_error, grad_error, parameter_error]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--arm", choices=["base", "candidate"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    commit = source_commit(repo, args.arm)

    torch.set_num_threads(1)
    if not dist.is_initialized():
        from datetime import timedelta

        dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    if dist.get_world_size() != 2 or dist.get_backend() != "gloo":
        raise RuntimeError(f"expected two-process Gloo, got {dist.get_world_size()}/{dist.get_backend()}")

    create_error = [None]
    if rank == 0:
        try:
            if args.output.exists():
                raise FileExistsError(f"output must be new: {args.output}")
            args.output.mkdir(parents=True)
        except BaseException as exc:
            create_error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(create_error, src=0)
    if create_error[0]:
        raise RuntimeError(create_error[0])
    dist.barrier()

    from transformers import Trainer, TrainingArguments
    from trl.experimental.async_grpo import AsyncGRPOTrainer

    results = {}
    for case_name, (rank_rows, slots) in cases().items():
        per_rank = train_case(
            AsyncGRPOTrainer,
            Trainer,
            TrainingArguments,
            rank_rows,
            slots,
            rank,
            args.output / f"trainer-{case_name}-rank-{rank}",
        )
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, per_rank)
        if rank == 0:
            first = gathered[0]
            results[case_name] = {
                "rank_results": gathered,
                "rank_state_equal": all(item["post_step_state_sha256"] == first["post_step_state_sha256"] for item in gathered),
                "max_loss_abs_error": max(item["loss_abs_error"] for item in gathered),
                "max_gradient_abs_error": max(item["max_gradient_abs_error"] for item in gathered),
                "max_sgd_parameter_abs_error": max(item["max_sgd_parameter_abs_error"] for item in gathered),
                "finite": all(item["finite"] for item in gathered),
            }
        dist.barrier()

    if rank == 0:
        output = {
            "protocol_id": "trl-async-grpo-pr7249-full-trainer-ddp-v4",
            "arm": args.arm,
            "source_commit": commit,
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "accelerate": __import__("accelerate").__version__,
            },
            "backend": "gloo",
            "world_size": 2,
            "device": "cpu",
            "elapsed_seconds": time.monotonic() - started,
            "cases": results,
        }
        (args.output / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps(output, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
