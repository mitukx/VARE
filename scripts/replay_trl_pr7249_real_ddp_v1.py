#!/usr/bin/env python3
"""Replay TRL AsyncGRPO token-mean loss with an actual two-process CPU DDP reducer."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.distributed as dist


BASE_COMMIT = "a98fa6a4428f9aae58dfb26d729d7437f662f27a"
CANDIDATE_COMMIT = "5234eb7c70f4ca7eb92fe8e01611a33eaff17f40"
EXPECTED_SOURCE_HASHES = {
    "base": {
        "trl/experimental/async_grpo/async_grpo_trainer.py": "b84c4544a4ac2aac5b08681a93fb3d49d1f24ff2f2f9fe53a6c35ab5c8f0ea40",
    },
    "candidate": {
        "trl/experimental/async_grpo/async_grpo_trainer.py": "5c4fc857c563db0d9f937a2edf89dd74c4f218a31f3a05bf33bffbf5285749dc",
    },
}


def sample(n_tokens: int, advantage: float, sample_id: int) -> dict:
    # Prompt/context positions are masked; only `n_tokens` completion positions contribute.
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
    # Each inner list is one optimizer-window microbatch, with examples ordered by rank.
    return {
        "unequal_rank_and_microbatch_counts": (
            [
                [[sample(1, -2.0, 0)], [sample(9, -0.5, 1)]],
                [[sample(15, -0.75, 2)], [sample(1, -1.5, 3)]],
            ],
            2,
        ),
        "equal_token_control": (
            [
                [[sample(4, -2.0, 10)], [sample(4, -0.5, 11)]],
                [[sample(4, -0.75, 12)], [sample(4, -1.5, 13)]],
            ],
            2,
        ),
        "zero_local_tokens_positive_global": (
            [
                [[sample(0, -2.0, 20)], [sample(8, -0.5, 21)]],
                [[sample(0, -0.75, 22)], [sample(4, -1.5, 23)]],
            ],
            2,
        ),
        "short_final_window": (
            [[[sample(1, -2.0, 30)], [sample(9, -0.5, 31)]]],
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


def oracle_loss(model: torch.nn.Module, collated: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=collated["input_ids"],
        position_ids=collated["position_ids"],
        labels=collated["input_ids"],
        completion_mask=collated["completion_mask"],
        use_cache=False,
    )
    mask = collated["completion_mask"][:, 1:].float()
    old_log_probs = collated["old_log_probs"][:, 1:]
    advantages = collated["advantages"][:, 1:].float()
    ratio = torch.exp(outputs["log_probs"] - old_log_probs)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    per_token = -torch.minimum(ratio * advantages, clipped * advantages)
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def max_tensor_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def verify_source(repo: Path, arm: str) -> str:
    expected_commit = BASE_COMMIT if arm == "base" else CANDIDATE_COMMIT
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    if commit != expected_commit or status.strip():
        raise RuntimeError(f"source checkout mismatch: commit={commit}, dirty={bool(status.strip())}")
    import hashlib

    for rel, expected in EXPECTED_SOURCE_HASHES[arm].items():
        actual = hashlib.sha256((repo / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"source hash mismatch for {rel}: {actual}")
    return commit


def run_case(trainer, rank: int, rank_rows: list[list[dict]], requested_slots: int,
             initial_state: dict[str, torch.Tensor]) -> dict:
    from torch.nn.parallel import DistributedDataParallel

    trainer.model.module.load_state_dict(initial_state)
    dist.barrier()
    microbatches = [collate_rank_rows(rows) for rows in rank_rows]
    local_batches = [
        {key: value[rank : rank + 1].contiguous() for key, value in batch.items()}
        for batch in microbatches
    ]
    batch_samples, num_items = trainer.get_batch_samples(
        iter(local_batches), requested_slots, trainer.args.device
    )
    actual_slots = len(batch_samples)
    trainer.current_gradient_accumulation_steps = actual_slots
    trainer._step_forward_tokens = trainer._step_trained_tokens = 0
    trainer._step_seq_len_weighted = trainer._step_samples = trainer._step_forward_s = 0
    trainer._step_microbatches = trainer._current_train_step_time = 0
    trainer.model.zero_grad(set_to_none=True)
    local_reported_loss = 0.0
    for index, batch in enumerate(batch_samples):
        sync_context = trainer.model.no_sync() if index < actual_slots - 1 else torch.enable_grad()
        with sync_context:
            loss = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_items)
            local_reported_loss += float(loss.detach())
            loss.backward()

    if not isinstance(trainer.model, DistributedDataParallel):
        raise RuntimeError(f"expected DDP wrapper, got {type(trainer.model).__name__}")
    reported_loss_tensor = torch.tensor(local_reported_loss, dtype=torch.float64)
    dist.all_reduce(reported_loss_tensor, op=dist.ReduceOp.SUM)
    reported_loss = float(reported_loss_tensor.item() / dist.get_world_size())
    ddp_grads = {name: param.grad.detach().clone() for name, param in trainer.model.module.named_parameters()}
    pre_step = {name: param.detach().clone() for name, param in trainer.model.module.named_parameters()}
    with torch.no_grad():
        for param in trainer.model.module.parameters():
            param.add_(param.grad, alpha=-0.1)
    after_step = {name: param.detach().clone() for name, param in trainer.model.module.named_parameters()}

    all_examples = [sample for rows in rank_rows for rank_examples in rows for sample in rank_examples]
    oracle = build_model()
    oracle.load_state_dict(pre_step)
    oracle_batch = collate_rank_rows([all_examples])
    oracle_batch = {key: value[:1].contiguous() for key, value in oracle_batch.items()}
    expected_loss = oracle_loss(oracle, oracle_batch)
    expected_loss.backward()
    expected_grads = {name: param.grad.detach().clone() for name, param in oracle.named_parameters()}
    with torch.no_grad():
        for param in oracle.parameters():
            param.add_(param.grad, alpha=-0.1)
    expected_after = {name: param.detach().clone() for name, param in oracle.named_parameters()}

    grad_error = max(max_tensor_error(ddp_grads[name], expected_grads[name]) for name in ddp_grads)
    parameter_error = max(max_tensor_error(after_step[name], expected_after[name]) for name in after_step)
    loss_error = abs(reported_loss - float(expected_loss.detach()))
    grad_hash = hashlib.sha256(b"".join(
        name.encode() + ddp_grads[name].contiguous().cpu().numpy().tobytes()
        for name in sorted(ddp_grads)
    )).hexdigest()
    parameter_hash = hashlib.sha256(b"".join(
        name.encode() + after_step[name].contiguous().cpu().numpy().tobytes()
        for name in sorted(after_step)
    )).hexdigest()
    return {
        "rank": rank,
        "actual_accumulation_slots": actual_slots,
        "window_token_count": int(num_items.item()) if num_items is not None else None,
        "loss": reported_loss,
        "oracle_loss": float(expected_loss.detach()),
        "loss_abs_error": loss_error,
        "max_gradient_abs_error": grad_error,
        "max_sgd_parameter_abs_error": parameter_error,
        "ddp_gradient_sha256": grad_hash,
        "post_sgd_parameter_sha256": parameter_hash,
        "finite": all(math.isfinite(value) for value in [reported_loss, loss_error, grad_error, parameter_error]),
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
    commit = verify_source(repo, args.arm)

    torch.set_num_threads(1)
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", timeout=timedelta(seconds=180))
    if dist.get_backend() != "gloo" or dist.get_world_size() != 2:
        raise RuntimeError(f"expected two-process Gloo, got {dist.get_backend()}/{dist.get_world_size()}")
    rank = dist.get_rank()
    create_error = [None]
    if rank == 0:
        try:
            if args.output.exists():
                raise FileExistsError(f"output directory must be new: {args.output}")
            args.output.mkdir(parents=True)
        except BaseException as exc:
            create_error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(create_error, src=0)
    if create_error[0] is not None:
        raise RuntimeError(create_error[0])
    dist.barrier()

    from collections import defaultdict
    from transformers import Trainer, TrainingArguments
    from trl.experimental.async_grpo import AsyncGRPOTrainer
    from trl.models.utils import _ForwardRedirection

    torch.manual_seed(12345)
    trainer = AsyncGRPOTrainer.__new__(AsyncGRPOTrainer)
    training_args = TrainingArguments(
        output_dir=str(args.output / f"trainer-rank-{rank}"),
        use_cpu=True,
        report_to="none",
        gradient_accumulation_steps=2,
        ddp_find_unused_parameters=False,
        max_grad_norm=0.0,
        remove_unused_columns=False,
    )
    Trainer.__init__(
        trainer,
        model=build_model(),
        args=training_args,
        compute_loss_func="non-None value to disable Trainer auto-scaling",
    )
    trainer.model_accepts_loss_kwargs = False
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.aux_loss_enabled = False
    trainer.router_aux_loss_coef = 0.1
    trainer._metrics = {"train": defaultdict(list)}
    trainer._forward_redirection = _ForwardRedirection()
    trainer.args.beta = 0.0
    trainer.model = trainer.accelerator.prepare_model(trainer.model)
    if trainer.accelerator.num_processes != 2:
        raise RuntimeError(f"Trainer Accelerator did not detect two processes: {trainer.accelerator.num_processes}")
    initial_state = {name: value.detach().clone() for name, value in trainer.model.module.state_dict().items()}

    results = {}
    for name, (rank_rows, requested_slots) in cases().items():
        local = run_case(trainer, rank, rank_rows, requested_slots, initial_state)
        gathered: list[dict | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        if rank == 0:
            assert all(result is not None for result in gathered)
            first = gathered[0]
            results[name] = {
                "rank_results": gathered,
                "rank_count": len(gathered),
                "max_loss_abs_error": max(result["loss_abs_error"] for result in gathered),
                "max_gradient_abs_error": max(result["max_gradient_abs_error"] for result in gathered),
                "max_sgd_parameter_abs_error": max(result["max_sgd_parameter_abs_error"] for result in gathered),
                "rank_state_equal": all(
                    result["ddp_gradient_sha256"] == first["ddp_gradient_sha256"]
                    and result["post_sgd_parameter_sha256"] == first["post_sgd_parameter_sha256"]
                    for result in gathered
                ),
                "finite": all(result["finite"] for result in gathered),
            }
        dist.barrier()

    if rank == 0:
        output = {
            "protocol_id": "trl-async-grpo-pr7249-real-ddp-v1",
            "arm": args.arm,
            "source_commit": commit,
            "backend": "gloo",
            "world_size": 2,
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "accelerate": __import__("accelerate").__version__,
            },
            "device": "cpu",
            "elapsed_seconds": time.monotonic() - started,
            "cases": results,
        }
        out = args.output / "summary.json"
        out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
