#!/usr/bin/env python3
"""Run pinned AsyncDistillation normalization through Trainer.train on two CPU Gloo ranks."""
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
    "base": "449b8f5b2006e20a2420fd8e53d6fef2d96a026cb0ff51cf00f6add55d4a55e2",
    "candidate": "7a99d26b9124787978fd6c29752a666ff4160694c3feb999abf68afb6a49a1ff",
}


def sample(n_tokens: int, teacher_p: float, sample_id: int) -> dict:
    mask = [0, 0] + [1] * (n_tokens // 2) + [0] + [1] * (n_tokens - n_tokens // 2)
    return {
        "input_ids": [1] * len(mask),
        "completion_mask": mask,
        "teacher_topk_ids": [[0, 1]] * len(mask),
        "teacher_topk_logprobs": [[math.log(teacher_p), math.log(1 - teacher_p)]] * len(mask),
        "teacher_id": "teacher",
        "prompt_id": sample_id,
        "metrics": {},
    }


def cases() -> dict[str, tuple[list[list[list[dict]]], list[list[int]], list[int]]]:
    return {
        "unequal_rank_and_microbatch_counts": (
            [
                [[sample(2, 0.9, 401)], [sample(7, 0.8, 402)]],
                [[sample(9, 0.6, 403)], [sample(2, 0.55, 404)]],
            ],
            [[2, 7], [9, 2]],
            [11, 9],
        ),
        "equal_token_control": (
            [
                [[sample(4, 0.9, 411)], [sample(6, 0.8, 412)]],
                [[sample(6, 0.6, 413)], [sample(4, 0.55, 414)]],
            ],
            [[4, 6], [6, 4]],
            [10, 10],
        ),
        "zero_local_tokens_positive_global": (
            [
                [[sample(0, 0.9, 421)], [sample(0, 0.8, 422)]],
                [[sample(5, 0.6, 423)], [sample(4, 0.55, 424)]],
            ],
            [[0, 0], [5, 4]],
            [5, 4],
        ),
        "short_final_window": (
            [
                [[sample(3, 0.9, 431)]],
                [[sample(8, 0.55, 432)]],
            ],
            [[3], [8]],
            [11],
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
            logits = self.lm_head(hidden[:, :-1])
            log_probs = logits.log_softmax(-1)
            return {"log_probs": log_probs}

    return TokenModel()


def collate_rank_rows(rank_rows: list[list[dict]], n_ranks: int):
    from trl.experimental.async_distillation.async_distillation_trainer import DataCollatorForRollout

    collator = DataCollatorForRollout(pad_token_id=0, teacher_top_k=2, num_processes=n_ranks)
    return collator([rank_rows])


def pooled_oracle(model, examples: list[dict]):
    """Independent two-token forward-KL objective, summed by completion token then globally averaged."""
    hidden = model.base_model(
        input_ids=torch.tensor([[1]], dtype=torch.long),
        position_ids=torch.tensor([[0]], dtype=torch.long),
    ).last_hidden_state[:, 0, :]
    student_log_probs = model.lm_head(hidden).log_softmax(-1)[0]
    total_tokens = sum(sum(example["completion_mask"]) for example in examples)
    if total_tokens <= 0:
        raise RuntimeError("oracle requires positive global completion-token count")
    loss_sum = student_log_probs.sum() * 0.0
    for example in examples:
        count = sum(example["completion_mask"])
        if count:
            p = example["teacher_topk_logprobs"][0]
            teacher_log_probs = torch.tensor(p, dtype=student_log_probs.dtype)
            teacher_probs = teacher_log_probs.exp()
            loss_sum = loss_sum + count * (teacher_probs * (teacher_log_probs - student_log_probs)).sum()
    return loss_sum / total_tokens


def tensor_error(left, right) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def source_commit(repo: Path, arm: str) -> str:
    expected = BASE_COMMIT if arm == "base" else CANDIDATE_COMMIT
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    digest = hashlib.sha256((repo / "trl/experimental/async_distillation/async_distillation_trainer.py").read_bytes()).hexdigest()
    if actual != expected or dirty or digest != SOURCE_HASHES[arm]:
        raise RuntimeError(f"source checkout mismatch: commit={actual}; dirty={bool(dirty)}; hash={digest}")
    return actual


def train_case(AsyncDistillationTrainer, Trainer, TrainingArguments, case_name, rank_rows, expected_local,
               expected_global, rank, output):
    from trl.models.utils import _ForwardRedirection

    local_counts = [[sum(sample["completion_mask"]) for sample in microbatch] for microbatch in rank_rows]
    if len(rank_rows) != dist.get_world_size() or local_counts != expected_local:
        raise RuntimeError(f"frozen local-token layout mismatch for {case_name}: {local_counts}")
    n_microbatches = len(rank_rows[0])
    if any(len(rows) != n_microbatches for rows in rank_rows):
        raise RuntimeError(f"rank/microbatch shape mismatch for {case_name}")

    all_examples = [sample for rank_rows_i in rank_rows for microbatch in rank_rows_i for sample in microbatch]
    rank_batches, global_counts = [], []
    for j in range(n_microbatches):
        batch = collate_rank_rows([rank_rows[i][j] for i in range(len(rank_rows))], len(rank_rows))
        count = int(batch["global_n_tokens"][0].item())
        global_counts.append(count)
        rank_batches.append({key: value[rank:rank + 1].contiguous() for key, value in batch.items()})
    if global_counts != expected_global:
        raise RuntimeError(f"frozen global-token layout mismatch for {case_name}: {global_counts}")

    torch.manual_seed(12345)
    model = build_model()
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    args = TrainingArguments(
        output_dir=str(output), use_cpu=True, report_to="none", logging_strategy="no", save_strategy="no",
        disable_tqdm=True, max_steps=1, gradient_accumulation_steps=2, per_device_train_batch_size=1,
        max_grad_norm=0.0, remove_unused_columns=False, ddp_find_unused_parameters=False, seed=12345,
    )
    trainer = AsyncDistillationTrainer.__new__(AsyncDistillationTrainer)
    Trainer.__init__(trainer, model=model, args=args,
                     compute_loss_func="non-None value to disable Trainer auto-scaling",
                     optimizers=(optimizer, scheduler))
    trainer.model_accepts_loss_kwargs = False
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._teacher_ids = ["teacher"]
    trainer._forward_redirection = _ForwardRedirection()
    trainer._step_forward_tokens = trainer._step_trained_tokens = 0.0
    trainer._step_seq_len_weighted = trainer._step_samples = trainer._step_forward_s = 0.0
    trainer._step_microbatches = trainer._current_train_step_time = 0.0
    trainer._step_optimizer_s = trainer._last_step_end_time = 0.0
    trainer._last_forward_time_s = 0.0
    trainer._rollout_dataset = None
    trainer.args.beta = 0.0
    trainer.args.teacher_temperature = 1.0
    trainer.args.add_tail_bucket = False
    trainer._inner_training_loop = Trainer._inner_training_loop.__get__(trainer, AsyncDistillationTrainer)
    trainer.log = Trainer.log.__get__(trainer, AsyncDistillationTrainer)
    trainer.get_train_dataloader = lambda: DataLoader(rank_batches, batch_size=None)
    captured_grads, step_losses = {}, []
    original_training_step = trainer.training_step

    def capture_training_step(model_arg, inputs, num_items_in_batch):
        loss = original_training_step(model_arg, inputs, num_items_in_batch)
        unwrapped = trainer.accelerator.unwrap_model(model_arg)
        captured_grads.clear()
        captured_grads.update({name: param.grad.detach().clone() for name, param in unwrapped.named_parameters()
                               if param.grad is not None})
        step_losses.append(float(loss))
        return loss

    trainer.training_step = capture_training_step
    result = trainer.train()
    trained = trainer.accelerator.unwrap_model(trainer.model_wrapped)
    after = {name: value.detach().clone() for name, value in trained.state_dict().items()}
    reduced_loss = torch.tensor(sum(step_losses), dtype=torch.float64)
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
    global_step_loss = float((reduced_loss / dist.get_world_size()).item())

    oracle = build_model()
    oracle.load_state_dict(initial)
    expected_loss = pooled_oracle(oracle, all_examples)
    expected_loss.backward()
    expected_grads = {name: param.grad.detach().clone() for name, param in oracle.named_parameters()}
    with torch.no_grad():
        for param in oracle.parameters():
            param.add_(param.grad, alpha=-0.1)
    expected_after = {name: value.detach().clone() for name, value in oracle.state_dict().items()}
    if set(captured_grads) != set(expected_grads):
        raise RuntimeError(f"gradient coverage mismatch: {sorted(captured_grads)} vs {sorted(expected_grads)}")
    grad_error = max(tensor_error(captured_grads[name], expected_grads[name]) for name in captured_grads)
    parameter_error = max(tensor_error(after[name], expected_after[name]) for name in after)
    loss_error = abs(global_step_loss - float(expected_loss.detach()))
    digest = hashlib.sha256(b"".join(name.encode() + after[name].contiguous().cpu().numpy().tobytes()
                                     for name in sorted(after))).hexdigest()
    return {
        "rank": rank, "global_step": result.global_step, "global_step_loss": global_step_loss,
        "oracle_loss": float(expected_loss.detach()), "loss_abs_error": loss_error,
        "max_gradient_abs_error": grad_error, "max_sgd_parameter_abs_error": parameter_error,
        "local_tokens_by_rank_per_microbatch": local_counts, "global_tokens_per_microbatch": global_counts,
        "post_step_state_sha256": digest,
        "finite": all(math.isfinite(x) for x in (loss_error, grad_error, parameter_error)),
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
        raise RuntimeError("expected two-process CPU Gloo")
    error = [None]
    if rank == 0:
        try:
            if args.output.exists():
                raise FileExistsError(args.output)
            args.output.mkdir(parents=True)
        except BaseException as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    dist.barrier()
    from transformers import Trainer, TrainingArguments
    from trl.experimental.async_distillation import AsyncDistillationTrainer

    results = {}
    for case_name, (rank_rows, local_expected, global_expected) in cases().items():
        row = train_case(AsyncDistillationTrainer, Trainer, TrainingArguments, case_name, rank_rows,
                         local_expected, global_expected, rank, args.output / f"{case_name}-rank-{rank}")
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, row)
        if rank == 0:
            results[case_name] = {
                "rank_results": gathered,
                "rank_state_equal": all(item["post_step_state_sha256"] == gathered[0]["post_step_state_sha256"]
                                         for item in gathered),
                "max_loss_abs_error": max(item["loss_abs_error"] for item in gathered),
                "max_gradient_abs_error": max(item["max_gradient_abs_error"] for item in gathered),
                "max_sgd_parameter_abs_error": max(item["max_sgd_parameter_abs_error"] for item in gathered),
                "finite": all(item["finite"] for item in gathered),
            }
        dist.barrier()
    if rank == 0:
        summary = {
            "protocol_id": "trl-async-distillation-pr7249-full-trainer-ddp-v3", "arm": args.arm,
            "source_commit": commit, "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": __import__("transformers").__version__, "accelerate": __import__("accelerate").__version__},
            "backend": "gloo", "world_size": 2, "device": "cpu", "elapsed_seconds": time.monotonic() - started,
            "cases": results,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
