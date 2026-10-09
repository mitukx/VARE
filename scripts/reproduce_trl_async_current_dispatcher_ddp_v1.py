#!/usr/bin/env python3
"""Run the pinned AsyncGRPO collator, Accelerate dispatcher, Trainer, and DDP on CPU.

Launch two ranks with `python -m torch.distributed.run --standalone --nproc_per_node=2`.
The fixed fixture is a scalar token-local model, not a pretrained language model.
"""

from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType

import accelerate
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from transformers import Trainer, TrainingArguments
from transformers.utils import ModelOutput

from trl.experimental.async_grpo.async_grpo_trainer import AsyncGRPOTrainer, DataCollatorForRollout


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/trl-async-window-normalization-current-main-v1/run-4/summary.json"
PROTOCOL = ROOT / "protocols/trl_async_window_normalization_current_dispatcher_ddp_v1.lock.json"
SOURCE = Path(__import__(AsyncGRPOTrainer.__module__, fromlist=["__file__"]).__file__).resolve()
OLD = """        tokens_per_rank = (global_n_tokens / world_size).clamp(min=1.0)
        loss = loss / tokens_per_rank.to(torch.float32)
        # For DAPO, we would scale like this instead:
        # loss = loss / max(per_token_loss.size(0), 1)
        loss = loss / self.current_gradient_accumulation_steps
"""
NEW = """        if num_items_in_batch is None:
            num_items_in_batch = global_n_tokens * self.current_gradient_accumulation_steps
        normalizer = num_items_in_batch.clamp(min=1.0).to(torch.float32) / world_size
        loss = loss / normalizer
"""


@dataclass
class TokenOutput(ModelOutput):
    log_probs: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None


class TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, input_ids, **_kwargs):
        log_probs = self.theta * input_ids[:, 1:].to(torch.float32)
        return TokenOutput(log_probs=log_probs, entropy=torch.zeros_like(log_probs), aux_loss=torch.zeros((), device=input_ids.device))


class BatchStream(IterableDataset):
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


class EmptyBatchStream(IterableDataset):
    def __iter__(self):
        return iter(())


def sample(n_tokens: int, per_token_gradient: float, group_id: int) -> dict:
    return {
        "input_ids": [0] + [1] * n_tokens,
        "completion_mask": [0] + [1] * n_tokens,
        "old_log_probs": [0.0] * (n_tokens + 1),
        "advantage": -per_token_gradient,
        "group_id": group_id,
        "metrics": {},
    }


def extract_compute_loss(source: str):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AsyncGRPOTrainer")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "compute_loss")
    fn = copy.deepcopy(fn)
    fn.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    import trl.experimental.async_grpo.async_grpo_trainer as globals_module

    env = {name: getattr(globals_module, name) for name in dir(globals_module) if not name.startswith("__")}
    exec(compile(module, str(SOURCE), "exec"), env)
    return env["compute_loss"]


def collate_candidates():
    source = SOURCE.read_text()
    if source.count(OLD) != 1:
        raise SystemExit("pinned production loss normalization block was not unique")
    return extract_compute_loss(source.replace(OLD, NEW, 1))


def make_trainer(tmp_path: str, batches, accumulation_steps: int, candidate: bool):
    args = TrainingArguments(
        output_dir=tmp_path,
        use_cpu=True,
        report_to=[],
        gradient_accumulation_steps=accumulation_steps,
        per_device_train_batch_size=1,
        max_steps=1,
        num_train_epochs=1,
        learning_rate=0.1,
        weight_decay=0.0,
        warmup_steps=0,
        lr_scheduler_type="constant",
        max_grad_norm=0.0,
        logging_strategy="no",
        save_strategy="no",
        disable_tqdm=True,
        ddp_backend="gloo",
        accelerator_config={"split_batches": True, "dispatch_batches": True},
    )
    trainer = AsyncGRPOTrainer.__new__(AsyncGRPOTrainer)
    Trainer.__init__(trainer, model=TokenModel(), args=args)
    trainer.model_accepts_loss_kwargs = False
    trainer.aux_loss_enabled = False
    trainer.router_aux_loss_coef = 0.1
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer._metrics = {"train": defaultdict(list)}
    trainer._step_forward_tokens = 0.0
    trainer._step_trained_tokens = 0.0
    trainer._step_seq_len_weighted = 0.0
    trainer._step_samples = 0.0
    trainer._step_forward_s = 0.0
    trainer._last_forward_time_s = 0.0
    trainer._step_microbatches = 0
    trainer._current_train_step_time = 0.0
    trainer.rollout_worker = None
    trainer.weight_transfer = None
    trainer._trained_groups = set()
    trainer._groups_before_resume = 0
    trainer.log = Trainer.log.__get__(trainer)
    num_processes = trainer.accelerator.num_processes
    collator = DataCollatorForRollout(pad_token_id=0, num_processes=num_processes)
    source = BatchStream(batches) if trainer.accelerator.is_main_process else EmptyBatchStream()
    raw_loader = DataLoader(source, batch_size=1, collate_fn=collator, num_workers=0)
    loader = trainer.accelerator.prepare(raw_loader)
    trainer._vare_loader_type = type(loader).__name__
    trainer.get_train_dataloader = lambda: loader
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)

    window_counts, gradients_seen = [], []
    original_training_step = AsyncGRPOTrainer.training_step.__get__(trainer)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = original_training_step(model, inputs, num_items_in_batch)
        if self.model.theta.grad is not None:
            gradients_seen.append(float(self.model.theta.grad.detach()))
        return loss

    trainer.training_step = MethodType(training_step, trainer)
    if candidate:
        base_get_batch_samples = Trainer.get_batch_samples

        def get_batch_samples(self, epoch_iterator, num_batches, device):
            batch_samples, _ = base_get_batch_samples(self, epoch_iterator, num_batches, device)
            count = None
            if batch_samples:
                count = sum(batch["global_n_tokens"][0] for batch in batch_samples).to(device)
            window_counts.append(None if count is None else float(count))
            return batch_samples, count

        trainer.get_batch_samples = MethodType(get_batch_samples, trainer)
        trainer.compute_loss = MethodType(collate_candidates(), trainer)
    return trainer, window_counts, gradients_seen


def run_trainer(global_batches, accumulation_steps: int, candidate: bool):
    with tempfile.TemporaryDirectory(prefix="vare-trl-ddp-") as tmp:
        trainer, window_counts, gradients_seen = make_trainer(tmp, global_batches, accumulation_steps, candidate)
        result = trainer.train()
        model = trainer.accelerator.unwrap_model(trainer.model)
        local = {
            "global_step": int(result.global_step),
            "training_loss": float(result.training_loss),
            "gradient": gradients_seen[-1] if gradients_seen else None,
            "parameter_delta": float(model.theta.detach()),
            "window_counts_seen": window_counts,
            "loader_type": trainer._vare_loader_type,
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local)
        if any(item["global_step"] != 1 for item in gathered):
            raise RuntimeError(f"not every rank completed one update: {gathered}")
        if any(abs(item["parameter_delta"] - gathered[0]["parameter_delta"]) > 1e-7 for item in gathered):
            raise RuntimeError(f"DDP parameters diverged: {gathered}")
        return gathered[0], gathered


def evaluate_case(global_batches, accumulation_steps: int, candidate: bool):
    observed, ranks = run_trainer(global_batches, accumulation_steps, candidate)
    full_batches = [
        [
            [sample for batch in global_batches for sample in batch[rank]]
            for rank in range(dist.get_world_size())
        ]
    ]
    reference, reference_ranks = run_trainer(full_batches, 1, candidate)
    return {
        "observed": observed,
        "per_rank_observed": ranks,
        "full_batch_reference": reference,
        "per_rank_reference": reference_ranks,
        "gradient_abs_error": abs(observed["gradient"] - reference["gradient"]),
        "parameter_delta_abs_error": abs(observed["parameter_delta"] - reference["parameter_delta"]),
    }


def fixtures():
    # Two tokens per rank avoid the current loss's min-one denominator clamp on the first window.
    counterexample = [
        [[sample(2, 1.0, 1)], [sample(2, 1.0, 2)]],
        [[sample(18, 0.0, 3)], [sample(18, 0.0, 4)]],
    ]
    negative = [
        [[sample(2, 1.0, 1)], [sample(2, 1.0, 2)]],
        [[sample(18, 1.0, 3)], [sample(18, 1.0, 4)]],
    ]
    return {"counterexample": counterexample, "negative_control": negative}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cli = parser.parse_args()
    torch.set_num_threads(1)
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise SystemExit("launch exactly two torch.distributed ranks")
    protocol = json.loads(PROTOCOL.read_text())
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_hash != protocol["trl"]["sha256"]:
        raise SystemExit(f"TRL source hash mismatch: {source_hash}")
    source_commit = subprocess.check_output(["git", "-C", str(SOURCE.parents[3]), "rev-parse", "HEAD"], text=True).strip()
    if source_commit != protocol["trl"]["commit"]:
        raise SystemExit(f"expected TRL commit {protocol['trl']['commit']}, found {source_commit}")
    import transformers

    if (torch.__version__, transformers.__version__, accelerate.__version__) != (
        protocol["runtime"]["torch"], protocol["runtime"]["transformers"], protocol["runtime"]["accelerate"]
    ):
        raise SystemExit(f"runtime mismatch: {torch.__version__}, {transformers.__version__}, {accelerate.__version__}")

    outcomes = {}
    for name, batches in fixtures().items():
        accumulation_steps = 2
        outcomes[name] = {
            "base": evaluate_case(batches, accumulation_steps, candidate=False),
            "candidate": evaluate_case(batches, accumulation_steps, candidate=True),
        }
    passed = (
        outcomes["counterexample"]["base"]["parameter_delta_abs_error"] > 0.02
        and outcomes["negative_control"]["base"]["parameter_delta_abs_error"] <= 1e-7
        and all(
            outcomes[name]["candidate"]["parameter_delta_abs_error"] <= 1e-6
            and outcomes[name]["candidate"]["gradient_abs_error"] <= 1e-6
            for name in outcomes
        )
        and outcomes["counterexample"]["candidate"]["observed"]["window_counts_seen"] == [40.0]
        and all(
            arm_case[arm]["observed"]["loader_type"] == "DataLoaderDispatcher"
            for arm_case in outcomes.values()
            for arm in ("base", "candidate")
        )
    )
    report = {
        "study_id": protocol["study_id"],
        "trl_source_commit": source_commit,
        "trl_source_sha256": source_hash,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "world_size": dist.get_world_size(),
        "backend": dist.get_backend(),
        "outcomes": outcomes,
        "acceptance_passed": passed,
        "claim_boundary": "Executes current TRL rollout collator, Accelerate DataLoaderDispatcher, Transformers Trainer loop, DDP/Gloo, production compute_loss, and one CPU optimizer step. Uses a scalar synthetic model and bypasses the live async queue, rollout worker, vLLM, and pretrained-model evaluation.",
    }
    if dist.get_rank() == 0:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("frozen DDP acceptance criteria failed")


if __name__ == "__main__":
    main()
