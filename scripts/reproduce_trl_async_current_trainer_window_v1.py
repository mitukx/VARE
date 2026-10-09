#!/usr/bin/env python3
"""Exercise current TRL AsyncGRPO compute_loss through Transformers Trainer.train().

Run from a clean checkout of the frozen TRL commit with compatible dependencies,
for example: PYTHONPATH=/path/to/transformers-5.19:/path/to/trl python this_script.py
The script uses only a deterministic scalar CPU model and the production rollout
collator. It does not contact a rollout server or train a pretrained model.
"""

from __future__ import annotations

import ast
import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import accelerate
from transformers import Trainer, TrainingArguments

from trl.experimental.async_grpo.async_grpo_trainer import AsyncGRPOTrainer, DataCollatorForRollout


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/trl-async-window-normalization-current-main-v1/run-3/summary.json"
PROTOCOL = ROOT / "protocols/trl_async_window_normalization_current_trainer_v1.lock.json"
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


class TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, input_ids, **_kwargs):
        log_probs = self.theta * input_ids[:, 1:].to(torch.float32)
        return SimpleNamespace(log_probs=log_probs, entropy=torch.zeros_like(log_probs), aux_loss=self.theta * 0)


def make_sample(n_tokens: int, per_token_gradient: float, group_id: int) -> dict:
    return {
        "input_ids": [0] + [1] * n_tokens,
        "completion_mask": [0] + [1] * n_tokens,
        "old_log_probs": [0.0] * (n_tokens + 1),
        "advantage": -per_token_gradient,
        "group_id": group_id,
        "metrics": {},
    }


def collate(samples: list[dict]) -> dict:
    return DataCollatorForRollout(pad_token_id=0, num_processes=1)([[samples]])


def extract_compute_loss(source: str):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AsyncGRPOTrainer")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "compute_loss")
    fn = copy.deepcopy(fn)
    fn.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    import trl.experimental.async_grpo.async_grpo_trainer as module_globals

    env = {name: getattr(module_globals, name) for name in dir(module_globals) if not name.startswith("__")}
    exec(compile(module, str(SOURCE), "exec"), env)
    return env["compute_loss"]


def candidate_compute_loss(source: str):
    if source.count(OLD) != 1:
        raise SystemExit("pinned production loss normalization block was not unique")
    return extract_compute_loss(source.replace(OLD, NEW, 1))


def initialize_trainer(tmp_path: str, batches: list[dict], accumulation_steps: int, candidate: bool):
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
    trainer.get_train_dataloader = lambda: torch.utils.data.DataLoader(batches, batch_size=None)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    window_counts = []
    gradients_seen = []
    original_training_step = AsyncGRPOTrainer.training_step.__get__(trainer)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = original_training_step(model, inputs, num_items_in_batch)
        if self.model.theta.grad is not None:
            gradients_seen.append(float(self.model.theta.grad.detach()))
        return loss

    trainer.training_step = MethodType(training_step, trainer)
    if candidate:
        original_get_batch_samples = Trainer.get_batch_samples

        def get_batch_samples(self, epoch_iterator, num_batches, device):
            batch_samples, _ = original_get_batch_samples(self, epoch_iterator, num_batches, device)
            count = None
            if batch_samples:
                count = sum(batch["global_n_tokens"][0] for batch in batch_samples).to(device)
            window_counts.append(None if count is None else float(count))
            return batch_samples, count

        trainer.get_batch_samples = MethodType(get_batch_samples, trainer)
        trainer.compute_loss = MethodType(candidate_compute_loss(SOURCE.read_text()), trainer)
    return trainer, window_counts, gradients_seen


def run_trainer(batches: list[dict], accumulation_steps: int, candidate: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="vare-trl-trainer-") as tmp:
        trainer, window_counts, gradients_seen = initialize_trainer(tmp, batches, accumulation_steps, candidate)
        result = trainer.train()
        return {
            "global_step": int(result.global_step),
            "training_loss": float(result.training_loss),
            "gradient": gradients_seen[-1] if gradients_seen else None,
            "parameter_delta": float(trainer.model.theta.detach()),
            "window_counts_seen": window_counts,
        }


def evaluate_case(samples: list[dict], candidate: bool) -> dict:
    # The short-final-window case intentionally keeps the configured value at 2
    # while exposing a one-batch dataset; Trainer then sets the actual window to 1.
    accumulation = 2
    split_batches = [collate([sample]) for sample in samples]
    full_batch = [collate(samples)]
    observed = run_trainer(split_batches, accumulation, candidate)
    reference = run_trainer(full_batch, 1, candidate)
    return {
        "observed": observed,
        "full_batch_reference": reference,
        "gradient_abs_error": abs(observed["gradient"] - reference["gradient"]),
        "parameter_delta_abs_error": abs(observed["parameter_delta"] - reference["parameter_delta"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args_cli = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(PROTOCOL.read_text())
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_hash != protocol["trl"]["source_sha256"]:
        raise SystemExit(f"TRL source hash mismatch: {SOURCE} -> {source_hash}")
    source_root = SOURCE.parents[3]
    source_commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if source_commit != protocol["trl"]["commit"]:
        raise SystemExit(f"expected TRL commit {protocol['trl']['commit']}, found {source_commit}")
    if torch.__version__ != protocol["runtime"]["torch"]:
        raise SystemExit(f"expected torch {protocol['runtime']['torch']}, found {torch.__version__}")
    import transformers

    if transformers.__version__ != protocol["runtime"]["transformers"]:
        raise SystemExit(f"expected transformers {protocol['runtime']['transformers']}, found {transformers.__version__}")
    if accelerate.__version__ != protocol["runtime"]["accelerate"]:
        raise SystemExit(f"expected accelerate {protocol['runtime']['accelerate']}, found {accelerate.__version__}")

    specs = {
        "counterexample": [make_sample(1, 1.0, 1), make_sample(9, 0.0, 2)],
        "negative_control": [make_sample(1, 1.0, 1), make_sample(9, 1.0, 2)],
        "short_final_window": [make_sample(1, 1.0, 1)],
    }
    outcomes = {}
    for name, samples in specs.items():
        outcomes[name] = {
            "base": evaluate_case(samples, candidate=False),
            "candidate": evaluate_case(samples, candidate=True),
        }
    passed = (
        outcomes["counterexample"]["base"]["parameter_delta_abs_error"] > 0.02
        and outcomes["negative_control"]["base"]["parameter_delta_abs_error"] <= 1e-7
        and all(
            outcomes[name]["candidate"]["parameter_delta_abs_error"] <= 1e-6
            and outcomes[name]["candidate"]["gradient_abs_error"] <= 1e-6
            for name in specs
        )
        and all(outcomes[name][arm]["observed"]["global_step"] == 1 for name in specs for arm in ("base", "candidate"))
        and outcomes["counterexample"]["candidate"]["observed"]["window_counts_seen"] == [10.0]
        and outcomes["short_final_window"]["candidate"]["observed"]["window_counts_seen"] == [1.0]
    )
    report = {
        "study_id": protocol["study_id"],
        "trl_source_commit": protocol["trl"]["commit"],
        "trl_source_path": str(SOURCE),
        "trl_source_sha256": source_hash,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "outcomes": outcomes,
        "acceptance_passed": passed,
        "claim_boundary": "The current Transformers Trainer optimizer loop and TRL production collator/loss run on CPU. No distributed reducer, async queue/worker, pretrained model, or downstream task is included.",
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("frozen acceptance criteria failed")


if __name__ == "__main__":
    main()
