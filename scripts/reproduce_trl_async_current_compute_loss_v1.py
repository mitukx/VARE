#!/usr/bin/env python3
"""Execute pinned TRL main's production compute_loss method on a CPU scalar model."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/trl_async_window_normalization_current_compute_loss_v1.lock.json"
SOURCE = ROOT / "results/trl-async-window-normalization-current-main-v1/source/async_grpo_trainer.py"
OUT_DIR = ROOT / "results/trl-async-window-normalization-current-main-v1/run-2"
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


class AcceleratorStub:
    num_processes = 1

    @staticmethod
    def reduce(value, reduction="sum"):
        return value

    @staticmethod
    def gather(value):
        return value


class TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, input_ids, **_kwargs):
        log_probs = self.theta * input_ids[:, 1:].to(torch.float32)
        return SimpleNamespace(log_probs=log_probs, entropy=torch.zeros_like(log_probs), aux_loss=self.theta * 0)


class TrainerStub:
    def __init__(self, steps: int):
        self.accelerator = AcceleratorStub()
        self.epsilon_low = 0.2
        self.epsilon_high = 0.2
        self.current_gradient_accumulation_steps = steps
        self.aux_loss_enabled = False
        self.router_aux_loss_coef = 0.1
        self._metrics = {"train": defaultdict(list)}
        self._step_forward_tokens = 0.0
        self._step_trained_tokens = 0.0
        self._step_seq_len_weighted = 0.0
        self._step_samples = 0.0
        self._step_forward_s = 0.0
        self._last_forward_time_s = 0.0


def nanmin(tensor):
    values = tensor[~torch.isnan(tensor)]
    return values.min() if values.numel() else torch.tensor(float("nan"), device=tensor.device)


def nanmax(tensor):
    values = tensor[~torch.isnan(tensor)]
    return values.max() if values.numel() else torch.tensor(float("nan"), device=tensor.device)


def method_from_source(source: str):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AsyncGRPOTrainer")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "compute_loss")
    fn = copy.deepcopy(fn)
    fn.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    env = {"torch": torch, "time": time, "nanmin": nanmin, "nanmax": nanmax}
    exec(compile(module, "pinned_async_grpo_trainer.py", "exec"), env)
    return env["compute_loss"]


def patch_source(source: str) -> str:
    if source.count(OLD) != 1:
        raise SystemExit("expected exactly one pinned loss-normalization block")
    return source.replace(OLD, NEW, 1)


def batch(samples: list[dict[str, float]]) -> dict[str, torch.Tensor]:
    ids, masks, advantages, positions = [], [], [], []
    for sample in samples:
        n = int(sample["active_tokens"])
        ids.extend([0] + [1] * n)
        masks.extend([0] + [1] * n)
        advantages.extend([0.0] + [-sample["per_token_gradient"]] * n)
        positions.extend(range(n + 1))
    total_tokens = sum(int(sample["active_tokens"]) for sample in samples)
    forward_tokens = len(ids)
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        "completion_mask": torch.tensor([masks], dtype=torch.long),
        "old_log_probs": torch.zeros((1, len(ids)), dtype=torch.float32),
        "position_ids": torch.tensor([positions], dtype=torch.long),
        "advantages": torch.tensor([advantages], dtype=torch.float32),
        "global_n_tokens": torch.tensor([float(total_tokens)]),
        "global_n_forward_tokens": torch.tensor([float(forward_tokens)]),
        "mean_seq_len": torch.tensor([float(forward_tokens / len(samples))]),
    }


def execute_window(method, samples, candidate: bool, direct_fallback=False):
    steps = len(samples)
    model = TokenModel()
    trainer = TrainerStub(steps)
    bound = MethodType(method, trainer)
    if direct_fallback:
        batches = [batch([samples[0]])]
        window_count = None
    else:
        batches = [batch([sample]) for sample in samples]
        window_count = torch.tensor(float(sum(int(s["active_tokens"]) for s in samples)))
    losses = []
    for inputs in batches:
        loss = bound(model, inputs, num_items_in_batch=window_count if candidate else None)
        losses.append(float(loss.detach()))
        loss.backward()
    gradient = float(model.theta.grad)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    return {"loss_sum": sum(losses), "gradient": gradient, "parameter_delta": float(model.theta.detach())}


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-7)


def main():
    protocol = json.loads(PROTOCOL.read_text())
    source_bytes = SOURCE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != protocol["source"]["sha256"]:
        raise SystemExit(f"source hash mismatch: {source_hash}")
    source = source_bytes.decode()
    candidate_source = patch_source(source)
    candidate_path = OUT_DIR / "async_grpo_trainer_candidate.py"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(candidate_source)
    base_method = method_from_source(source)
    candidate_method = method_from_source(candidate_source)

    fixtures = {
        "counterexample": protocol["fixture"]["counterexample"],
        "negative_control": protocol["fixture"]["negative_control"],
        "short_final_window": protocol["fixture"]["short_final_window"],
    }
    outcomes = {}
    for name, samples in fixtures.items():
        base_window = execute_window(base_method, samples, candidate=False)
        candidate_window = execute_window(candidate_method, samples, candidate=True)
        # A single full-batch reference uses the same sample rows, with K=1.
        reference_method = candidate_method
        reference = execute_window(reference_method, [samples[0] | {"active_tokens": sum(s["active_tokens"] for s in samples), "per_token_gradient": sum(s["active_tokens"] * s["per_token_gradient"] for s in samples) / max(sum(s["active_tokens"] for s in samples), 1)}], candidate=True, direct_fallback=True)
        outcomes[name] = {
            "base_accumulation": base_window,
            "candidate_accumulation": candidate_window,
            "full_batch_reference": reference,
            "base_gradient_abs_error": abs(base_window["gradient"] - reference["gradient"]),
            "candidate_gradient_abs_error": abs(candidate_window["gradient"] - reference["gradient"]),
            "base_delta_abs_error": abs(base_window["parameter_delta"] - reference["parameter_delta"]),
            "candidate_delta_abs_error": abs(candidate_window["parameter_delta"] - reference["parameter_delta"]),
        }

    # Candidate direct-call fallback for a 1-microbatch window preserves the old scale.
    one_sample = [fixtures["short_final_window"][0]]
    base_single = execute_window(base_method, one_sample, candidate=False)
    candidate_single = execute_window(candidate_method, one_sample, candidate=True, direct_fallback=True)
    outcomes["direct_call_fallback"] = {
        "base": base_single,
        "candidate": candidate_single,
        "gradient_abs_difference": abs(base_single["gradient"] - candidate_single["gradient"]),
    }

    ok = (
        outcomes["counterexample"]["base_gradient_abs_error"] > 0.1
        and all(outcomes[name]["candidate_gradient_abs_error"] <= 1e-7 for name in fixtures)
        and all(outcomes[name]["candidate_delta_abs_error"] <= 1e-7 for name in fixtures)
        and close(outcomes["direct_call_fallback"]["gradient_abs_difference"], 0.0)
        and close(outcomes["negative_control"]["base_gradient_abs_error"], 0.0)
    )
    result = {
        "study_id": protocol["study_id"],
        "source_commit": protocol["source"]["commit"],
        "source_sha256": source_hash,
        "candidate_source_sha256": hashlib.sha256(candidate_source.encode()).hexdigest(),
        "outcomes": outcomes,
        "acceptance_passed": ok,
        "claim_boundary": protocol["claim_boundary"],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit("frozen acceptance criteria failed")


if __name__ == "__main__":
    main()
