#!/usr/bin/env python3
"""Exercise pinned TRL AsyncGRPOTrainer.compute_loss on a finite CPU fixture."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/trl_async_grpo_truncated_support_v1.lock.json"


class IdentityAccelerator:
    num_processes = 1

    @staticmethod
    def reduce(value, reduction="sum"):
        if reduction != "sum":
            raise ValueError(f"unsupported reduction: {reduction}")
        return value

    @staticmethod
    def gather(value):
        return value


class ThreeActionModel(torch.nn.Module):
    """Differentiable causal-LM output fixture with the exact frozen raw policy."""

    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.base_logits = torch.log(torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64))

    def forward(self, input_ids, position_ids, labels, fused_lm_head=True, **kwargs):
        if not fused_lm_head:
            raise ValueError("production loss call must request fused_lm_head")
        logits = self.base_logits.to(input_ids.device).clone()
        logits[2] = logits[2] + self.theta
        logp = torch.log_softmax(logits, dim=-1)
        action_ids = input_ids[:, 1:]
        log_probs = logp[action_ids]
        probabilities = logp.exp()
        entropy_value = -(probabilities * logp).sum()
        entropy = entropy_value.expand_as(log_probs)
        return SimpleNamespace(log_probs=log_probs, entropy=entropy)


def _check_sources(source_root: Path, lock: dict) -> tuple[str, dict[str, str]]:
    source_root = source_root.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    expected_revision = lock["source"]["revision"]
    if head != expected_revision:
        raise ValueError(f"TRL revision mismatch: expected {expected_revision}, got {head}")
    actual_hashes = {}
    for relative, expected_hash in lock["source"]["files"].items():
        path = source_root / relative
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        actual_hashes[relative] = actual_hash
        if actual_hash != expected_hash:
            raise ValueError(f"source hash mismatch for {relative}: {actual_hash}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    return head, actual_hashes


def _one_action_loss(AsyncGRPOTrainer, action: int, q_probability: float, advantage: float):
    model = ThreeActionModel()
    trainer = AsyncGRPOTrainer.__new__(AsyncGRPOTrainer)
    trainer.aux_loss_enabled = False
    trainer.epsilon_low = 0.25
    trainer.epsilon_high = 0.25
    trainer.accelerator = IdentityAccelerator()
    trainer.current_gradient_accumulation_steps = 1
    trainer._metrics = {"train": defaultdict(list)}
    trainer._step_forward_tokens = 0.0
    trainer._step_trained_tokens = 0.0
    trainer._step_seq_len_weighted = 0.0
    trainer._step_samples = 0.0
    trainer._step_forward_s = 0.0

    inputs = {
        "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        "input_ids": torch.tensor([3, action], dtype=torch.long),
        "completion_mask": torch.tensor([0, 1], dtype=torch.long),
        "old_log_probs": torch.tensor([0.0, math.log(q_probability)], dtype=torch.float64),
        "position_ids": torch.tensor([0, 1], dtype=torch.long),
        "advantages": torch.tensor([0.0, advantage], dtype=torch.float64),
        "global_n_tokens": torch.tensor([1.0]),
        "global_n_forward_tokens": torch.tensor([2.0]),
        "mean_seq_len": torch.tensor([2.0]),
    }
    loss = AsyncGRPOTrainer.compute_loss(trainer, model, inputs)
    loss.backward()
    ratio = trainer._metrics["train"]["ratio"][-1][0]
    return {
        "action": action,
        "q_probability": q_probability,
        "advantage": advantage,
        "loss": float(loss.detach()),
        "loss_gradient_theta": float(model.theta.grad),
        "update_direction_theta": float(-model.theta.grad),
        "production_ratio_metric": float(ratio),
        "production_kl_metric": float(trainer._metrics["train"]["kl"][-1][0]),
        "production_clip_region_metric": float(trainer._metrics["train"]["clip_ratio/region_mean"][-1][0]),
    }


def run(source_root: Path) -> dict:
    lock = json.loads(LOCK.read_text())
    head, source_hashes = _check_sources(source_root, lock)
    from trl.experimental.async_grpo.async_grpo_trainer import AsyncGRPOTrainer

    p = [0.5, 0.3, 0.2]
    q_cases = {
        "identity_control": [0.5, 0.3, 0.2],
        "top_p_truncation": [0.625, 0.375, 0.0],
    }
    advantages = [0.0, 0.0, 1.0]
    results = {}
    for name, q in q_cases.items():
        per_action = []
        for action, q_probability in enumerate(q):
            if q_probability == 0.0:
                continue
            per_action.append(
                _one_action_loss(
                    AsyncGRPOTrainer,
                    action,
                    q_probability,
                    advantages[action],
                )
            )
        expected_ratio = sum(item["q_probability"] * item["production_ratio_metric"] for item in per_action)
        expected_direction = sum(item["q_probability"] * item["update_direction_theta"] for item in per_action)
        results[name] = {
            "support": [item["action"] for item in per_action],
            "retained_raw_mass": sum(p[action] for action in [item["action"] for item in per_action]),
            "per_action": per_action,
            "q_weighted_ratio": expected_ratio,
            "q_weighted_update_direction_theta": expected_direction,
        }

    raw_target_gradient = p[2] * advantages[2] * (1.0 - p[2])
    checks = {
        "identity_ratios_equal_one": abs(results["identity_control"]["q_weighted_ratio"] - 1.0) <= 1e-6,
        "identity_update_matches_raw_target": abs(
            results["identity_control"]["q_weighted_update_direction_theta"] - raw_target_gradient
        ) <= 1e-6,
        "truncated_ratio_equals_retained_mass": abs(
            results["top_p_truncation"]["q_weighted_ratio"] - 0.8
        ) <= 1e-6,
        "truncated_unsupported_reward_has_zero_update": abs(
            results["top_p_truncation"]["q_weighted_update_direction_theta"]
        ) <= 1e-6,
        "raw_target_gradient_is_four_twenty_fifths": abs(raw_target_gradient - 4 / 25) <= 1e-6,
        "no_cuda_initialized": not torch.cuda.is_initialized(),
    }
    return {
        "protocol_id": lock["protocol_id"],
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "source_revision": head,
        "source_hashes": source_hashes,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "cuda_initialized": torch.cuda.is_initialized(),
        },
        "frozen_policy": {"p": p, "advantage": advantages},
        "raw_target_update_direction_theta": raw_target_gradient,
        "cases": results,
        "claim_boundary": "Exact finite-support production loss-method execution with controlled q outputs; no vLLM runtime, full trainer initialization, optimizer step, model capability, or upstream defect claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="clean TRL checkout at locked revision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
