#!/usr/bin/env python3
"""Invoke pinned TRL GRPOTrainer._compute_loss on a CPU half-precision fixture."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/trl_grpo_kl_clip_dtype_underflow_v1.lock.json"
REVISIONS = {
    "base": "2b0d16b7839732f0b652ea7dfd4f492f00e15a48",
    "candidate": "0aaea03f2fa449bc7a91f1973e7940da11da65da",
}


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


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float16))


def _source_state(source_root: Path, expected_revision: str) -> dict:
    head = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != expected_revision:
        raise ValueError(f"source revision mismatch: expected {expected_revision}, got {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    if dirty:
        raise ValueError("source checkout is dirty")
    paths = ("trl/trainer/grpo_config.py", "trl/trainer/grpo_trainer.py")
    hashes = {}
    for relative in paths:
        hashes[relative] = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
    return {"revision": head, "clean": True, "files_sha256": hashes}


def _call_loss(source_root: Path, revision_name: str, clip: float | None, bias_correction: bool) -> dict:
    expected = REVISIONS[revision_name]
    state = _source_state(source_root.resolve(), expected)
    source = str(source_root.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)

    from trl.trainer.grpo_trainer import GRPOTrainer

    model = DummyModel()
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(
        use_bias_correction_kl=bias_correction,
        delta=None,
        kl_log_ratio_clip=clip,
        steps_per_generation=1,
    )
    trainer.accelerator = IdentityAccelerator()
    trainer.aux_loss_enabled = False
    trainer.beta = 0.1
    trainer.current_gradient_accumulation_steps = 1
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.2
    trainer.importance_sampling_level = "token"
    trainer.loss_type = "grpo"
    trainer.off_policy_mask_threshold = None
    trainer.top_entropy_quantile = 1.0
    trainer.use_vllm = False
    trainer.vllm_importance_sampling_correction = False
    trainer._entropy_bonus_enabled = False
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    policy_logps = torch.zeros((1, 1), dtype=torch.float16, requires_grad=True)
    entropies = torch.zeros((1, 1), dtype=torch.float16)

    def fake_model_logps(self, *args, **kwargs):
        return policy_logps, entropies, None

    trainer._get_per_token_logps_and_entropies = MethodType(fake_model_logps, trainer)
    inputs = {
        "prompt_ids": torch.tensor([[1]], dtype=torch.long),
        "prompt_mask": torch.tensor([[1]], dtype=torch.long),
        "completion_ids": torch.tensor([[2]], dtype=torch.long),
        "completion_mask": torch.tensor([[1]], dtype=torch.long),
        "advantages": torch.zeros((1,), dtype=torch.float16),
        "old_per_token_logps": torch.zeros((1, 1), dtype=torch.float16),
        "ref_per_token_logps": torch.full((1, 1), 20.0, dtype=torch.float16),
    }

    try:
        loss = GRPOTrainer._compute_loss(trainer, model, inputs)
        loss.backward()
        loss_value = float(loss.detach().float().item())
        gradient = float(policy_logps.grad.float().item())
        returned = {
            "execution": "completed",
            "loss": loss_value,
            "loss_finite": math.isfinite(loss_value),
            "loss_gradient_theta": gradient,
            "gradient_finite": math.isfinite(gradient),
            "production_kl_metric": float(trainer._metrics["train"]["kl"][-1]),
        }
    except Exception as exc:  # preserve an implementation-side validation/error as a result
        returned = {"execution": "raised", "exception_type": type(exc).__name__, "exception": str(exc)}

    return {
        "source": state,
        "revision_name": revision_name,
        "clip": clip,
        "clip_value_in_working_dtype": (
            float(torch.tensor(clip, dtype=torch.float16).item()) if clip is not None else None
        ),
        "bias_correction": bias_correction,
        "dtype": "float16",
        "kl_log_ratio": 20.0,
        "beta": 0.1,
        "advantage": 0.0,
        "importance_ratio": 1.0,
        "config_validation": config_outcome,
        **returned,
    }


def run(source_root: Path, revision_name: str, clip: float | None) -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    cases = []
    for bias_correction in (False, True):
        cases.append(_call_loss(source_root, revision_name, clip, bias_correction))
    return {
        "protocol_id": lock["protocol_id"],
        "status": "complete",
        "cases": cases,
        "claim_boundary": "Exact production GRPOTrainer._compute_loss execution on a one-token CPU float16 fixture; no dataset, pretrained model, GPU, Liger kernel, optimizer update, or task evaluation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--revision-name", choices=tuple(REVISIONS), required=True)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_root, args.revision_name, args.clip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
