#!/usr/bin/env python3
"""Run pinned GRPOTrainer._compute_loss controls for a local KL-clip precision patch."""
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
LOCK = ROOT / "protocols/trl_grpo_kl_clip_precision_patch_v1.lock.json"
CANDIDATE_REVISION = "0aaea03f2fa449bc7a91f1973e7940da11da65da"
SOURCE_PATH = "trl/trainer/grpo_trainer.py"


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
    def __init__(self, dtype):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=dtype))


def source_state(source_root: Path, patched: bool) -> dict:
    source_root = source_root.resolve()
    head = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if head != CANDIDATE_REVISION:
        raise ValueError(f"candidate revision mismatch: expected {CANDIDATE_REVISION}, got {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    if patched:
        changed = subprocess.check_output(
            ["git", "-C", str(source_root), "diff", "--name-only", "HEAD"], text=True
        ).splitlines()
        untracked = subprocess.check_output(
            ["git", "-C", str(source_root), "ls-files", "--others", "--exclude-standard"], text=True
        ).splitlines()
        if changed != [SOURCE_PATH] or untracked:
            raise ValueError(f"patched checkout must change only {SOURCE_PATH}; changed={changed}, untracked={untracked}")
        patch_bytes = subprocess.check_output(["git", "-C", str(source_root), "diff", "--binary", "HEAD"])
        patch_hash = hashlib.sha256(patch_bytes).hexdigest()
    else:
        if dirty:
            raise ValueError("unmodified comparison checkout is dirty")
        patch_hash = None
    return {
        "revision": head,
        "patched": patched,
        "patch_sha256": patch_hash,
        "grpo_trainer_sha256": hashlib.sha256((source_root / SOURCE_PATH).read_bytes()).hexdigest(),
    }


def run_case(source_root: Path, *, patched: bool, dtype_name: str, clip: float | None, bias_correction: bool) -> dict:
    dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_name]
    provenance = source_state(source_root, patched)
    resolved = str(source_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

    from trl.trainer.grpo_config import GRPOConfig
    from trl.trainer.grpo_trainer import GRPOTrainer

    if clip is not None:
        try:
            GRPOConfig(output_dir="/tmp/vare-kl-clip-patch", kl_log_ratio_clip=clip)
            config = {"status": "accepted"}
        except Exception as exc:
            config = {"status": "raised", "exception_type": type(exc).__name__, "message": str(exc)}
    else:
        config = None

    model = DummyModel(dtype)
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

    policy_logps = torch.zeros((1, 1), dtype=dtype, requires_grad=True)
    entropies = torch.zeros((1, 1), dtype=dtype)

    def fake_model_logps(self, *args, **kwargs):
        return policy_logps, entropies, None

    trainer._get_per_token_logps_and_entropies = MethodType(fake_model_logps, trainer)
    inputs = {
        "prompt_ids": torch.tensor([[1]], dtype=torch.long),
        "prompt_mask": torch.tensor([[1]], dtype=torch.long),
        "completion_ids": torch.tensor([[2]], dtype=torch.long),
        "completion_mask": torch.tensor([[1]], dtype=torch.long),
        "advantages": torch.zeros((1,), dtype=dtype),
        "old_per_token_logps": torch.zeros((1, 1), dtype=dtype),
        "ref_per_token_logps": torch.full((1, 1), 20.0, dtype=dtype),
    }
    try:
        loss = GRPOTrainer._compute_loss(trainer, model, inputs)
        loss.backward()
        loss_value = float(loss.detach().float().item())
        grad_value = float(policy_logps.grad.float().item())
        outcome = {
            "execution": "completed",
            "loss": loss_value,
            "loss_finite": math.isfinite(loss_value),
            "gradient": grad_value,
            "gradient_finite": math.isfinite(grad_value),
            "kl_metric": float(trainer._metrics["train"]["kl"][-1]),
        }
    except Exception as exc:
        outcome = {"execution": "raised", "exception_type": type(exc).__name__, "message": str(exc)}

    return {
        "source": provenance,
        "config": config,
        "dtype": dtype_name,
        "clip": clip,
        "bias_correction": bias_correction,
        "beta": 0.1,
        "kl_log_ratio": 20.0,
        "importance_ratio": 1.0,
        "advantage": 0.0,
        **outcome,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--patched", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "float32"), required=True)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--bias-correction", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_case(
        args.source_root,
        patched=args.patched,
        dtype_name=args.dtype,
        clip=args.clip,
        bias_correction=args.bias_correction,
    )
    result["protocol_id"] = json.loads(LOCK.read_text(encoding="utf-8"))["protocol_id"]
    result["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
