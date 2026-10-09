#!/usr/bin/env python3
"""Execute the frozen multi-token/sequence-level KL precision fixture on pinned TRL source."""
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

HEAD = "0aaea03f2fa449bc7a91f1973e7940da11da65da"
PATCH_SHA256 = "0f6c60de679c3c8b467b2bc8b77b01e06f890486cd567322940e5ce8a2137468"
SOURCE_PATH = "trl/trainer/grpo_trainer.py"
PROTOCOL_ID = "trl_grpo_kl_clip_precision_patch_v2"

class IdentityAccelerator:
    num_processes = 1
    @staticmethod
    def reduce(value, reduction="sum"):
        if reduction != "sum":
            raise ValueError(reduction)
        return value
    @staticmethod
    def gather(value):
        return value

class DummyModel(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=dtype))

def source_state(root: Path, patched: bool) -> dict:
    root = root.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != HEAD:
        raise ValueError(f"source revision mismatch: expected {HEAD}, got {revision}")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
    actual_patch = None
    if patched:
        changed = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", "HEAD"], text=True).splitlines()
        untracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
        patch = subprocess.check_output(["git", "-C", str(root), "diff", "--binary", "HEAD"])
        actual_patch = hashlib.sha256(patch).hexdigest()
        if changed != [SOURCE_PATH] or untracked or actual_patch != PATCH_SHA256:
            raise ValueError(f"patched checkout mismatch: changed={changed}, untracked={untracked}, patch={actual_patch}")
    elif dirty:
        raise ValueError("unmodified checkout is dirty")
    return {"revision": revision, "patched": patched, "patch_sha256": actual_patch,
            "trainer_sha256": hashlib.sha256((root / SOURCE_PATH).read_bytes()).hexdigest()}

def run_case(source_root: Path, *, patched: bool, dtype_name: str, is_level: str) -> dict:
    provenance = source_state(source_root, patched)
    root = str(source_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from trl.trainer.grpo_config import GRPOConfig
    from trl.trainer.grpo_trainer import GRPOTrainer

    dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_name]
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=dtype)
    # Active x=ref-policy values are [[20,4],[20,20,4]]. Padding is deliberately extreme
    # so a mask leak changes the sequence importance ratio or gradient.
    policy = torch.tensor([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]], dtype=dtype, requires_grad=True)
    old = torch.zeros((2, 3), dtype=dtype)
    x = torch.tensor([[20.0, 4.0, 80.0], [20.0, 20.0, 4.0]], dtype=dtype)
    ref = policy.detach() + x
    model = DummyModel(dtype)
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(
        use_bias_correction_kl=True, delta=None, kl_log_ratio_clip=10.0,
        steps_per_generation=1,
    )
    trainer.accelerator = IdentityAccelerator()
    trainer.aux_loss_enabled = False
    trainer.beta = 0.1
    trainer.current_gradient_accumulation_steps = 1
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.importance_sampling_level = is_level
    trainer.loss_type = "grpo"
    trainer.off_policy_mask_threshold = None
    trainer.top_entropy_quantile = 1.0
    trainer.use_vllm = False
    trainer.vllm_importance_sampling_correction = False
    trainer._entropy_bonus_enabled = False
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    zeros = torch.zeros((2, 3), dtype=dtype)
    def fake_logps(self, *args, **kwargs):
        return policy, zeros, None
    trainer._get_per_token_logps_and_entropies = MethodType(fake_logps, trainer)
    inputs = {
        "prompt_ids": torch.tensor([[1], [1]], dtype=torch.long),
        "prompt_mask": torch.ones((2, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.long),
        "completion_mask": mask,
        "advantages": torch.zeros((2,), dtype=dtype),
        "old_per_token_logps": old,
        "ref_per_token_logps": ref,
    }
    try:
        cfg = GRPOConfig(output_dir="/tmp/vare-kl-precision-v2", kl_log_ratio_clip=10.0)
        loss = GRPOTrainer._compute_loss(trainer, model, inputs)
        loss.backward()
        return {
            "status": "complete", "config": "accepted", "loss": float(loss.detach().float().item()),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "gradient": policy.grad.detach().float().tolist(),
            "gradient_finite": bool(torch.isfinite(policy.grad).all().item()),
            "kl_metric": float(trainer._metrics["train"]["kl"][-1]),
        }
    except Exception as exc:
        return {"status": "error", "exception_type": type(exc).__name__, "message": str(exc)}

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--patched", action="store_true")
    p.add_argument("--dtype", required=True, choices=("float16", "float32"))
    p.add_argument("--importance-sampling-level", required=True, choices=("token", "sequence"))
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = {"protocol_id": PROTOCOL_ID, "source": source_state(a.source_root, a.patched),
              "dtype": a.dtype, "importance_sampling_level": a.importance_sampling_level,
              **run_case(a.source_root, patched=a.patched, dtype_name=a.dtype, is_level=a.importance_sampling_level)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 1

if __name__ == "__main__":
    raise SystemExit(main())
