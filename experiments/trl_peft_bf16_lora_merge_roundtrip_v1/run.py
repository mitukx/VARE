#!/usr/bin/env python3
"""Pinned CPU reproducer for PEFT LoRA merge/unmerge numerical roundtrips."""
import argparse
import hashlib
import json
import os
import platform
import random
import resource
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open

MODEL_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 20261009

class TinyQProj(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.q_proj = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype)
        with torch.no_grad():
            self.q_proj.weight.copy_(weight)


def tensor_metrics(before, after):
    diff = after.float() - before.float()
    return {
        "changed_elements": int(torch.count_nonzero(after != before).item()),
        "total_elements": before.numel(),
        "max_abs_drift": float(diff.abs().max().item()),
        "l2_drift": float(torch.linalg.vector_norm(diff).item()),
        "relative_l2_drift": float((torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(before.float())).item()),
        "bitwise_equal": bool(torch.equal(before, after)),
    }


def build(weight, dtype, seed, zero_b=False):
    torch.manual_seed(seed)
    base = TinyQProj(weight.to(dtype))
    model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj"],
        bias="none", task_type=None, init_lora_weights=True,
    ))
    with torch.no_grad():
        layer = model.base_model.model.q_proj
        layer.lora_B["default"].weight.normal_(mean=0.0, std=0.02)
        if zero_b:
            layer.lora_B["default"].weight.zero_()
    return model


def run_arm(label, source_weight, dtype, safe_merge, zero_b, cycles, x):
    model = build(source_weight, dtype, SEED, zero_b)
    layer = model.base_model.model.q_proj
    before = layer.get_base_layer().weight.detach().clone()
    with torch.inference_mode():
        y0 = model.q_proj(x.to(dtype)).detach().float()
    checkpoints = {}
    started = time.perf_counter()
    for i in range(1, cycles + 1):
        model.merge_adapter(safe_merge=safe_merge)
        model.unmerge_adapter()
        if i in (1, 10, 100):
            now = layer.get_base_layer().weight.detach().clone()
            checkpoints[str(i)] = tensor_metrics(before, now)
    elapsed = time.perf_counter() - started
    after = layer.get_base_layer().weight.detach().clone()
    with torch.inference_mode():
        y1 = model.q_proj(x.to(dtype)).detach().float()
    result = {
        "arm": label,
        "dtype": str(dtype),
        "safe_merge": safe_merge,
        "zero_b": zero_b,
        "cycles": cycles,
        "checkpoints": checkpoints,
        "final": tensor_metrics(before, after),
        "fixed_input_output_max_abs_difference": float((y1-y0).abs().max().item()),
        "elapsed_seconds": elapsed,
    }
    del model
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-file', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    random.seed(SEED)
    started = time.perf_counter()
    path = Path(args.model_file)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if file_sha != "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe":
        raise RuntimeError(f"unexpected model shard sha256: {file_sha}")
    with safe_open(str(path), framework="pt", device="cpu") as f:
        weight = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
    if tuple(weight.shape) != (896, 896) or weight.dtype != torch.bfloat16:
        raise RuntimeError(f"unexpected tensor metadata: {weight.shape}, {weight.dtype}")
    gen = torch.Generator(device="cpu").manual_seed(SEED + 1)
    x = torch.randn((4, 896), generator=gen, dtype=torch.float32)
    arms = [
        ("bf16_unsafe_nonzero", torch.bfloat16, False, False, 100),
        ("bf16_zero_b_control", torch.bfloat16, False, True, 100),
        ("bf16_safe_nonzero", torch.bfloat16, True, False, 100),
        ("fp32_unsafe_nonzero", torch.float32, False, False, 100),
    ]
    results = []
    for name, dtype, safe, zero_b, cycles in arms:
        results.append(run_arm(name, weight, dtype, safe, zero_b, cycles, x))
    # Control for the synchronization-equivalent path where no merge mutates the base.
    functional = build(weight, torch.bfloat16, SEED, False)
    before = functional.base_model.model.q_proj.get_base_layer().weight.detach().clone()
    with torch.inference_mode():
        baseline_out = functional.q_proj(x.to(torch.bfloat16)).detach().clone()
        for _ in range(100):
            _ = functional.q_proj(x.to(torch.bfloat16))
        functional_out = functional.q_proj(x.to(torch.bfloat16)).detach().clone()
    after = functional.base_model.model.q_proj.get_base_layer().weight.detach().clone()
    results.append({
        "arm": "bf16_functional_no_merge_control",
        "cycles": 100,
        "final": tensor_metrics(before, after),
        "fixed_input_output_max_abs_difference": float((functional_out.float()-baseline_out.float()).abs().max().item()),
        "elapsed_seconds": None,
    })
    report = {
        "study_id": "trl_peft_bf16_lora_merge_roundtrip_v1",
        "source_model_revision": MODEL_REV,
        "model_shard_sha256": file_sha,
        "source_tensor": "model.layers.0.self_attn.q_proj.weight",
        "source_tensor_shape": list(weight.shape),
        "source_tensor_dtype": str(weight.dtype),
        "seed": SEED,
        "device": "cpu",
        "torch_version": torch.__version__,
        "peft_version": __import__('peft').__version__,
        "python": platform.python_version(),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds_total": time.perf_counter() - started,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
