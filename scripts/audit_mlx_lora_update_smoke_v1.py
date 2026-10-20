#!/usr/bin/env python3
"""Independently audit the retained MLX LoRA feasibility run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from run_mlx_lora_update_smoke_v1 import completed_training_steps, sha256

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/mlx_lora_update_smoke_v1.lock.json"
DATA = ROOT / "benchmarks/dev_sft_update_smoke_v1/train.jsonl"


def audit(run: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    log = (run / "training.log").read_text(encoding="utf-8")
    plain = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", log).replace("\r", "\n")
    rows = re.findall(r"^\s*(\d+)\s+([-+0-9.eE]+)", plain, re.M)
    step_ids = completed_training_steps(log)
    losses = [float(value) for _, value in rows]
    adapter = run / "adapter" / "adapters.safetensors"
    adapter_hash = sha256(adapter) if adapter.is_file() else None
    from safetensors.numpy import load_file

    tensors = load_file(str(adapter)) if adapter.is_file() else {}
    learned = sum(
        int((value != 0).sum())
        for name, value in tensors.items()
        if name.endswith(".lora_b")
    )
    checks = {
        "protocol_and_data_pins_match": (
            summary.get("protocol_id") == protocol["protocol_id"]
            and summary.get("dataset_sha256") == protocol["data"]["sha256"]
            and sha256(DATA) == protocol["data"]["sha256"]
        ),
        "metal_gpu_was_selected": (
            summary.get("host", {}).get("device") == "Device(gpu, 0)"
            and summary.get("host", {}).get("metal_available") is True
        ),
        "ten_optimizer_steps_are_logged": step_ids == list(range(1, 11)),
        "losses_are_finite_and_decrease_on_training_example": (
            len(losses) == 10
            and all(math.isfinite(loss) for loss in losses)
            and losses[-1] < losses[0]
        ),
        "adapter_bytes_match_summary": (
            adapter_hash is not None
            and summary.get("training", {})
            .get("adapter_files", {})
            .get("adapters.safetensors")
            == adapter_hash
        ),
        "learned_lora_b_weights_are_nonzero": learned > 0,
        "adapter_reload_generated_tokens": (
            summary.get("reload_inference", {}).get("output_tokens", 0) > 0
            and summary.get("reload_inference", {}).get("tokens_per_second", 0) > 0
        ),
        "recorded_resource_limits_pass": (
            summary.get("resources", {}).get("peak_mlx_bytes", 2**63)
            < protocol["hardware"]["maximum_memory_bytes"]
            and summary.get("resources", {}).get("max_rss_bytes", 2**63) < 8_000_000_000
        ),
    }
    return {
        "protocol_id": protocol["protocol_id"],
        "raw_summary_sha256": sha256(summary_path),
        "training_log_sha256": sha256(run / "training.log"),
        "adapter_sha256": adapter_hash,
        "optimizer_step_ids": step_ids,
        "training_losses": losses,
        "learned_lora_b_parameters": learned,
        "checks": checks,
        "status": "PASS_UPDATE_FEASIBILITY_ONLY"
        if all(checks.values())
        else "NON_PASS",
        "runner_original_status": summary.get("status"),
        "runner_parser_defect": (
            "The original runner searched for an 'Iter N: Train loss' format, but MLX-LM 0.32.2 "
            "emits a terminal progress table. This independent audit reads its numbered rows; "
            "raw summary and logs remain unchanged."
        ),
        "claim_limit": "One development example; training-loss fit and adapter reload do not measure held-out task success, SFT benefit, or RL improvement.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.run.resolve())
    target = args.output or args.run.parent / "independent_audit.json"
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": report["status"], "checks": report["checks"]}, sort_keys=True
        )
    )
    return 0 if report["status"] == "PASS_UPDATE_FEASIBILITY_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
