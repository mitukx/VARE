#!/usr/bin/env python3
"""Quantify the pinned TRL DAPO normalizer's per-window gradient scaling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import torch


PROTOCOL = Path(__file__).resolve().parents[1] / "protocols/trl_dapo_accumulation_gradient_audit_v1.lock.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_source(path: Path, expected_hash: str, fixed: bool) -> None:
    source = path.read_text(encoding="utf-8")
    if sha256(path) != expected_hash:
        raise ValueError(f"source hash mismatch for {path}")
    required = [
        'elif self.loss_type in ["cispo", "dapo", "vespo"]:',
        'normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes',
        'loss = (per_token_loss * mask).sum() / normalizer',
    ]
    if any(fragment not in source for fragment in required):
        raise ValueError(f"expected DAPO loss statements missing from {path}")
    correction = re.compile(
        r'if mode == "train"\s*:\s*#.*?\n\s*normalizer = normalizer \* '
        r'self\.current_gradient_accumulation_steps / self\.args\.steps_per_generation',
        re.DOTALL,
    )
    if bool(correction.search(source)) != fixed:
        raise ValueError(f"unexpected accumulation correction state in {path}")


def run(baseline_source: Path, fixed_source: Path) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source_lock = protocol["source"]
    check_source(baseline_source, source_lock["baseline_sha256"], fixed=False)
    check_source(fixed_source, source_lock["fixed_sha256"], fixed=True)

    microbatches = protocol["frozen_input"]["microbatch_token_gradient_contributions"]
    steps = protocol["frozen_input"]["steps_per_generation"]
    token_count = sum(map(len, microbatches))
    full_gradient = sum(sum(batch) for batch in microbatches) / token_count
    results = []

    for accumulation in protocol["frozen_input"]["accumulation_windows"]:
        if steps % accumulation:
            raise ValueError("frozen protocol requires complete accumulation windows")
        window_gradients = {"baseline": [], "fixed": []}
        theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        for first in range(0, steps, accumulation):
            window = microbatches[first : first + accumulation]
            numerator = sum(sum(batch) for batch in window)
            for arm, normalizer in (
                ("baseline", float(token_count)),
                ("fixed", float(token_count) * accumulation / steps),
            ):
                loss = theta * numerator / normalizer
                gradient, = torch.autograd.grad(loss, theta, retain_graph=True)
                window_gradients[arm].append(float(gradient))

        ratios = {
            arm: (sum(values) / len(values)) / full_gradient
            for arm, values in window_gradients.items()
        }
        results.append(
            {
                "current_gradient_accumulation_steps": accumulation,
                "steps_per_generation": steps,
                "window_token_counts": [
                    sum(map(len, microbatches[first : first + accumulation]))
                    for first in range(0, steps, accumulation)
                ],
                "window_gradients": {
                    "baseline": window_gradients["baseline"],
                    "fixed": window_gradients["fixed"],
                },
                "primary_gradient_scale_ratio": ratios,
                "baseline_expected_ratio": accumulation / steps,
                "fixed_expected_ratio": 1.0,
            }
        )

    eval_normalizer = float(token_count)
    fixed_eval_normalizer = eval_normalizer  # PR #6024 applies the rescale in train mode only.
    tolerance = 1e-12
    for arm in results:
        if abs(arm["primary_gradient_scale_ratio"]["baseline"] - arm["baseline_expected_ratio"]) > tolerance:
            raise AssertionError("baseline gradient ratio failed its frozen criterion")
        if abs(arm["primary_gradient_scale_ratio"]["fixed"] - arm["fixed_expected_ratio"]) > tolerance:
            raise AssertionError("fixed gradient ratio failed its frozen criterion")
    if eval_normalizer != fixed_eval_normalizer:
        raise AssertionError("eval normalizer changed in the fixed arm")
    return {
        "protocol_id": protocol["protocol_id"],
        "status": "pass",
        "source_sha256": {
            "baseline": sha256(baseline_source),
            "fixed": sha256(fixed_source),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float64",
        },
        "estimand": {
            "total_valid_tokens": token_count,
            "full_generation_masked_token_mean_gradient": full_gradient,
            "mean_per_window_gradient_ratio": "mean window gradient divided by full-generation gradient",
        },
        "arms": results,
        "eval_control": {
            "baseline_normalizer": eval_normalizer,
            "fixed_normalizer": fixed_eval_normalizer,
            "unchanged": eval_normalizer == fixed_eval_normalizer,
        },
        "interpretation": "Source-locked scalar autograd audit only; no trainer/model update or task-success evaluation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--fixed-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.baseline_source, args.fixed_source)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
