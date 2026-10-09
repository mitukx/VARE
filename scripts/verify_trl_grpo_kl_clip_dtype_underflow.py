#!/usr/bin/env python3
"""Independent audit of the pinned production-loss underflow fixture results."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/trl_grpo_kl_clip_dtype_underflow_v1.lock.json"


def _finite(case: dict, key: str) -> bool:
    value = case.get(key)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def audit(base: dict, tiny: dict, control: dict) -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    errors = []
    if any(row.get("protocol_id") != lock["protocol_id"] for row in (base, tiny, control)):
        errors.append("protocol id mismatch")

    def cases_by_bias(result: dict) -> dict[bool, dict]:
        rows = result.get("cases", [])
        found = {row.get("bias_correction"): row for row in rows}
        if set(found) != {False, True}:
            errors.append("each run must include both bias-correction settings")
        return found

    base_cases = cases_by_bias(base)
    tiny_cases = cases_by_bias(tiny)
    control_cases = cases_by_bias(control)

    if base.get("status") != "complete" or tiny.get("status") != "complete" or control.get("status") != "complete":
        errors.append("primary runner did not complete")
    if not all(row.get("source", {}).get("clean") is True for row in
               [*base_cases.values(), *tiny_cases.values(), *control_cases.values()]):
        errors.append("one or more pinned source checkouts were not clean")

    base_revision = lock["source"]["base_revision"]
    candidate_revision = lock["source"]["candidate_revision"]
    if any(row.get("source", {}).get("revision") != base_revision for row in base_cases.values()):
        errors.append("base source revision mismatch")
    if any(row.get("source", {}).get("revision") != candidate_revision for row in
           [*tiny_cases.values(), *control_cases.values()]):
        errors.append("candidate source revision mismatch")

    for bias, row in base_cases.items():
        if row.get("execution") != "completed" or row.get("loss_finite") is not False:
            errors.append(f"base case with bias_correction={bias}: expected production loss overflow")

    for bias, row in tiny_cases.items():
        if row.get("config_validation", {}).get("status") != "accepted":
            errors.append("candidate config did not accept the positive tiny clip")
        if row.get("clip_value_in_working_dtype") != 0.0:
            errors.append("tiny positive clip did not cast to float16 zero")
        if not (_finite(row, "loss") and row.get("loss") == 0.0 and row.get("loss_gradient_theta") == 0.0):
            errors.append(f"tiny clip with bias_correction={bias}: expected silent zero KL and gradient")

    for bias, row in control_cases.items():
        if row.get("config_validation", {}).get("status") != "accepted":
            errors.append("candidate config did not accept the representable control clip")
        if row.get("clip_value_in_working_dtype") != 10.0:
            errors.append("representable control clip changed during float16 conversion")
        if not (_finite(row, "loss") and row.get("loss") > 0 and _finite(row, "loss_gradient_theta")):
            errors.append(f"representable control with bias_correction={bias}: loss or gradient is not finite/positive")
        if bias is False and not row.get("loss_gradient_theta", 0) < 0:
            errors.append("unbiased-correction control did not retain the expected negative loss gradient")

    corrected_control_gradient = control_cases.get(True, {}).get("loss_gradient_theta")
    expected_bias_corrected_gradient = -0.1 * 10.0
    findings = [
        "A positive clip can underflow to zero in float16 and be accepted, producing zero KL and zero gradient.",
        "With float16 and bias correction enabled, the representable clip control has a zero gradient despite a finite positive KL value; the uncanceled mathematical gradient is negative.",
    ]
    return {
        "protocol_id": lock["protocol_id"],
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "experiment_disposition": "mixed_nonpass" if corrected_control_gradient == 0.0 else "underflow_reproduced",
        "independent_calculations": {
            "torch_float16_1e_8": float(torch.tensor(1e-8, dtype=torch.float16).item()),
            "torch_float16_10": float(torch.tensor(10.0, dtype=torch.float16).item()),
            "expected_bias_corrected_gradient_at_clip_10": expected_bias_corrected_gradient,
            "observed_bias_corrected_gradient_at_clip_10": corrected_control_gradient,
            "bias_corrected_gradient_lost_to_float16_cancellation": corrected_control_gradient
            != expected_bias_corrected_gradient,
        },
        "findings": findings,
        "source_imports": "none; parses runner JSON only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tiny", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        json.loads(args.base.read_text(encoding="utf-8")),
        json.loads(args.tiny.read_text(encoding="utf-8")),
        json.loads(args.control.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
