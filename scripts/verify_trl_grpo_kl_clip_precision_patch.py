#!/usr/bin/env python3
"""Compare the patched production-loss fixture with frozen PR-head controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/trl_grpo_kl_clip_precision_patch_v1.lock.json"
HEAD = "0aaea03f2fa449bc7a91f1973e7940da11da65da"
REL_TOL = 1e-3
ABS_GRAD_TOL = 0.05


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(row: dict, key: str) -> bool:
    value = row.get(key)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _same_float32(a: dict, b: dict, label: str, errors: list[str]) -> None:
    for key in ("loss", "gradient", "kl_metric"):
        if a.get(key) != b.get(key):
            errors.append(f"{label}: {key} changed from {a.get(key)!r} to {b.get(key)!r}")


def audit(root: Path) -> dict:
    errors = []
    expected_patch_sha = hashlib.sha256((root / "candidate.patch").read_bytes()).hexdigest()
    files = {
        "pr_f16_c10_no_bias": "pr-head-f16-c10-bias-off.json",
        "pr_f32_c10_bias": "pr-head-f32-c10-bias-on.json",
        "pr_f32_no_clip": "pr-head-f32-no-clip-bias-on.json",
        "patched_f16_tiny": "patched-f16-tiny-bias-on.json",
        "patched_f16_c10_no_bias": "patched-f16-c10-bias-off.json",
        "patched_f16_c10_bias": "patched-f16-c10-bias-on.json",
        "patched_f32_c10_bias": "patched-f32-c10-bias-on.json",
        "patched_f32_no_clip": "patched-f32-no-clip-bias-on.json",
    }
    rows = {name: _load(root / filename) for name, filename in files.items()}
    protocol = _load(LOCK)
    if any(row.get("protocol_id") != protocol["protocol_id"] for row in rows.values()):
        errors.append("protocol id mismatch")
    for name, row in rows.items():
        source = row.get("source", {})
        if source.get("revision") != HEAD:
            errors.append(f"{name}: source revision mismatch")
        should_be_patched = name.startswith("patched_")
        if source.get("patched") is not should_be_patched:
            errors.append(f"{name}: patched-source marker mismatch")
        if should_be_patched and source.get("patch_sha256") != expected_patch_sha:
            errors.append(f"{name}: patch hash mismatch")
        if row.get("status") != "complete":
            errors.append(f"{name}: primary method did not complete")

    tiny = rows["patched_f16_tiny"]
    if tiny.get("execution") != "raised" or tiny.get("exception_type") != "ValueError":
        errors.append("tiny float16 clip did not fail closed")
    if "representable" not in tiny.get("message", ""):
        errors.append("tiny clip error does not explain dtype representability")

    no_bias_base = rows["pr_f16_c10_no_bias"]
    no_bias_patch = rows["patched_f16_c10_no_bias"]
    for row, name in ((no_bias_base, "base no-bias"), (no_bias_patch, "patch no-bias")):
        if not (_finite(row, "loss") and _finite(row, "gradient") and _finite(row, "kl_metric")):
            errors.append(f"{name}: expected finite output and gradient")
    for key in ("loss", "gradient", "kl_metric"):
        old, new = no_bias_base[key], no_bias_patch[key]
        if not math.isclose(old, new, rel_tol=REL_TOL, abs_tol=1e-5):
            errors.append(f"float16 no-bias {key} exceeded the frozen relative tolerance")
    if no_bias_patch.get("gradient", 0) >= 0:
        errors.append("float16 no-bias gradient is not corrective")

    bias_patch = rows["patched_f16_c10_bias"]
    if not (_finite(bias_patch, "loss") and bias_patch.get("loss", 0) > 0 and _finite(bias_patch, "gradient")):
        errors.append("float16 bias-corrected patch output is not finite and positive")
    if not math.isclose(bias_patch.get("gradient", math.inf), -1.0, rel_tol=0, abs_tol=ABS_GRAD_TOL):
        errors.append("float16 bias-corrected gradient differs from analytic -1")

    for patch_key, base_key, label in (
        ("patched_f32_c10_bias", "pr_f32_c10_bias", "float32 clipped control"),
        ("patched_f32_no_clip", "pr_f32_no_clip", "float32 unclipped control"),
    ):
        _same_float32(rows[patch_key], rows[base_key], label, errors)

    return {
        "protocol_id": protocol["protocol_id"],
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "patch_sha256": expected_patch_sha,
        "cases": {name: {"execution": row.get("execution"), "loss": row.get("loss"), "gradient": row.get("gradient")} for name, row in rows.items()},
        "source_imports": "none; reads only the eight raw JSON outputs and patch bytes",
        "claim_boundary": "A one-token production-loss fixture; no model, optimizer, sequence-level IS, Liger, GPU, or downstream evaluation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
