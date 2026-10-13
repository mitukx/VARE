#!/usr/bin/env python3
"""Check the frozen RVL/VARE parity result against protocol v5."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "protocols/rvl_trajectory_gradient_parity_v5.lock.json"
RUNNER = ROOT / "scripts/experiments/rvl_trajectory_gradient_parity_v5.py"
RESULT = ROOT / "outputs/rvl_trajectory_gradient_parity_v5.json"
AUDIT = ROOT / "outputs/rvl_trajectory_gradient_parity_v5.audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    result_data = RESULT.read_bytes()
    checks: dict[str, Any] = {}
    checks["protocol_status"] = lock.get("status") == "preregistered-before-run"
    checks["protocol_result_match"] = (
        result.get("protocol_id") == lock.get("protocol_id")
        and result.get("protocol_sha256") == sha256(LOCK)
    )
    checks["runner_hash_match"] = (
        lock.get("runner_sha256") == sha256(RUNNER)
        and result.get("runner_sha256") == lock.get("runner_sha256")
    )
    checks["vare_source_hashes"] = {
        path: sha256(ROOT / path) == expected
        and result.get("vare_source_files_sha256", {}).get(path) == expected
        for path, expected in lock["vare_source_files_sha256"].items()
    }
    checks["rvl_source_hashes"] = {
        path: result.get("rvl_source_files_sha256", {}).get(path) == expected
        for path, expected in lock["rvl_source_files_sha256"].items()
    }
    output = result["result"]
    tolerances = lock["numeric_tolerances"]
    checks["rvl_revision_match"] = result.get("rvl_revision") == lock.get("rvl_revision")
    checks["loss_tolerance"] = (
        output.get("status") == "completed"
        and output.get("absolute_loss_difference", float("inf"))
        <= tolerances["loss_abs_tolerance"]
    )
    checks["gradient_tolerance"] = (
        output.get("max_absolute_gradient_difference", float("inf"))
        <= tolerances["gradient_abs_tolerance"]
    )
    checks["negative_control"] = (
        output.get("prompt_merged_control_max_gradient_difference", 0.0)
        >= tolerances["negative_control_min_gradient_delta"]
    )
    checks["weight_sums"] = (
        abs(result["fixture"].get("batch_scaled_target_weight_sum", 0.0) - 1.0)
        <= 1e-12
        and all(
            abs(value - 1.0) <= 1e-12
            for value in result["fixture"].get("per_group_materialized_weight_sums", {}).values()
        )
    )
    checks["fixture_groups"] = (
        result["fixture"].get("sample_count") == lock["fixture"].get("total_samples")
        and len(result["fixture"].get("groups", [])) == len(lock["fixture"].get("groups", {}))
        and result["fixture"].get("prompt_id") == lock["fixture"].get("prompt_id")
    )
    checks["no_update_state_invariants"] = all(
        output.get(field) is True
        for field in (
            "model_parameters_and_buffers_unchanged",
            "optimizer_state_unchanged",
            "torch_cpu_rng_unchanged",
            "python_rng_unchanged",
            "module_modes_restored",
        )
    ) and output.get("parameter_update_performed") is False and output.get(
        "optimizer_step_calls_intercepted"
    ) == 1
    checks["gradient_finite_nonzero"] = (
        output.get("native_gradients_finite") is True
        and output.get("native_gradients_nonzero") is True
    )
    runtime = result.get("runtime", {})
    locked_runtime = lock.get("runtime", {})
    checks["runtime_match"] = (
        runtime.get("device") == "cpu"
        and runtime.get("python") == locked_runtime.get("python")
        and runtime.get("torch") == locked_runtime.get("torch")
        and runtime.get("threads") == locked_runtime.get("threads")
        and "macOS" in runtime.get("platform", "")
        and "arm64" in runtime.get("platform", "")
        and runtime.get("paid_compute") is False
    )
    checks["wall_cap"] = (
        output.get("wall_seconds", float("inf")) <= locked_runtime.get("max_wall_seconds", 0)
    )
    passed = all(
        value if isinstance(value, bool) else all(value.values())
        for value in checks.values()
    )
    return {
        "protocol_id": lock["protocol_id"],
        "decision": "pass" if passed else "fail",
        "checks": checks,
        "result_sha256": hashlib.sha256(result_data).hexdigest(),
        "limitation": "This checks the recorded result against the frozen protocol; rerunning the protocol is the numerical reproduction.",
    }


if __name__ == "__main__":
    report = audit()
    AUDIT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "audit": str(AUDIT)}))
    raise SystemExit(0 if report["decision"] == "pass" else 1)
