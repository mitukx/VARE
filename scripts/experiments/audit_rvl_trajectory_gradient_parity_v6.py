#!/usr/bin/env python3
"""Independently check the source-current RVL/VARE parity result against protocol v6."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "protocols/rvl_trajectory_gradient_parity_v6.lock.json"
RUNNER = ROOT / "scripts/experiments/rvl_trajectory_gradient_parity_v6.py"
AUDITOR = Path(__file__)
RESULT = ROOT / "outputs/rvl_trajectory_gradient_parity_v6.json"
AUDIT = ROOT / "outputs/rvl_trajectory_gradient_parity_v6.audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_commit(revision: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", f"{revision}^{{commit}}"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def git_file_sha256(revision: str, relpath: str) -> str | None:
    try:
        content = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{revision}:{relpath}"],
            check=True, capture_output=True,
        ).stdout
        return hashlib.sha256(content).hexdigest()
    except (subprocess.CalledProcessError, OSError):
        return None


def audit() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    result_data = RESULT.read_bytes()
    checks: dict[str, Any] = {}
    locked_vare_revision = lock.get("vare_revision")
    resolved_vare_revision = (
        resolve_commit(locked_vare_revision) if isinstance(locked_vare_revision, str) else None
    )
    checks["protocol_status"] = lock.get("status") == "preregistered-before-run"
    checks["source_refresh_scope"] = (
        lock.get("source_refresh", {}).get("claim")
        == "Source-current evidence refresh only; same measurement, no new method or result claim."
        and result.get("source_refresh") == lock.get("source_refresh")
    )
    checks["vare_revision_match"] = lock.get("vare_revision") == result.get("vare_revision")
    checks["vare_revision_semantics"] = (
        lock.get("vare_revision_semantics")
        == result.get("vare_revision_semantics")
        == "source revision under test; checkout HEAD may be a later v6 branch or merged revision"
    )
    checks["vare_source_revision_resolves_exactly"] = resolved_vare_revision == locked_vare_revision
    checks["protocol_result_match"] = (
        result.get("protocol_id") == lock.get("protocol_id")
        and result.get("protocol_sha256") == sha256(LOCK)
    )
    checks["runner_hash_match"] = (
        lock.get("runner_sha256") == sha256(RUNNER)
        and result.get("runner_sha256") == lock.get("runner_sha256")
    )
    checks["auditor_hash_match"] = lock.get("auditor_sha256") == sha256(AUDITOR)
    checks["vare_source_hashes"] = {
        path: sha256(ROOT / path) == expected
        and result.get("vare_source_files_sha256", {}).get(path) == expected
        for path, expected in lock["vare_source_files_sha256"].items()
    }
    checks["vare_source_commit_hashes"] = {
        path: resolved_vare_revision == locked_vare_revision
        and git_file_sha256(locked_vare_revision, path) == expected
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
        "vare_revision": lock["vare_revision"],
        "vare_revision_semantics": lock["vare_revision_semantics"],
        "decision": "pass" if passed else "fail",
        "checks": checks,
        "result_sha256": hashlib.sha256(result_data).hexdigest(),
        "limitation": "This source-current refresh repeats the v5 measurement at the locked VARE revision; it introduces no new method or result claim. This audit checks the recorded result; rerunning the v6 runner is the numerical reproduction.",
    }


if __name__ == "__main__":
    report = audit()
    AUDIT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "audit": str(AUDIT)}))
    raise SystemExit(0 if report["decision"] == "pass" else 1)
