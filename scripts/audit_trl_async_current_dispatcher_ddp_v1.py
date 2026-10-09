#!/usr/bin/env python3
"""Independently audit the retained two-rank AsyncGRPO Trainer result using stdlib only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/trl_async_window_normalization_current_dispatcher_ddp_v1.lock.json"
SUMMARY = ROOT / "results/trl-async-window-normalization-current-main-v1/run-4/summary.json"
SOURCE = Path("/tmp/trl-current-pr7249-audit/trl/experimental/async_grpo/async_grpo_trainer.py")
RUNNER = ROOT / "scripts/reproduce_trl_async_current_dispatcher_ddp_v1.py"
OUT = SUMMARY.parent / "independent-audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    lock, summary = json.loads(LOCK.read_text()), json.loads(SUMMARY.read_text())
    checks = {}
    checks["locked_runner_hash"] = sha256(RUNNER) == lock["runner"]["sha256"]
    checks["pinned_trl_source_hash"] = sha256(SOURCE) == lock["trl"]["sha256"] == summary["trl_source_sha256"]
    checks["pinned_commit"] = summary["trl_source_commit"] == lock["trl"]["commit"]
    checks["runtime_and_topology"] = (
        summary["torch"], summary["transformers"], summary["accelerate"], summary["world_size"], summary["backend"]
    ) == ("2.9.1", "5.19.0", "1.12.0", 2, "gloo")
    checks["runner_acceptance"] = summary["acceptance_passed"] is True
    checks["case_set"] = set(summary["outcomes"]) == {"counterexample", "negative_control"}

    counter = summary["outcomes"]["counterexample"]
    negative = summary["outcomes"]["negative_control"]
    checks["counterexample_base_gradient"] = close(counter["base"]["observed"]["gradient"], 0.5)
    checks["counterexample_reference_gradient"] = close(counter["base"]["full_batch_reference"]["gradient"], 0.1)
    checks["counterexample_base_update_error"] = close(counter["base"]["parameter_delta_abs_error"], 0.04)
    checks["counterexample_candidate_gradient"] = close(counter["candidate"]["observed"]["gradient"], 0.1)
    checks["counterexample_candidate_update_error"] = close(counter["candidate"]["parameter_delta_abs_error"], 0.0)
    checks["candidate_window_count"] = counter["candidate"]["observed"]["window_counts_seen"] == [40.0]
    checks["negative_control_base"] = close(negative["base"]["parameter_delta_abs_error"], 0.0)
    checks["negative_control_candidate"] = close(negative["candidate"]["parameter_delta_abs_error"], 0.0)
    checks["all_arms_dispatcher"] = all(
        case[arm][field]["loader_type"] == "DataLoaderDispatcher"
        for case in summary["outcomes"].values()
        for arm in ("base", "candidate")
        for field in ("observed", "full_batch_reference")
    )
    checks["all_rank_parameters_agree"] = all(
        all(abs(rank["parameter_delta"] - case[arm]["observed"]["parameter_delta"]) < 1e-7 for rank in case[arm]["per_rank_observed"])
        for case in summary["outcomes"].values()
        for arm in ("base", "candidate")
    )
    checks["both_ranks_completed"] = all(
        all(rank["global_step"] == 1 for rank in case[arm]["per_rank_observed"])
        for case in summary["outcomes"].values()
        for arm in ("base", "candidate")
    )
    result = {"audit": "stdlib-only independent reconstruction", "passed": sum(checks.values()), "total": len(checks), "checks": checks}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
