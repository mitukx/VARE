#!/usr/bin/env python3
"""Offline audit for the immutable Sprint XI recovery continuation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_journal(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    prev = "0" * 64
    for i, row in enumerate(rows):
        if row["sequence"] != i or row["previous_sha256"] != prev:
            raise AssertionError(f"journal chain invalid: {path} row {i}")
        event = {key: value for key, value in row.items() if key != "event_sha256"}
        if row["event_sha256"] != sha_bytes(canon(event)):
            raise AssertionError(f"journal event digest invalid: {path} row {i}")
        prev = row["event_sha256"]
    return rows


def check_manifest(root: Path) -> bool:
    manifest = json.loads((root / "sha256-manifest.json").read_text())
    return all(
        (root / row["path"]).is_file()
        and (root / row["path"]).stat().st_size == row["bytes"]
        and sha(root / row["path"]) == row["sha256"]
        for row in manifest["files"]
    )


def run(protocol_path: Path, original_protocol_path: Path, base: Path, root: Path, vare_root: Path, rvl_root: Path) -> dict[str, Any]:
    lock = json.loads(protocol_path.read_text())
    runner_path = Path(__file__).with_name("run_recovery_continuation_v1.py")
    original_runner = Path(__file__).with_name("run_model_coupled_v1.py")
    original = json.loads(original_protocol_path.read_text())
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    def check(name: str, result: bool, detail: Any = None) -> None:
        checks[name] = bool(result)
        if detail is not None:
            details[name] = detail
        if not result:
            raise AssertionError(name)

    check("continuation_runner_and_parent_protocol_hashes", sha(runner_path) == lock["runner_sha256"] and sha(original_protocol_path) == lock["original_protocol_sha256"] and sha(original_runner) == original["runner_sha256"])
    check("campaign_output_manifest", check_manifest(root))
    summary = json.loads((root / "continuation-summary.json").read_text())
    check("continuation_completed", summary["status"] == "completed")
    check("exact_model_revision", summary["model_revision"] == lock["model"]["revision"])

    # Verify frozen source identities independently through the original runner.
    oldmod = load(original_runner, "old_xi_runner_for_audit")
    protocol = oldmod._read_protocol(original_protocol_path)
    oldmod._validate_sources(protocol, vare_root, rvl_root)
    check("original_vare_rvl_source_identity", protocol["source"]["vare_revision"] == original["source"]["vare_revision"] and protocol["source"]["rvl_revision"] == original["source"]["rvl_revision"])

    parent_failure = json.loads((base / "attempt-failure.json").read_text())
    parent_ack = json.loads((base / "worker-recovery-ack.json").read_text())
    parent_rollout = json.loads((base / "rollouts/fresh-v0.json").read_text())
    check("parent_worker_was_killed_after_durable_rollout", parent_failure["failure_class"] == "harness-selection-condition" and parent_rollout["fault_after_persist"] and parent_ack["generation_replayed"] is False and parent_ack["effective_group_deliveries"] == 1)
    check("parent_v1_optimizer_commit_exists", (base / "learner-runs/update-v1/summary.json").is_file() and (base / "checkpoints/candidate-v1/manifest.json").is_file())

    stale = json.loads((root / "staleness-admission.json").read_text())
    check("stale_complete_group_rejected_by_VARE", stale["replay_group_added"] and stale["vare_admitted_members"] == 0 and stale["vare_rejected_stale_group"])
    check("same_stale_group_accepted_by_naive_sampler", stale["naive_admitted_members"] == 4 and stale["naive_accepted_same_stale_group"])
    check("constant_stale_group_not_optimizer_trained", stale["rewards"] == [0.0, 0.0, 0.0, 0.0] and stale["optimizer_update_on_constant_group"] is False and "NOT_RUN" in summary["naive_stale_optimizer_update"])
    details["stale_artifact_age_at_admission_seconds"] = stale["queue_age_seconds"]

    fresh = json.loads((root / "rollouts/fresh-v1.json").read_text())
    check("fresh_v1_new_rollout_has_nonconstant_rewards", fresh["status"] == "mixed_reward_group_found" and len(fresh["task_records"]) >= 1)
    check("predeclared_fresh_cohort_not_reused_before_continuation", not (base / "rollouts/fresh-v1.json").exists())

    cp1 = base / "checkpoints/candidate-v1"
    ref = root / "checkpoints/reference-v2"
    rec = root / "checkpoints/recovered-v2"
    ref_manifest, ref_ids = load(Path(__file__).with_name("audit_model_coupled_v1.py"), "audit_util").audit_checkpoint(ref, oldmod)
    rec_manifest, rec_ids = load(Path(__file__).with_name("audit_model_coupled_v1.py"), "audit_util_2").audit_checkpoint(rec, oldmod)
    cp1_manifest = oldmod._checkpoint_manifest(cp1)
    check("reference_and_recovery_branch_from_same_v1", ref_manifest["metadata"]["parent_checkpoint_sha256"] == cp1_manifest["checkpoint_sha256"] == rec_manifest["metadata"]["parent_checkpoint_sha256"])
    check("v2_adapter_identity_exact", ref_ids["adapter"] == rec_ids["adapter"], {"reference": ref_ids["adapter"], "recovered": rec_ids["adapter"]})
    check("v2_Adam_identity_exact", ref_ids["optimizer"] == rec_ids["optimizer"], {"reference": ref_ids["optimizer"], "recovered": rec_ids["optimizer"]})
    check("v2_RNG_identity_exact", ref_ids["rng"] == rec_ids["rng"], {"reference": ref_ids["rng"], "recovered": rec_ids["rng"]})
    check("v2_optimizer_state_step_count", ref_manifest["metadata"]["optimizer_step_count"] == rec_manifest["metadata"]["optimizer_step_count"] == 2)
    details["v2_reference_checkpoint_sha256"] = ref_manifest["checkpoint_sha256"]
    details["v2_recovered_checkpoint_sha256"] = rec_manifest["checkpoint_sha256"]

    fault_journal = audit_journal(root / "learner-runs/faulted-v2-attempt/learner.journal.jsonl")
    recovery_journal = audit_journal(root / "learner-runs/recovery-v2/learner.journal.jsonl")
    reference_journal = audit_journal(root / "learner-runs/reference-v2/learner.journal.jsonl")
    fault_row = next(row for row in summary["processes"] if row["name"] == "learner-sigkill-after-optimizer")
    check("real_learner_SIGKILL_after_optimizer", fault_row["returncode"] == -9 and any(row["kind"] == "optimizer_applied_uncommitted" for row in fault_journal))
    check("uncommitted_update_has_no_checkpoint", not any(row["kind"] == "checkpoint_committed" for row in fault_journal) and not (root / "checkpoints/uncommitted-v2").exists())
    check("restart_has_one_effective_commit", sum(row["kind"] == "checkpoint_committed" for row in recovery_journal) == 1 and summary["effective_v2_recovery_commits"] == 1)
    check("reference_has_one_commit", sum(row["kind"] == "checkpoint_committed" for row in reference_journal) == 1)
    check("duplicate_optimizer_computation_is_not_double_commit", summary["optimizer_attempts_for_v2_recovery"] == 2 and summary["effective_v2_recovery_commits"] == 1)

    evals = root / "evaluations"
    e1 = json.loads((evals / "v1.json").read_text())
    er = json.loads((evals / "reference-v2.json").read_text())
    ec = json.loads((evals / "recovered-v2.json").read_text())
    adverse = json.loads((evals / "adverse.json").read_text())
    rollback = json.loads((root / "rollback-negative-control.json").read_text())
    pointer = root / "promoted-policy.json"
    check("independent_reference_and_recovered_evaluations_match", er["mean_correct_choice_probability"] == ec["mean_correct_choice_probability"])
    check("heldout_synthetic_choice_score_non_decreasing", ec["mean_correct_choice_probability"] >= e1["mean_correct_choice_probability"])
    check("promotion_pointer_matches_one_checkpoint", json.loads(pointer.read_text())["checkpoint_sha256"] in {cp1_manifest["checkpoint_sha256"], rec_manifest["checkpoint_sha256"]})
    check("adverse_candidate_rejected", rollback["rejected"] and adverse["mean_correct_choice_probability"] < rollback["incumbent_score"])
    check("rejection_preserves_active_identity", rollback["incumbent_identity_preserved"] and rollback["pointer_sha256_after_rejection"] == sha(pointer))
    details["evaluation_mean_correct_choice_probability"] = {"v1": e1["mean_correct_choice_probability"], "reference_v2": er["mean_correct_choice_probability"], "recovered_v2": ec["mean_correct_choice_probability"], "adverse": adverse["mean_correct_choice_probability"]}

    limits = lock["resource_limits"]
    check("no_swap_growth", summary["host_swap_growth_bytes"] <= limits["max_swap_growth_bytes"])
    check("memory_caps", all(row["peak_process_tree_rss_bytes"] <= limits["max_process_tree_rss_bytes"] for row in summary["processes"]))
    check("wall_cap", summary["campaign_wall_seconds"] <= limits["max_wall_seconds"])
    details["recovery_latency_seconds"] = summary["recovery_latency_seconds"]
    details["host_ram_available_bytes_after"] = summary["host_ram_available_bytes_after"]
    details["host_swap_growth_bytes"] = summary["host_swap_growth_bytes"]

    result = {
        "schema_version": 1,
        "protocol_id": lock["protocol_id"],
        "protocol_sha256": sha(protocol_path),
        "runner_sha256": sha(runner_path),
        "auditor_sha256": sha(Path(__file__).resolve()),
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "details": details,
        "limitations": ["Synthetic arithmetic rewards and four evaluation tasks only.", "The pretrained Qwen base is frozen; only the output-bias adapter was trained.", "The naive asynchronous arm demonstrated stale admission but did not run optimizer training because all stale-group rewards were zero.", "This is a one-seed, single-host systems result, not long agent trajectories or coding capability."],
    }
    (root / "audit.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "sha256-manifest-final.json":
            continue
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    (root / "sha256-manifest-final.json").write_text(json.dumps({"schema_version": 1, "files": files}, sort_keys=True, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--vare-root", type=Path, required=True)
    parser.add_argument("--rvl-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol.resolve(), args.original_protocol.resolve(), args.base_run.resolve(), args.run.resolve(), args.vare_root.resolve(), args.rvl_root.resolve())
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
