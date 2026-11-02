#!/usr/bin/env python3
"""Independent offline audit for Sprint XI model-coupled recovery evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_hash(value: Any) -> str:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        header = canonical({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
        return sha_bytes(header + tensor.numpy().tobytes())
    if isinstance(value, dict):
        return sha_bytes(canonical([
            [str(key), logical_hash(value[key])]
            for key in sorted(value, key=lambda item: str(item))
        ]))
    if isinstance(value, (list, tuple)):
        return sha_bytes(canonical([logical_hash(item) for item in value]))
    if isinstance(value, float):
        return sha_bytes(canonical({"float_hex": value.hex()}))
    return sha_bytes(canonical(value))


def audit_journal(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    previous = "0" * 64
    for index, row in enumerate(rows):
        if row["sequence"] != index or row["previous_sha256"] != previous:
            raise AssertionError(f"journal sequence/hash chain broken at {index}: {path}")
        payload = {key: value for key, value in row.items() if key != "event_sha256"}
        expected = sha_bytes(canonical(payload))
        if row["event_sha256"] != expected:
            raise AssertionError(f"journal event hash mismatch at {index}: {path}")
        previous = row["event_sha256"]
    return rows


def verify_manifest(root: Path, filename: str) -> dict[str, Any]:
    manifest_path = root / filename
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha_file(path) != row["sha256"]:
            raise AssertionError(f"campaign file manifest mismatch: {row['path']}")
    return manifest


def audit_checkpoint(path: Path, runner_module: Any) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = runner_module._checkpoint_manifest(path)
    import numpy as np
    import torch

    bias = np.load(path / "adapter_bias.npy", allow_pickle=False)
    optimizer = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
    rng = torch.load(path / "cpu_rng.pt", map_location="cpu", weights_only=False)
    identities = {
        "adapter": sha_bytes(canonical({"dtype": str(bias.dtype), "shape": list(bias.shape)}) + bias.tobytes()),
        "optimizer": logical_hash(optimizer),
        "rng": logical_hash(rng),
    }
    if sha_file(path / "adapter_bias.npy") != manifest["metadata"]["adapter_sha256"]:
        raise AssertionError(f"adapter identity mismatch: {path}")
    return manifest, identities


def run(protocol_path: Path, run_root: Path, vare_root: Path, rvl_root: Path) -> dict[str, Any]:
    runner_path = Path(__file__).with_name("run_model_coupled_v1.py")
    spec = __import__("importlib.util", fromlist=["spec_from_file_location"]).spec_from_file_location("sprint_xi_runner", runner_path)
    module = __import__("importlib.util", fromlist=["module_from_spec"]).module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = module._read_protocol(protocol_path)
    module._validate_sources(protocol, vare_root, rvl_root)

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks[name] = bool(condition)
        if detail is not None:
            details[name] = detail
        if not condition:
            raise AssertionError(name)

    campaign_manifest = verify_manifest(run_root, "sha256-manifest.json")
    check("campaign_manifest_integrity", bool(campaign_manifest["files"]))
    summary = json.loads((run_root / "campaign-summary.json").read_text())
    check("campaign_success_status", summary["status"] == "completed")
    check("exact_protocol_and_model_identity", summary["protocol_sha256"] == sha_file(protocol_path) and summary["model_revision"] == protocol["model"]["revision"])

    rollout = run_root / "rollouts/fresh-v0.json"
    sidecar = json.loads(rollout.with_suffix(rollout.suffix + ".manifest.json").read_text())
    rollout_payload = json.loads(rollout.read_text())
    ack = json.loads((run_root / "worker-recovery-ack.json").read_text())
    process_rows = summary["processes"]
    worker_fault = next(row for row in process_rows if row["name"] == "worker-fresh-v0-sigkill")
    check("real_rollout_worker_SIGKILL", worker_fault["returncode"] == -9 and rollout_payload["status"] == "mixed_reward_group_found")
    check("worker_trajectory_persisted_before_SIGKILL", sha_file(rollout) == sidecar["artifact_sha256"] and rollout_payload["total_generated_tokens"] > 0)
    check("worker_resume_did_not_regenerate", ack["generation_replayed"] is False and ack["effective_group_deliveries"] == 1 and ack["artifact_sha256"] == sidecar["artifact_sha256"])
    check("worker_generation_identity_complete", len(rollout_payload["task_records"]) >= 1 and all(len(row["generations"]) == 4 for row in rollout_payload["task_records"]))

    stale = json.loads((run_root / "staleness-admission.json").read_text())
    check("VARE_rejects_stale_group", stale["replay_group_added"] and stale["vare_rejected_stale_group"] and stale["vare_admitted_members"] == 0)
    check("naive_admits_same_stale_group", stale["naive_accepted_stale_group"] and stale["naive_admitted_members"] == 4)
    details["stale_queue_age_seconds"] = stale["queue_seconds"]

    cp1 = run_root / "checkpoints/candidate-v1"
    ref = run_root / "checkpoints/reference-v2"
    recovered = run_root / "checkpoints/recovered-v2"
    naive = run_root / "checkpoints/naive-stale-v2"
    cp1_manifest, cp1_ids = audit_checkpoint(cp1, module)
    ref_manifest, ref_ids = audit_checkpoint(ref, module)
    recovered_manifest, recovered_ids = audit_checkpoint(recovered, module)
    naive_manifest, naive_ids = audit_checkpoint(naive, module)
    import numpy as np
    cp1_bias = np.load(cp1 / "adapter_bias.npy", allow_pickle=False)
    check("v1_actual_policy_and_Adam_update", cp1_manifest["metadata"]["optimizer_step_count"] == 1 and not np.array_equal(cp1_bias, np.zeros_like(cp1_bias)) and bool(cp1_manifest["metadata"]["optimizer_serialization_sha256"]))
    check("reference_v2_has_two_optimizer_steps", ref_manifest["metadata"]["optimizer_step_count"] == 2)
    check("recovered_v2_has_two_optimizer_steps", recovered_manifest["metadata"]["optimizer_step_count"] == 2)
    check("same_committed_parent_for_v2", ref_manifest["metadata"]["parent_checkpoint_sha256"] == cp1_manifest["checkpoint_sha256"] == recovered_manifest["metadata"]["parent_checkpoint_sha256"])
    check("recovered_adapter_matches_reference", recovered_ids["adapter"] == ref_ids["adapter"], {"reference": ref_ids["adapter"], "recovered": recovered_ids["adapter"]})
    check("recovered_Adam_matches_reference", recovered_ids["optimizer"] == ref_ids["optimizer"], {"reference": ref_ids["optimizer"], "recovered": recovered_ids["optimizer"]})
    check("recovered_RNG_matches_reference", recovered_ids["rng"] == ref_ids["rng"], {"reference": ref_ids["rng"], "recovered": recovered_ids["rng"]})
    check("naive_stale_branch_is_separate", naive_manifest["metadata"]["admission_policy"] == "naive" and naive_manifest["metadata"]["parent_checkpoint_sha256"] == cp1_manifest["checkpoint_sha256"])

    fault_journal = audit_journal(run_root / "learner-runs/faulted-v2-attempt/learner.journal.jsonl")
    recovery_journal = audit_journal(run_root / "learner-runs/recovery-v2/learner.journal.jsonl")
    reference_journal = audit_journal(run_root / "learner-runs/reference-v2/learner.journal.jsonl")
    learner_fault = next(row for row in process_rows if row["name"] == "learner-sigkill-after-optimizer")
    check("learner_SIGKILL_after_real_optimizer_call", learner_fault["returncode"] == -9 and any(row["kind"] == "optimizer_applied_uncommitted" for row in fault_journal))
    check("faulted_attempt_has_no_effective_commit", not any(row["kind"] == "checkpoint_committed" for row in fault_journal) and not (run_root / "checkpoints/uncommitted-v2").exists())
    check("recovery_replays_and_commits_once", sum(row["kind"] == "checkpoint_committed" for row in recovery_journal) == 1)
    check("uninterrupted_reference_commits_once", sum(row["kind"] == "checkpoint_committed" for row in reference_journal) == 1)
    check("two_optimizer_attempts_one_effective_recovery_commit", summary["optimizer_attempts_for_v2_recovery"] == 2 and summary["effective_v2_commits"] == 1)

    evaluations = run_root / "evaluations"
    base_eval = json.loads((evaluations / "base.json").read_text())
    v1_eval = json.loads((evaluations / "v1.json").read_text())
    v2_eval = json.loads((evaluations / "recovered-v2.json").read_text())
    adverse_eval = json.loads((evaluations / "adverse-candidate.json").read_text())
    pointer = json.loads((run_root / "promoted-policy.json").read_text())
    rollback = json.loads((run_root / "rollback-negative-control.json").read_text())
    check("evaluation_files_are_policy_bound", v1_eval["checkpoint_sha256"] == cp1_manifest["checkpoint_sha256"] and v2_eval["checkpoint_sha256"] == recovered_manifest["checkpoint_sha256"])
    check("promotion_pointer_is_policy_bound", pointer["checkpoint_sha256"] == (ref_manifest["checkpoint_sha256"] if pointer["selected_checkpoint"] == "reference-v2" else recovered_manifest["checkpoint_sha256"] if pointer["selected_checkpoint"] == "recovered-v2" else cp1_manifest["checkpoint_sha256"] if pointer["selected_checkpoint"] == "candidate-v1" else base_eval["policy_sha256"]))
    check("adverse_candidate_is_rejected", rollback["adverse_candidate_rejected"] and adverse_eval["mean_correct_choice_probability"] < rollback["incumbent_score"])
    check("rollback_keeps_incumbent_identity", rollback["active_checkpoint_unchanged"] and rollback["active_pointer_sha256_after_rejection"] == sha_file(run_root / "promoted-policy.json"))
    details["evaluation_means"] = {
        "base": base_eval["mean_correct_choice_probability"],
        "v1": v1_eval["mean_correct_choice_probability"],
        "reference_v2": json.loads((evaluations / "reference-v2.json").read_text())["mean_correct_choice_probability"],
        "recovered_v2": v2_eval["mean_correct_choice_probability"],
        "naive_stale_v2": json.loads((evaluations / "naive-stale-v2.json").read_text())["mean_correct_choice_probability"],
        "adverse": adverse_eval["mean_correct_choice_probability"],
    }
    check("all_child_memory_caps_respected", all(row["peak_process_tree_rss_bytes"] <= protocol["resource_limits"]["max_process_tree_rss_bytes"] for row in process_rows))
    check("swap_growth_cap_respected", summary["host_swap_growth_bytes"] <= protocol["resource_limits"]["max_swap_growth_bytes"])
    check("resource_and_time_caps_respected", summary["campaign_wall_seconds"] <= protocol["resource_limits"]["max_wall_seconds"])
    details["parameter_scope"] = "frozen Qwen base; trainable vocabulary-logit-bias adapter only"
    details["training_steps"] = {"v1": cp1_manifest["metadata"]["optimizer_step_count"], "reference_v2": ref_manifest["metadata"]["optimizer_step_count"], "recovered_v2": recovered_manifest["metadata"]["optimizer_step_count"], "naive_v2": naive_manifest["metadata"]["optimizer_step_count"]}

    audit = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha_file(protocol_path),
        "runner_sha256": sha_file(runner_path),
        "auditor_sha256": sha_file(Path(__file__).resolve()),
        "run_manifest_sha256": sha_file(run_root / "sha256-manifest.json"),
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "details": details,
        "result": "PASS" if all(checks.values()) else "FAIL",
        "scope": "bounded model-coupled RL systems/recovery evidence; synthetic rewards; no coding capability or generalization claim",
    }
    (run_root / "audit.json").write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    final_files = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.name == "sha256-manifest-final.json":
            continue
        final_files.append({"path": path.relative_to(run_root).as_posix(), "sha256": sha_file(path), "bytes": path.stat().st_size})
    (run_root / "sha256-manifest-final.json").write_text(json.dumps({"schema_version": 1, "files": final_files}, sort_keys=True, indent=2) + "\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--vare-root", type=Path, required=True)
    parser.add_argument("--rvl-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol.resolve(), args.run.resolve(), args.vare_root.resolve(), args.rvl_root.resolve())
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
