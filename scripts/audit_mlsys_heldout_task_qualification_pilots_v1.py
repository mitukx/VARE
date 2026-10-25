#!/usr/bin/env python3
"""Independently replay source, record-manifest, and gate checks for qualification pilots v1-v3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/mlx-lm.git"
CATALOG = ROOT / "results/mlsys-task-deep-qualification-v1/qualification.json"
ALLOCATION = ROOT / "results/mlsys-task-deep-qualification-v1/protocol.json"
RUNS = {
    "mlsys-heldout-task-qualification-pilot-v1": ROOT / "results/mlsys-heldout-task-qualification-pilot-v1/run-3",
    "mlsys-heldout-task-qualification-pilot-v2": ROOT / "results/mlsys-heldout-task-qualification-pilot-v2/run-1",
    "mlsys-heldout-task-qualification-pilot-v3": ROOT / "results/mlsys-heldout-task-qualification-pilot-v3/run-1",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob(revision: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE), "-c", "core.useReplaceRefs=false", "show", f"{revision}:{path}"],
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        timeout=30,
    )


def verify_manifest(run: Path) -> dict:
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = digest(manifest_path.read_bytes())
    expected_hash_line = f"{manifest_hash}  manifest.json\n"
    if (run / "manifest.sha256").read_text(encoding="utf-8") != expected_hash_line:
        raise ValueError(f"manifest checksum mismatch: {run}")
    expected_files = manifest["files"]
    actual_names = {path.name for path in run.iterdir() if path.is_file()} - {"manifest.json", "manifest.sha256"}
    if actual_names != set(expected_files):
        raise ValueError(f"manifest inventory mismatch: {run}")
    for name, expected in expected_files.items():
        if digest((run / name).read_bytes()) != expected:
            raise ValueError(f"manifest content mismatch: {run / name}")
    return manifest


def verify_protocol(protocol_path: Path, lock_path: Path) -> tuple[dict, dict]:
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if digest(protocol_raw) != lock["protocol_sha256"]:
        raise ValueError(f"protocol hash mismatch: {protocol_path}")
    if digest(CATALOG.read_bytes()) != protocol["source_catalog"]["catalog_sha256"]:
        raise ValueError("catalog hash mismatch")
    if digest(ALLOCATION.read_bytes()) != protocol["source_catalog"]["allocation_sha256"]:
        raise ValueError("allocation hash mismatch")
    if lock.get("exposure_audit_sha256"):
        exposure_path = ROOT / protocol["exposure"]["audit_path"]
        if digest(exposure_path.read_bytes()) != lock["exposure_audit_sha256"]:
            raise ValueError("frozen exposure-audit hash mismatch")
    return protocol, lock


def audit() -> dict:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    allocation = json.loads(ALLOCATION.read_text(encoding="utf-8"))
    catalog_tasks = {task["task_id"]: task for task in catalog["tasks"]}
    results = {}
    protocol_ids = {
        "mlsys-heldout-task-qualification-pilot-v1": (
            ROOT / "protocols/mlsys_heldout_task_qualification_pilot_v1.json",
            ROOT / "protocols/mlsys_heldout_task_qualification_pilot_v1.lock.json",
        ),
        "mlsys-heldout-task-qualification-pilot-v2": (
            ROOT / "protocols/mlsys_heldout_task_qualification_pilot_v2.json",
            ROOT / "protocols/mlsys_heldout_task_qualification_pilot_v2.lock.json",
        ),
        "mlsys-heldout-task-qualification-pilot-v3": (
            ROOT / "protocols/mlsys_heldout_task_qualification_pilot_v3.json",
            ROOT / "protocols/mlsys_heldout_task_qualification_pilot_v3.lock.json",
        ),
    }
    for protocol_id, run in RUNS.items():
        protocol, lock = verify_protocol(*protocol_ids[protocol_id])
        manifest = verify_manifest(run)
        summary = json.loads((run / "qualification.json").read_text(encoding="utf-8"))
        if summary["protocol_id"] != protocol_id or summary["protocol_sha256"] != lock["protocol_sha256"]:
            raise ValueError(f"run/protocol identity mismatch: {protocol_id}")
        exposure = json.loads((run / "exposure-audit.json").read_text(encoding="utf-8"))
        if exposure["observed_vare_candidate_exposure"] or exposure["candidate_rollout_or_training_matches"]:
            raise ValueError(f"exposure audit is not clean within scope: {protocol_id}")
        if summary["exposure_audit"]["audit_file_sha256"] != digest((run / "exposure-audit.json").read_bytes()):
            raise ValueError(f"run did not bind exposure audit: {protocol_id}")
        if not summary["shared_boundary_probe"]["passed"]:
            raise ValueError(f"shared candidate boundary failed: {protocol_id}")
        task_results = {}
        for task in protocol["task_contracts"]:
            task_id = task["task_id"]
            catalog_task = catalog_tasks[task_id]
            if catalog_task["split"] != "heldout" or task_id not in allocation["split"]["heldout"]:
                raise ValueError(f"task allocation changed: {task_id}")
            for key, revision, path in (
                ("baseline_target_sha256", task["baseline_commit"], task["target_path"]),
                ("fixed_target_sha256", task["fixed_commit"], task["target_path"]),
                ("grader_sha256", task["grader_revision"], task["grader_path"]),
            ):
                if digest(git_blob(revision, path)) != task[key]:
                    raise ValueError(f"pinned upstream blob mismatch: {task_id}/{key}")
            license_path = task.get("license_file_path") or catalog_task.get("license_file_path")
            license_sha = task.get("license_file_sha256") or catalog_task.get("license_file_sha256")
            if license_path and digest(git_blob(task["fixed_commit"], license_path)) != license_sha:
                raise ValueError(f"license blob mismatch: {task_id}")
            if digest((ROOT / task["candidate_prompt_path"]).read_bytes()) != task["candidate_prompt_sha256"]:
                raise ValueError(f"candidate task prompt mismatch: {task_id}")
            variant_rows = {}
            for row in next(item for item in summary["task_certificates"] if item["task_id"] == task_id)["variants"]:
                record_path = run / row["record"]
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if digest(record_path.read_bytes()) != row["record_sha256"] or digest(record_path.read_bytes()) != manifest["files"][row["record"]]:
                    raise ValueError(f"variant record hash mismatch: {record_path}")
                if record["task_id"] != task_id or record["variant"] != row["variant"] or record["decision"] != row["decision"]:
                    raise ValueError(f"variant identity mismatch: {record_path}")
                variant_rows[row["variant"]] = record
            pass_gate = (
                variant_rows["pinned_baseline"]["decision"] == "FAIL"
                and variant_rows["pinned_known_fix_calibration_only"]["decision"] == "PASS"
                and variant_rows["prespecified_incorrect_mutant"]["decision"] == "FAIL"
                and variant_rows["pinned_baseline"]["candidate_visible_grader"] is False
                and variant_rows["pinned_baseline"]["candidate_visible_git_history"] is False
                and variant_rows["pinned_baseline"]["candidate_visible_task_prompt"] is False
                and all(
                    row["candidate_visible_grader"] is False
                    and row["candidate_visible_git_history"] is False
                    and row["candidate_visible_task_prompt"] is False
                    for row in variant_rows.values()
                )
                and next(item for item in summary["task_certificates"] if item["task_id"] == task_id)["patch_admission_controls"]["path_traversal_rejected"]
                and next(item for item in summary["task_certificates"] if item["task_id"] == task_id)["patch_admission_controls"]["non_allowlisted_target_rejected"]
            )
            if pass_gate != next(item for item in summary["task_certificates"] if item["task_id"] == task_id)["certificate_gate_passed"]:
                raise ValueError(f"reported task certificate does not match raw variants: {task_id}")
            task_results[task_id] = {
                "status": "CERTIFIED_WITHIN_AUDITED_VARE_SCOPE" if pass_gate else "UNQUALIFIED",
                "baseline": variant_rows["pinned_baseline"]["decision"],
                "known_fix": variant_rows["pinned_known_fix_calibration_only"]["decision"],
                "mutant": variant_rows["prespecified_incorrect_mutant"]["decision"],
            }
        if protocol_id.endswith(("v2", "v3")):
            if summary["candidate_invocations"] != 4 or summary["candidate_invocations"] > summary["candidate_invocation_budget"]:
                raise ValueError(f"{protocol_id} invocation accounting is inconsistent")
        if protocol_id.endswith("v2"):
            recovered = summary["recovered_prior_attempt_failure"]
            if set(recovered["completed_variants_reused_without_reexecution"]) != {
                "mlxlm-1938-phi3small-logits-mask/pinned_baseline",
                "mlxlm-1938-phi3small-logits-mask/pinned_known_fix_calibration_only",
            }:
                raise ValueError("v2 resume did not preserve exactly the completed variants")
            if not (run / "failure.json").is_file():
                raise ValueError("v2 original interruption record was not preserved")
        memory_samples = summary["candidate_memory_sampling"]["samples"]
        usable_memory_samples = [
            row["memory_bytes"] for row in memory_samples if row["memory_bytes"] > 0
        ]
        results[protocol_id] = {
            "manifest_sha256": digest((run / "manifest.json").read_bytes()),
            "decision": summary["decision"],
            "tasks": task_results,
            "candidate_invocations": summary["candidate_invocations"],
            "host_wall_seconds_recorded": summary["host_resource_usage"]["wall_seconds"],
            "elapsed_from_recorded_start_to_completion_seconds": (
                datetime.fromisoformat(summary["completed_at_utc"])
                - datetime.fromisoformat(summary["started_at_utc"])
            ).total_seconds(),
            "wall_time_note": (
                "The v2 runner's host_wall_seconds covers its resume process only; elapsed timestamps include the original attempt and the preserved interruption interval."
                if protocol_id.endswith("v2")
                else "Elapsed time is computed from the frozen run start and completion timestamps."
            ),
            "host_swap_growth_bytes": summary["host_resource_usage"]["swap_growth_bytes"],
            "candidate_memory_sample_count": summary["candidate_memory_sampling"]["sample_count"],
            "candidate_memory_usable_sample_count": len(usable_memory_samples),
            "candidate_memory_sampled_peak_bytes": max(usable_memory_samples, default=None),
            "candidate_memory_zero_only_samples": bool(memory_samples) and not usable_memory_samples,
        }
    qualified_tasks = {
        task_id: task
        for row in results.values()
        for task_id, task in row["tasks"].items()
    }
    fixed_statuses = {
        "transformers-49412-fusion-idempotence": ("RETIRED_EXPOSED", "Nine retained Qwen3-4B Development attempts are recorded in the frozen readiness audit."),
        "mlxlm-1902-continuation-score-truncation": ("UNQUALIFIED", "Frozen pilot v1 baseline, known fix and mutant all failed parent-owned assertions; known-fix false-negative remains unresolved."),
        "mlxlm-1935-zero-max-tokens": ("UNQUALIFIED_NOT_RUN", "Static preflight excluded the task because its pinned test fixture would download model weights."),
    }
    matrix = []
    for task_id in allocation["split"]["heldout"]:
        if task_id in qualified_tasks:
            status = qualified_tasks[task_id]["status"]
            if status == "CERTIFIED_WITHIN_AUDITED_VARE_SCOPE":
                reason = "Frozen baseline/fix/mutant and boundary gates passed within the audited local VARE exposure scope."
            else:
                reason = "The frozen calibration did not pass all baseline/fix/mutant gates; the task remains in its original split but is not eligible."
        elif task_id in fixed_statuses:
            status, reason = fixed_statuses[task_id]
        else:
            status, reason = "UNASSESSED", "No complete current task certificate was produced in this cycle."
        matrix.append({"task_id": task_id, "status": status, "reason": reason})
    return {
        "audit_id": "mlsys-heldout-task-qualification-pilots-v1-independent-audit",
        "audit_implementation_sha256": digest(Path(__file__).resolve().read_bytes()),
        "supersedes_audit_file": "results/mlsys-heldout-task-qualification-pilot-v1/evidence-audit-v3.json; this audit adds the subsequently frozen and measured v3 #1979 certificate, and interprets raw 0 B Docker memory samples as unusable rather than a zero peak. Earlier audit-v1's #1902 reason-text correction remains preserved in evidence-audit-v2.json.",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_catalog_sha256": digest(CATALOG.read_bytes()),
        "frozen_allocation_sha256": digest(ALLOCATION.read_bytes()),
        "task_results": {task_id: result for row in results.values() for task_id, result in row["tasks"].items()},
        "original_frozen_heldout_matrix": matrix,
        "pilot_runs": results,
        "certificate_counts": {
            "qualified": sum(task["status"] == "CERTIFIED_WITHIN_AUDITED_VARE_SCOPE" for row in results.values() for task in row["tasks"].values()),
            "unqualified": sum(task["status"] == "UNQUALIFIED" for row in results.values() for task in row["tasks"].values()),
        },
        "heldout_gate": {
            "required_eligible_tasks": 15,
            "eligible_tasks_now": sum(task["status"] == "CERTIFIED_WITHIN_AUDITED_VARE_SCOPE" for task in qualified_tasks.values()),
            "additional_eligible_tasks_required": 15 - sum(task["status"] == "CERTIFIED_WITHIN_AUDITED_VARE_SCOPE" for task in qualified_tasks.values()),
            "original_reserved_families_after_known_exposure": 8,
            "original_families_not_certified_or_retired": sum(
                row["status"] not in {"CERTIFIED_WITHIN_AUDITED_VARE_SCOPE", "RETIRED_EXPOSED"}
                for row in matrix
            ),
            "model_learning_admitted": False,
        },
        "claim_boundary": "Same-host independent audit of frozen task source identities, exposure records, raw variant records, manifests, and declared certificate gates. Not an external reproduction or model/agent result.",
        "external_pretraining_corpus": "UNKNOWN_NOT_AUDITED",
        "model_inference_or_training": "NOT_RUN",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("output exists; audit records are append-only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = audit()
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, args.output)
        directory_fd = os.open(args.output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({"qualified": result["certificate_counts"]["qualified"], "unqualified": result["certificate_counts"]["unqualified"], "decision": "AUDIT_PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
