#!/usr/bin/env python3
"""Independently audit the frozen deterministic coding-interface systems replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def blob(repository: Path, revision: str, target: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), "show", f"{revision}:{target}"],
        check=True, capture_output=True,
    ).stdout


def pinned_blob(repository: Path, revision: str, target: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), "show", f"{revision}:{target}"],
        check=True, capture_output=True,
    ).stdout


def verify_artifact_manifest(results: Path) -> str:
    manifest_path = results / "artifact-manifest.json"
    sidecar_path = results / "artifact-manifest.json.sha256"
    manifest_sha = file_digest(manifest_path)
    if sidecar_path.read_text(encoding="utf-8").split()[0] != manifest_sha:
        raise ValueError("prior artifact manifest sidecar SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != manifest.get(
        "files_including_independent_audit"
    ):
        raise ValueError("prior artifact manifest file count mismatch")
    seen: set[str] = set()
    for entry in files:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe path in prior artifact manifest: {relative}")
        if relative.as_posix() in seen:
            raise ValueError(f"duplicate path in prior artifact manifest: {relative}")
        seen.add(relative.as_posix())
        path = results / relative
        if (not path.is_file() or path.stat().st_size != entry["bytes"]
                or file_digest(path) != entry["sha256"]):
            raise ValueError(f"prior artifact manifest entry mismatch: {relative}")
    return manifest_sha


def verify_journal(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"empty journal: {path}")
    previous = "0" * 64
    events = []
    for seq, row in enumerate(rows, 1):
        event = row["event"]
        if event.get("sequence") != seq or row["previous_event_sha256"] != previous:
            raise ValueError(f"journal sequence/link mismatch: {path}")
        canonical = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        current = digest(canonical.encode())
        if row["event_sha256"] != current:
            raise ValueError(f"journal event hash mismatch: {path}")
        previous = current
        events.append(event)
    return events, previous


def write_new_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            n = os.write(fd, view)
            if n <= 0:
                raise OSError("audit write made no progress")
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit-id", default="v2")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,15}", args.audit_id):
        raise ValueError("audit ID must be 1-16 lowercase alphanumeric/hyphen characters")
    root = args.repository.resolve(strict=True)
    results = args.results.resolve(strict=True)
    source = args.source_repo.resolve(strict=True)
    worktrees = args.worktree_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = file_digest(protocol_path)
    if protocol_sha != Path(str(protocol_path) + ".sha256").read_text().split()[0]:
        raise ValueError("protocol sidecar SHA mismatch")
    old = protocol["amendment_history"][-1]
    if file_digest(root / old["prior_file_preserved_at"]) != old["prior_protocol_sha256"]:
        raise ValueError("pre-run protocol archive mismatch")
    prior_manifest_sha = verify_artifact_manifest(results)
    for relative, expected in {
        **protocol["implementation_files_sha256"],
        **protocol["supporting_files_sha256"],
    }.items():
        pinned_bytes = pinned_blob(root, protocol["runner_commit"], relative)
        if digest(pinned_bytes) != expected:
            raise ValueError(f"frozen commit/source hash mismatch: {relative}")
    runner_bytes = pinned_blob(
        root, protocol["runner_commit"], "scripts/run_agent_edit_interface_systems_v1.py"
    )
    if digest(runner_bytes) != protocol["implementation_files_sha256"][
        "scripts/run_agent_edit_interface_systems_v1.py"
    ]:
        raise ValueError("pinned runner commit/source hash mismatch")
    if git(source, "remote", "get-url", "origin") != protocol["source_repository"]:
        raise ValueError("upstream source origin mismatch")

    all_path = results / "all-cell-results.json"
    aggregate = json.loads(all_path.read_text())
    if aggregate["protocol_sha256"] != protocol_sha:
        raise ValueError("aggregate result uses a different protocol")
    if digest(all_path.read_bytes()) != "5a63f9ec901d3204c42f1fa54309bcca1fc8f6dd3167f09d5469831efd89133b":
        raise ValueError("aggregate raw result SHA differs from run receipt")
    expected = {
        (task, interface, mode["name"])
        for task in protocol["tasks"] for interface in protocol["interfaces"]
        for mode in protocol["execution_modes"]
    }
    seen: set[tuple[str, str, str]] = set()
    per_cell = {}
    for path in results.glob("*/**/cell-result.json"):
        item = json.loads(path.read_text())
        per_cell[item["workspace_path"]] = (path, item)
    if len(per_cell) != 8 or len(aggregate["attempts"]) != 8:
        raise ValueError("expected eight per-cell and aggregate records")
    warnings = []
    checks = []
    paired: dict[str, set[tuple[str, str, str]]] = {
        mode["name"]: set() for mode in protocol["execution_modes"]
    }
    for item in aggregate["attempts"]:
        task_id, interface = item["task_id"], item["interface"]
        workspace = Path(item["workspace_path"]).resolve(strict=True)
        mode = workspace.relative_to(worktrees).parts[0]
        key = (task_id, interface, mode)
        if key not in expected or key in seen:
            raise ValueError(f"unexpected/duplicate cell: {key}")
        seen.add(key)
        task = protocol["task_cases"][task_id]
        if git(workspace, "rev-parse", "HEAD") != task["base"]:
            raise ValueError(f"cell is not at pinned base: {key}")
        target = workspace / task["target"]
        target_sha = file_digest(target)
        if (digest(blob(source, task["base"], task["target"])) != task["base_target_sha256"]
                or digest(blob(source, task["fix"], task["target"])) != task["fix_target_sha256"]
                or target_sha != task["fix_target_sha256"]):
            raise ValueError(f"base/fix/edited target hash mismatch: {key}")
        diff = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--binary", "HEAD", "--", task["target"]],
            check=True, capture_output=True,
        ).stdout
        if digest(diff) != item["diff_sha256"]:
            raise ValueError(f"candidate diff hash mismatch: {key}")
        cell_path, cell = per_cell[item["workspace_path"]]
        if cell != item:
            raise ValueError(f"aggregate/per-cell record mismatch: {key}")
        grade = item["independent_grade"]
        expected_prefix = ["sandbox-exec", "-p", "(version 1) (deny network*) (allow default)"]
        if (grade["passed"] is not True or grade["returncode"] != 0
                or grade["grader_sha256"] != task["grader_sha256"]
                or grade["network_denied"] is not True or grade["command"][:3] != expected_prefix):
            raise ValueError(f"trusted grade/isolation mismatch: {key}")
        if file_digest(cell_path.parent / "trusted-grade.stdout.log") != grade["stdout_sha256"]:
            raise ValueError(f"grader stdout hash mismatch: {key}")
        if file_digest(cell_path.parent / "trusted-grade.stderr.log") != grade["stderr_sha256"]:
            raise ValueError(f"grader stderr hash mismatch: {key}")

        controller = item["controller"]
        if (controller["base_sha256"] != task["base_target_sha256"]
                or controller["fixed_sha256"] != task["fix_target_sha256"]
                or controller["expected_fixed_sha256"] != task["fix_target_sha256"]
                or len(controller["actions"]) != 2 * controller["operation_count"]
                or sum(action["output_tokens"] for action in controller["actions"])
                != controller["serialized_tool_call_tokens"]):
            raise ValueError(f"controller patch/action/token audit mismatch: {key}")
        edit_name = "replace_lines" if interface == "lease_bound_lines_v1" else "replace_text"
        allowed_names = ({"read_lines", "replace_lines"} if interface == "lease_bound_lines_v1"
                         else {"read_file_range", "replace_text"})
        if any(action["name"] not in allowed_names for action in controller["actions"]):
            raise ValueError(f"unexpected action in trace: {key}")
        if not all(action["tool_response"].get("applied") is True
                   for action in controller["actions"] if action["name"] == edit_name):
            raise ValueError(f"edit response rejected: {key}")
        if interface == "lease_bound_lines_v1" and not all(
            lease["consumed"] for lease in controller["lease_audit"]
        ):
            raise ValueError(f"unconsumed edit lease: {key}")

        journal = cell_path.parent / "events.jsonl"
        events, final_event_sha = verify_journal(journal)
        if (file_digest(journal) != item["journal_sha256"]
                or len(events) != item["journal_event_count"]):
            raise ValueError(f"journal hash/count mismatch: {key}")
        context = next(event for event in events if event["event_type"] == "deterministic_run_context")
        if (context["task_id"] != task_id or context["interface"] != interface
                or context["base_revision"] != task["base"]
                or context["fix_revision"] != task["fix"]
                or context["base_sha256"] != task["base_target_sha256"]
                or context["fix_sha256"] != task["fix_target_sha256"]):
            raise ValueError(f"journal context mismatch: {key}")
        tool_events = [event for event in events if event["event_type"] == "deterministic_tool_calls"]
        commit_events = [event for event in events if event["event_type"] == "deterministic_tool_action_committed"]
        terminal_events = [event for event in events if event["event_type"] == "known_fix_controller_complete"]
        if (len(tool_events) != controller["operation_count"]
                or len(commit_events) != controller["operation_count"]
                or [action for event in tool_events for action in event["calls"]]
                != controller["actions"]):
            raise ValueError(f"journal actions/commits mismatch: {key}")
        if len(terminal_events) != 1:
            warnings.append(
                f"{key}: {len(terminal_events)} identical completion markers; "
                "journal hash chain and unique edit-commit events are valid"
            )
        paired[mode].add((task_id, interface, target_sha))
        checks.append({
            "task_id": task_id, "interface": interface, "mode": mode,
            "target_sha256": target_sha, "trusted_grade_passed": True,
            "operation_count": controller["operation_count"],
            "journal_event_count": len(events), "journal_final_event_sha256": final_event_sha,
            "completion_marker_count": len(terminal_events),
            "serialized_tool_call_tokens": controller["serialized_tool_call_tokens"],
        })
    if seen != expected or paired["synchronous"] != paired["asynchronous"]:
        raise ValueError("frozen cell set or matched workload mismatch")

    mode_metrics = {}
    for mode in protocol["execution_modes"]:
        name = mode["name"]
        summary = json.loads((results / name / "mode-summary.json").read_text())
        if summary["task_interface_cell_count"] != 4 or summary["verified_known_fix_count"] != 4:
            raise ValueError(f"mode success denominator mismatch: {name}")
        seconds = summary["execution_and_grading_seconds_after_worktree_materialization"]
        mode_metrics[name] = {
            "worker_count": mode["worker_count"], "verified_cells": 4,
            "wall_seconds": seconds, "cells_per_second": 4 / seconds,
            "queue_inclusive_cell_latencies_seconds": summary[
                "queue_inclusive_cell_latencies_seconds"
            ],
            "serialized_tool_call_tokens_by_interface": summary[
                "serialized_tool_call_tokens_by_interface"
            ],
        }

    recovery = aggregate["crash_recovery"]
    task = protocol["task_cases"]["mlxlm-2004-jamba-moe-routing"]
    recovered = worktrees / "recovery" / "mlxlm-2004-jamba-moe-routing" / task["target"]
    recovered_sha = file_digest(recovered)
    recovery_dir = results / "recovery-after-commit-before-grade"
    recovery_events, last_event = verify_journal(recovery_dir / "events.jsonl")
    recovery_grade = recovery["independent_grade"]
    if (recovery["worker_exitcode"] != -9
            or recovery["expected_upstream_fix_sha256"] != task["fix_target_sha256"]
            or recovery["target_sha256_before_recovery"] != task["fix_target_sha256"]
            or recovery["target_sha256_after_recovery"] != recovered_sha
            or recovered_sha != task["fix_target_sha256"]
            or recovery["duplicate_edit_during_recovery"] is not False
            or recovery_grade["passed"] is not True
            or recovery_grade["grader_sha256"] != task["grader_sha256"]
            or len(recovery_events) != recovery["journal_event_count"]
            or file_digest(recovery_dir / "events.jsonl") != recovery["journal_sha256"]
            or last_event != recovery["journal_final_event_sha256"]):
        raise ValueError("injected-failure recovery audit failed")
    if sum(event["event_type"] == "worker_killed_after_edit_before_independent_grade"
           for event in recovery_events) != 1:
        raise ValueError("recovery event does not show one declared process kill")
    if (file_digest(recovery_dir / "trusted-grade.stdout.log") != recovery_grade["stdout_sha256"]
            or file_digest(recovery_dir / "trusted-grade.stderr.log") != recovery_grade["stderr_sha256"]):
        raise ValueError("recovery raw grader output hash mismatch")

    samples = [json.loads(line) for line in (results / "resources.jsonl").read_text().splitlines()]
    limits = protocol["resource_limits"]
    if (samples[0]["phase"] != "preflight" or samples[-1]["phase"] != "final"
            or any(row["available_memory_bytes"] < limits["minimum_available_memory_bytes"] for row in samples)
            or any(row["free_swap_bytes"] < limits["minimum_free_swap_bytes"] for row in samples)
            or any(row["free_disk_bytes"] < limits["minimum_disk_free_bytes"] for row in samples)
            or any(row["process_tree_rss_bytes"] > limits["max_process_tree_rss_bytes"] for row in samples)
            or any(row["swap_growth_from_preflight_bytes"] > limits["max_swap_growth_bytes"] for row in samples)
            or aggregate["resource_summary"]["abort_reason"] is not None
            or file_digest(results / "resources.jsonl") != aggregate["resource_summary"]["resource_sample_sha256"]):
        raise ValueError("resource telemetry/threshold verification failed")

    sync_s = mode_metrics["synchronous"]["wall_seconds"]
    async_s = mode_metrics["asynchronous"]["wall_seconds"]
    audit = {
        "schema_version": 1,
        "audit_id": args.audit_id,
        "audit_status": "PASS_WITH_JOURNAL_WARNING" if warnings else "PASS",
        "auditor_source_sha256": file_digest(Path(__file__).resolve()),
        "protocol_sha256": protocol_sha,
        "previous_artifact_manifest_sha256": prior_manifest_sha,
        "aggregate_result_sha256": file_digest(all_path),
        "verified_known_fix_cells": len(checks),
        "cells": sorted(checks, key=lambda row: (row["mode"], row["task_id"], row["interface"])),
        "mode_metrics": mode_metrics,
        "descriptive_async_to_sync_wall_ratio": sync_s / async_s,
        "recovery": {
            "worker_exitcode": recovery["worker_exitcode"],
            "journal_event_count": len(recovery_events),
            "recovered_target_sha256": recovered_sha,
            "duplicate_edit": False,
            "trusted_grade_passed": True,
        },
        "resource_summary": aggregate["resource_summary"],
        "warnings": warnings,
        "claim_limit": "Two Development families; deterministic known-fix controller; one workload per mode; one process-kill boundary. Timing is descriptive and does not establish coding-agent quality or asynchronous RL improvement.",
    }
    audit_name = f"independent-audit-{args.audit_id}.json"
    manifest_name = f"artifact-manifest-{args.audit_id}.json"
    manifest_sidecar_name = f"{manifest_name}.sha256"
    audit_path = results / audit_name
    manifest_path = results / manifest_name
    sidecar_path = results / manifest_sidecar_name
    write_new_json(audit_path, audit)
    manifest_files = []
    for path in sorted(item for item in results.rglob("*") if item.is_file()):
        if path == manifest_path or path == sidecar_path:
            continue
        manifest_files.append({
            "path": path.relative_to(results).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        })
    manifest = {
        "schema_version": 2, "audit_id": args.audit_id,
        "protocol_sha256": protocol_sha,
        "previous_artifact_manifest_sha256": prior_manifest_sha,
        "auditor_source_sha256": audit["auditor_source_sha256"],
        "files_including_independent_audit": len(manifest_files),
        "files": manifest_files,
    }
    write_new_json(manifest_path, manifest)
    manifest_sha = file_digest(manifest_path)
    with sidecar_path.open("x", encoding="utf-8") as stream:
        stream.write(manifest_sha + f"  {manifest_name}\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "audit_status": audit["audit_status"],
        "verified_known_fix_cells": audit["verified_known_fix_cells"],
        "warning_count": len(warnings),
        "async_to_sync_wall_ratio": audit["descriptive_async_to_sync_wall_ratio"],
        "auditor_source_sha256": audit["auditor_source_sha256"],
        "previous_artifact_manifest_sha256": prior_manifest_sha,
        "artifact_manifest_sha256": manifest_sha,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
