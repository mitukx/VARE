#!/usr/bin/env python3
"""Audit the frozen MLSys model-update readiness inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_status(task: dict[str, Any], split: dict[str, Any]) -> str:
    task_id = task.get("id")
    if task_id in (split.get("train") or []):
        return "train"
    if task_id in (split.get("heldout") or []):
        return "heldout"
    return str(task.get("metadata", {}).get("split", "unassigned"))


def audit_protocol(protocol: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    eligible_hashes: set[str] = set()
    task_entry_count = 0
    task_hashes: set[str] = set()
    checked_packs: list[dict[str, Any]] = []

    for pack in protocol["observed_inventory"]["packs"]:
        pack_errors_before = len(errors)
        pack_root = repo_root / pack["pack_root"]
        manifest_path = pack_root / "manifest.json"
        split_path = pack_root / "split.json"
        embedded_revision = pack.get("embedded_repo_commit", pack["pack_commit"])
        try:
            subprocess.run(
                ["git", "cat-file", "-e", f"{embedded_revision}^{{commit}}"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            errors.append(f"pack commit is unavailable: {pack['pack_id']}")
        else:
            try:
                subprocess.run(
                    [
                        "git",
                        "merge-base",
                        "--is-ancestor",
                        embedded_revision,
                        protocol["audited_source_commit"],
                    ],
                    cwd=repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError):
                errors.append(f"pack commit is not an ancestor of audited source: {pack['pack_id']}")
        if not manifest_path.is_file() or _sha256(manifest_path) != pack["manifest_sha256"]:
            errors.append(f"manifest hash mismatch: {pack['pack_id']}")
            continue
        if not split_path.is_file() or _sha256(split_path) != pack["split_sha256"]:
            errors.append(f"split hash mismatch: {pack['pack_id']}")
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_entries = manifest.get("files")
        if not isinstance(manifest_entries, list):
            errors.append(f"manifest file list is invalid: {pack['pack_id']}")
            continue
        listed_paths: set[str] = set()
        for item in manifest_entries:
            relative = item.get("path") if isinstance(item, dict) else None
            if not isinstance(relative, str):
                errors.append(f"manifest path is invalid: {pack['pack_id']}")
                continue
            relative_path = PurePosixPath(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                errors.append(f"manifest path escapes pack root: {pack['pack_id']}:{relative}")
                continue
            listed_paths.add(relative)
            file_path = pack_root.joinpath(*relative_path.parts)
            if not file_path.is_file():
                errors.append(f"manifest file is missing: {pack['pack_id']}:{relative}")
                continue
            if file_path.stat().st_size != item.get("bytes") or _sha256(file_path) != item.get(
                "sha256"
            ):
                errors.append(f"manifest file hash mismatch: {pack['pack_id']}:{relative}")

        actual_files = {
            path.relative_to(pack_root).as_posix()
            for path in pack_root.rglob("*")
            if path.is_file()
            and ".git" not in path.relative_to(pack_root).parts
            and "__pycache__" not in path.relative_to(pack_root).parts
            and path.suffix != ".pyc"
            and path.name != "manifest.json"
        }
        if listed_paths != actual_files:
            errors.append(f"manifest file inventory mismatch: {pack['pack_id']}")

        split = json.loads(split_path.read_text(encoding="utf-8"))
        actual_paths = sorted((pack_root / "tasks").glob("*/task.json"))
        expected_paths = sorted(
            repo_root / entry["task_json_path"] for entry in pack["tasks"]
        )
        if actual_paths != expected_paths:
            errors.append(f"task inventory path mismatch: {pack['pack_id']}")

        for entry in pack["tasks"]:
            task_entry_count += 1
            if entry["pack_commit"] != pack["pack_commit"]:
                errors.append(f"source pack commit mismatch: {entry['task_id']}")
            task_path = repo_root / entry["task_json_path"]
            if not task_path.is_file():
                errors.append(f"missing task definition: {entry['task_json_path']}")
                continue
            task_hash = _sha256(task_path)
            if task_hash != entry["task_json_sha256"]:
                errors.append(f"task hash mismatch: {entry['task_json_path']}")
                continue
            task_hashes.add(task_hash)
            task = json.loads(task_path.read_text(encoding="utf-8"))
            if task.get("id") != entry["task_id"]:
                errors.append(f"task id mismatch: {entry['task_json_path']}")
            if task.get("family") != entry["family"]:
                errors.append(f"task family mismatch: {entry['task_json_path']}")
            if task.get("metadata", {}).get("task_type") != entry["task_type"]:
                errors.append(f"task type mismatch: {entry['task_json_path']}")
            if _split_status(task, split) != entry["split_status"]:
                errors.append(f"split membership mismatch: {entry['task_json_path']}")
            if task.get("metadata", {}).get("historical_upstream_commit", False) != entry[
                "historical_upstream_commit"
            ]:
                errors.append(f"source provenance mismatch: {entry['task_json_path']}")
            if entry["eligible_for_final_heldout"]:
                eligible_hashes.add(task_hash)
            elif not entry["exclusion_reason"]:
                errors.append(f"missing exclusion reason: {entry['task_json_path']}")

        checked_packs.append(
            {
                "pack_id": pack["pack_id"],
                "source_pack_commit": pack["pack_commit"],
                "embedded_repo_commit": embedded_revision,
                "manifest_sha256": _sha256(manifest_path),
                "manifest_file_count": len(manifest_entries),
                "split_sha256": _sha256(split_path),
                "task_count": len(pack["tasks"]),
                "checks_passed": len(errors) == pack_errors_before,
            }
        )

    target = protocol["readiness_gate"]["required_unique_eligible_heldout_tasks"]
    eligible_count = len(eligible_hashes)
    inventory_ceiling = len(task_hashes)
    reproduced_status = (
        "NO_GO"
        if inventory_ceiling < target or eligible_count < target
        else "READY_FOR_SEPARATE_PROTOCOL"
    )
    expected_decision = protocol["decision"]
    if inventory_ceiling != expected_decision["unique_task_definitions_by_task_json_sha256"]:
        errors.append("unique task count differs from the frozen inventory")
    if (inventory_ceiling < target) != expected_decision["inventory_alone_fails_minimum"]:
        errors.append("inventory ceiling no-go classification differs from the frozen decision")
    if eligible_count != expected_decision["eligible_unique_heldout_tasks"]:
        errors.append("eligible task count differs from the frozen decision")
    if reproduced_status != expected_decision["status"]:
        errors.append("readiness status differs from the frozen decision")

    source_commit = protocol["audited_source_commit"]
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        source_commit_present = True
    except (OSError, subprocess.CalledProcessError):
        source_commit_present = False
        errors.append("audited source commit is unavailable")

    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        ).hexdigest(),
        "audited_source_commit": source_commit,
        "source_commit_present": source_commit_present,
        "packs": checked_packs,
        "task_entries_checked": task_entry_count,
        "unique_task_definitions_by_sha256": len(task_hashes),
        "eligible_unique_heldout_tasks": eligible_count,
        "maximum_possible_unique_heldout_tasks": inventory_ceiling,
        "inventory_alone_fails_minimum": inventory_ceiling < target,
        "required_unique_heldout_tasks": target,
        "status": reproduced_status,
        "audit_passed": not errors,
        "errors": errors,
        "claim_boundary": expected_decision["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/mlsys_actual_model_update_no_go_v1.lock.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/mlsys-actual-model-update-readiness-v1/audit.json")
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    protocol_path = repo_root / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = audit_protocol(protocol, repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["audit_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
