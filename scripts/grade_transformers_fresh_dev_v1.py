"""Process-isolated grader for the frozen v5 Transformers Development tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grade_transformers_phase_a_v1 import (
    _inspect_frozen_image,
    _relative_name,
    _scan_candidate,
    _stage_verified_snapshot,
)
from vare.isolated_candidate import FunctionCall, grade_calls

PACK = ROOT / "benchmarks/mlsys_repo_eval_v5"
PROTOCOL = PACK / "protocol-v6.json"
LOCK = PACK / "protocol-v6.lock.json"
TASK_IDS = (
    "transformers-49481-bloom-static-cache-v1",
    "transformers-49361-phi3-longrope-v1",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_frozen_protocol() -> tuple[dict[str, Any], dict[str, Any], str]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    protocol_bytes = PROTOCOL.read_bytes()
    protocol_sha = sha256_bytes(protocol_bytes)
    if protocol_sha != lock.get("protocol_sha256"):
        raise ValueError("frozen task protocol hash differs from its lock")
    protocol = json.loads(protocol_bytes)
    if protocol.get("protocol_id") != lock.get("protocol_id"):
        raise ValueError("frozen task protocol identity differs from its lock")
    for relative, expected in lock.get("implementation_sha256", {}).items():
        path = ROOT / _relative_name(relative)
        if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"grading implementation differs from its lock: {relative}")
    return protocol, lock, protocol_sha


def load_task_artifacts(
    task_id: str, protocol: dict[str, Any], lock: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, bytes]:
    if task_id not in TASK_IDS:
        raise ValueError("unknown fresh Development task")
    spec = protocol.get("tasks", {}).get(task_id)
    if not isinstance(spec, dict):
        raise ValueError("task is missing from the frozen protocol")
    task_root = PACK / spec["directory"]
    task_bytes = (task_root / "task.json").read_bytes()
    manifest_bytes = (task_root / "baseline-source-manifest-v1.json").read_bytes()
    task_lock_bytes = (task_root / "protocol.lock.json").read_bytes()
    adapter_path = task_root / "evaluator/adapter.py"
    entrypoint_path = task_root / "evaluator/grade.py"
    if sha256_bytes(task_bytes) != spec.get("task_json_sha256"):
        raise ValueError("task descriptor differs from the frozen protocol")
    manifest_entry = lock.get("baseline_manifests", {}).get(task_id)
    if not isinstance(manifest_entry, dict) or sha256_bytes(manifest_bytes) != manifest_entry.get("sha256"):
        raise ValueError("baseline source manifest differs from the frozen protocol")
    adapter_hash = sha256_bytes(adapter_path.read_bytes())
    if adapter_hash != spec.get("adapter_sha256"):
        raise ValueError("trusted grader adapter differs from the frozen protocol")
    task_lock_hash = sha256_bytes(task_lock_bytes)
    if task_lock_hash != spec.get("task_lock_sha256"):
        raise ValueError("task lock differs from the frozen protocol")
    entrypoint_hash = sha256_bytes(entrypoint_path.read_bytes())
    if entrypoint_hash != spec.get("entrypoint_sha256"):
        raise ValueError("task grader entry point differs from the frozen protocol")
    task = json.loads(task_bytes)
    task_lock = json.loads(task_lock_bytes)
    manifest = json.loads(manifest_bytes)
    if (task.get("task_id") != task_id
            or task.get("source", {}).get("base_revision") != spec.get("source_revision")
            or task.get("source", {}).get("fixed_revision") != spec.get("fixed_revision")
            or manifest.get("workspace_snapshot_sha256") != spec.get("baseline_snapshot_sha256")):
        raise ValueError("task source identity differs from the frozen protocol")
    if (task_lock.get("task_json_sha256") != spec["task_json_sha256"]
            or task_lock.get("baseline_manifest_sha256") != manifest_entry["sha256"]):
        raise ValueError("inner task lock provenance differs from the frozen protocol")
    return task, manifest, adapter_path, task_bytes


def grade(task_id: str, workspace: Path, image: str) -> dict[str, Any]:
    protocol, lock, protocol_sha = load_frozen_protocol()
    expected_image = protocol.get("container_image")
    if not isinstance(expected_image, str) or image != expected_image or "@sha256:" not in image:
        raise ValueError("grading image must exactly match the frozen protocol digest")
    task, manifest, adapter_path, _ = load_task_artifacts(task_id, protocol, lock)
    spec = protocol["tasks"][task_id]
    files, file_count, directory_count = _scan_candidate(
        workspace, manifest, spec["target_path"],
    )
    image_id = _inspect_frozen_image(image)

    staging_root = Path.home() / ".cache"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vare-fresh-dev-subject-", dir=staging_root) as directory:
        subject = Path(directory) / "subject"
        subject.mkdir()
        _stage_verified_snapshot(subject, manifest, files)
        adapter_name = "vare_fresh_dev_adapter.py"
        adapter_bytes = adapter_path.read_bytes()
        (subject / adapter_name).write_bytes(adapter_bytes)
        call = FunctionCall(
            case_id=task_id,
            module="vare_fresh_dev_adapter",
            function="grade",
            args=[],
            expected=True,
        )
        report = grade_calls(
            subject,
            [call],
            task_kind="correctness",
            image=image,
            timeout_s=float(protocol["limits"]["timeout_s"]),
            memory_mb=int(protocol["limits"]["memory_mb"]),
            discard_candidate_stdout=True,
        )
    result = report.to_dict()
    result.update({
        "task_id": task_id,
        "candidate_target": spec["target_path"],
        "expected_held_by_parent": True,
        "grader_mounted": False,
        "candidate_git_history_present": False,
        "candidate_workspace_inventory_verified": True,
        "candidate_file_count": file_count,
        "candidate_directory_count": directory_count,
        "candidate_target_sha256": sha256_bytes(files[spec["target_path"]]),
        "baseline_manifest_sha256": lock["baseline_manifests"][task_id]["sha256"],
        "task_json_sha256": spec["task_json_sha256"],
        "trusted_adapter_sha256": spec["adapter_sha256"],
        "task_lock_sha256": spec["task_lock_sha256"],
        "task_grader_entrypoint_sha256": spec["entrypoint_sha256"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "grader_implementation_sha256": lock["implementation_sha256"]["scripts/grade_transformers_fresh_dev_v1.py"],
        "grading_image": image,
        "container_image_id": image_id,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("refusing to overwrite evidence")
    workspace = Path(os.path.abspath(os.fspath(args.workspace)))
    output = Path(os.path.abspath(os.fspath(args.output)))
    if output == workspace or workspace in output.parents:
        parser.error("evidence output must be outside the candidate workspace")
    result = grade(args.task_id, workspace, args.image)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
