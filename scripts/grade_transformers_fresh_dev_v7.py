"""Grade the four-task v6 cohort under corrected protocol v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from scripts.grade_transformers_phase_a_v1 import (
    _inspect_frozen_image,
    _relative_name,
    _scan_candidate,
    _stage_verified_snapshot,
)
from vare.container_isolation import DockerIsolationRunner
from vare.isolated_candidate import CandidateException, _one_call

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks/mlsys_repo_eval_v6"
PROTOCOL = PACK / "protocol-v2.json"
LOCK = PACK / "protocol-v2.lock.json"
WORKER = ROOT / "scripts/transformers_fresh_dev_worker_v3.py"
IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)
TASK_IDS = (
    "transformers-49315-watermark-repeat-v1",
    "transformers-49515-rope-setter-v1",
    "transformers-49731-generation-device-type-v1",
    "transformers-49767-janus-background-v1",
)
WORKERS = {
    TASK_IDS[0]: ("watermark_repeated_ngrams", []),
    TASK_IDS[1]: ("rope_legacy_setter", []),
    TASK_IDS[2]: ("move_to_device_preserves_output_type", []),
    TASK_IDS[3]: ("janus_background_color", []),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_protocol() -> tuple[dict[str, Any], dict[str, Any], str]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    protocol_bytes = PROTOCOL.read_bytes()
    protocol_sha = sha256(protocol_bytes)
    if protocol_sha != lock.get("protocol_sha256"):
        raise ValueError("frozen v6 protocol v2 hash differs from its lock")
    protocol = json.loads(protocol_bytes)
    if protocol.get("protocol_id") != lock.get("protocol_id"):
        raise ValueError("frozen v6 protocol v2 identity differs from its lock")
    for relative, expected in lock.get("implementation_sha256", {}).items():
        path = ROOT / _relative_name(relative)
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f"grader implementation differs from its lock: {relative}")
    return protocol, lock, protocol_sha


def _load_task(task_id: str, protocol: dict[str, Any], lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if task_id not in TASK_IDS:
        raise ValueError("unknown v6 Development task")
    spec = protocol.get("tasks", {}).get(task_id)
    if not isinstance(spec, dict):
        raise ValueError("task is missing from frozen v6 protocol")
    task_root = PACK / task_id
    task_bytes = (task_root / "task.json").read_bytes()
    manifest_bytes = (task_root / "baseline-source-manifest-v1.json").read_bytes()
    task = json.loads(task_bytes)
    manifest = json.loads(manifest_bytes)
    actual = {
        "task_json_sha256": sha256(task_bytes),
        "baseline_manifest_sha256": sha256(manifest_bytes),
        "public_test_sha256": sha256((task_root / "test_public.py").read_bytes()),
    }
    if actual != spec.get("task_artifact_sha256"):
            raise ValueError("frozen v6 task artifact hash differs")
    manifest_lock = lock.get("baseline_manifests", {}).get(task_id)
    if not isinstance(manifest_lock, dict) or manifest_lock.get("sha256") != actual["baseline_manifest_sha256"]:
        raise ValueError("baseline source manifest differs from v6 protocol lock")
    if (task.get("task_id") != task_id
            or task.get("source", {}).get("base_revision") != spec.get("source_revision")
            or task.get("source", {}).get("fixed_revision") != spec.get("fixed_revision")
            or task.get("source", {}).get("baseline_workspace_sha256") != manifest.get("workspace_snapshot_sha256")):
        raise ValueError("task source provenance differs from v6 protocol")
    return task, manifest


def _verify(task_id: str, value: Any) -> tuple[bool, str | None, dict[str, Any]]:
    if isinstance(value, CandidateException):
        return False, f"candidate_exception:{value.type_name}", {}
    if not isinstance(value, dict):
        return False, "malformed_worker_value", {}
    if task_id == TASK_IDS[0]:
        scored, green = value.get("num_tokens_scored"), value.get("num_green_tokens")
        ok = (scored == [1.0] and isinstance(green, list) and len(green) == 1
              and isinstance(green[0], (int, float)) and 0.0 <= green[0] <= 1.0)
        return ok, None if ok else "repeated_ngram_count_mismatch", {
            "expected_unique_windows": 1, "scored": scored, "green": green,
        }
    if task_id == TASK_IDS[1]:
        expected = {
            "rope_type": "yarn", "factor": 4.0, "future_warning": True,
            "none_future_warning": True, "none_preserved": True,
        }
        ok = value == expected
        return ok, None if ok else "legacy_rope_normalization_mismatch", value
    if task_id == TASK_IDS[2]:
        expected = {
            "output_type": "BaseModelOutput", "hidden_states_type": "tuple",
            "last_hidden_state_shape": [1, 2, 4], "nested_values_preserved": True,
        }
        ok = value == expected
        return ok, None if ok else "structured_output_type_or_value_mismatch", value
    expected = {
        "default_mean": [0.48145466, 0.4578275, 0.40821073],
        "default_background": [122, 116, 104],
        "custom_background": [127, 63, 0],
    }
    ok = value == expected
    return ok, None if ok else "janus_background_color_mismatch", value


def grade(task_id: str, workspace: Path, image: str) -> dict[str, Any]:
    protocol, lock, protocol_sha = _load_protocol()
    if image != protocol.get("container_image") or image != IMAGE:
        raise ValueError("runtime image must exactly match the frozen v6 protocol v2 digest")
    task, manifest = _load_task(task_id, protocol, lock)
    spec = protocol["tasks"][task_id]
    files, file_count, directory_count = _scan_candidate(workspace, manifest, spec["target_path"])
    image_id = _inspect_frozen_image(image)
    work_root = Path.home() / ".cache"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vare-fresh-v6v2-subject-", dir=work_root) as folder:
        subject = Path(folder) / "subject"
        subject.mkdir()
        subject_files = {name: data for name, data in files.items() if name != "test_public.py"}
        _stage_verified_snapshot(subject, manifest, subject_files)
        shutil.copyfile(WORKER, subject / WORKER.name)
        if (subject / "test_public.py").exists() or (subject / "evaluator").exists():
            raise ValueError("grader or tests leaked into candidate subject")
        function, args = WORKERS[task_id]
        value, elapsed, failure = _one_call(
            subject, "transformers_fresh_dev_worker_v3", function, args, {},
            float(protocol["limits"]["timeout_s"]), image, DockerIsolationRunner(),
            memory_mb=int(protocol["limits"]["memory_mb"]), discard_candidate_stdout=True,
            cpus=float(protocol["limits"]["cpus"]), pids_limit=int(protocol["limits"]["pids"]),
        )
    if failure is not None:
        passed, reason, summary = False, str(failure), {}
    else:
        passed, reason, summary = _verify(task_id, value)
    target = spec["target_path"]
    payload = {
        "task_id": task_id,
        "passed": passed,
        "failure": reason,
        "verified": summary,
        "elapsed_seconds": elapsed,
        "candidate_file_count": file_count,
        "candidate_directory_count": directory_count,
        "candidate_target_sha256": sha256(files[target]),
        "baseline_manifest_sha256": lock["baseline_manifests"][task_id]["sha256"],
        "task_json_sha256": sha256((PACK / task_id / "task.json").read_bytes()),
        "task_source_revision": task["source"]["base_revision"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "grader_implementation_sha256": lock["implementation_sha256"]["scripts/grade_transformers_fresh_dev_v7.py"],
        "worker_sha256": lock["implementation_sha256"]["scripts/transformers_fresh_dev_worker_v3.py"],
        "grading_image": image,
        "container_image_id": image_id,
        "container_limits": protocol["limits"],
        "grader_files_in_candidate_subject": False,
        "candidate_workspace_inventory_verified": True,
        "candidate_observation": value if not isinstance(value, CandidateException) else {"exception_type": value.type_name},
    }
    return payload


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
        parser.error("evidence must be outside candidate workspace")
    result = grade(args.task_id, workspace, args.image)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
