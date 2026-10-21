"""Grade v5 Development tasks with host-only assertions and a generic RPC worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
PACK = ROOT / "benchmarks/mlsys_repo_eval_v5"
PROTOCOL = PACK / "protocol-v7.json"
LOCK = PACK / "protocol-v7.lock.json"
WORKER = ROOT / "scripts/transformers_fresh_dev_worker_v1.py"
TASK_IDS = (
    "transformers-49481-bloom-static-cache-v1",
    "transformers-49361-phi3-longrope-v1",
)
IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)
BLOOM_INPUT = {
    "config": {"vocab_size": 31, "hidden_size": 16, "n_layer": 1, "n_head": 2},
    "tokens": [[1, 2, 3]],
    "max_new_tokens": 3,
    "seed": 11,
}
BLOOM_TOKENS: list[list[int]] = [[1, 2, 3]]
BLOOM_MAX_NEW_TOKENS = 3
PHI3_INPUT = {
    "config": {
        "vocab_size": 99,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "original_max_position_embeddings": 8,
        "max_position_embeddings": 32,
        "eos_token_id": None,
        "pad_token_id": 0,
        "rope_parameters": {
            "rope_type": "longrope",
            "rope_theta": 10000.0,
            "short_factor": [1.0] * 4,
            "long_factor": [2.0] * 4,
            "original_max_position_embeddings": 8,
            "factor": 4.0,
        },
    },
    "tokens": [[2, 7, 11, 13, 17, 19, 23]],
    "seed": 47,
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_protocol() -> tuple[dict[str, Any], dict[str, Any], str]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    protocol_bytes = PROTOCOL.read_bytes()
    protocol_sha = sha256(protocol_bytes)
    if protocol_sha != lock.get("protocol_sha256"):
        raise ValueError("frozen v7 protocol hash differs from its lock")
    protocol = json.loads(protocol_bytes)
    if protocol.get("protocol_id") != lock.get("protocol_id"):
        raise ValueError("frozen v7 protocol identity differs from its lock")
    for relative, expected in lock.get("implementation_sha256", {}).items():
        path = ROOT / _relative_name(relative)
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f"grader implementation differs from its lock: {relative}")
    return protocol, lock, protocol_sha


def _load_task(task_id: str, protocol: dict[str, Any], lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if task_id not in TASK_IDS:
        raise ValueError("unknown fresh Development task")
    spec = protocol.get("tasks", {}).get(task_id)
    if not isinstance(spec, dict):
        raise ValueError("task is missing from the frozen v7 protocol")
    task_root = PACK / spec["directory"]
    task_bytes = (task_root / "task.json").read_bytes()
    task_lock_bytes = (task_root / "protocol.lock.json").read_bytes()
    manifest_bytes = (task_root / "baseline-source-manifest-v1.json").read_bytes()
    task_lock = json.loads(task_lock_bytes)
    task = json.loads(task_bytes)
    manifest = json.loads(manifest_bytes)
    hashes = {
        "task_json_sha256": sha256(task_bytes),
        "task_lock_sha256": sha256(task_lock_bytes),
        "baseline_manifest_sha256": sha256(manifest_bytes),
        "legacy_adapter_sha256": sha256((task_root / "evaluator/adapter.py").read_bytes()),
        "legacy_entrypoint_sha256": sha256((task_root / "evaluator/grade.py").read_bytes()),
    }
    if hashes != spec.get("task_artifact_sha256"):
        raise ValueError("frozen task metadata or legacy artifact hashes differ")
    manifest_entry = lock.get("baseline_manifests", {}).get(task_id)
    if not isinstance(manifest_entry, dict) or manifest_entry.get("sha256") != hashes["baseline_manifest_sha256"]:
        raise ValueError("baseline source manifest differs from protocol lock")
    if (task.get("task_id") != task_id
            or task.get("source", {}).get("base_revision") != spec.get("source_revision")
            or task.get("source", {}).get("fixed_revision") != spec.get("fixed_revision")
            or task.get("source", {}).get("snapshot_sha256") != manifest["workspace_snapshot_sha256"]):
        raise ValueError("task source provenance differs from v7 protocol")
    if task_lock.get("task_json_sha256") != hashes["task_json_sha256"]:
        raise ValueError("task's inner lock does not bind its task descriptor")
    if task_lock.get("baseline_manifest_sha256") != hashes["baseline_manifest_sha256"]:
        raise ValueError("task's inner lock does not bind its source manifest")
    return task, manifest


def _all_close(left: Any, right: Any, *, rtol: float, atol: float) -> bool:
    if type(left) is list and type(right) is list:
        return len(left) == len(right) and all(
            _all_close(a, b, rtol=rtol, atol=atol) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        a, b = float(left), float(right)
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return type(left) is type(right) and left == right


def _run_candidate(task_id: str, subject: Path, image: str, protocol: dict[str, Any]) -> dict[str, Any]:
    runner = DockerIsolationRunner()
    limits = protocol["limits"]
    if task_id == TASK_IDS[0]:
        request = BLOOM_INPUT
        function = "bloom_static_cache"
        args = [request["config"], request["tokens"], request["max_new_tokens"], request["seed"]]
    else:
        request = PHI3_INPUT
        function = "phi3_longrope_logits"
        args = [request["config"], request["tokens"], request["seed"]]
    value, elapsed, failure = _one_call(
        subject,
        "transformers_fresh_dev_worker_v1",
        function,
        args,
        {},
        float(limits["timeout_s"]),
        image,
        runner,
        memory_mb=int(limits["memory_mb"]),
        discard_candidate_stdout=True,
    )
    return {"value": value, "elapsed_seconds": elapsed, "failure": failure}


def _verify_result(task_id: str, result: dict[str, Any]) -> tuple[bool, str | None, dict[str, Any]]:
    if result["failure"] is not None:
        return False, str(result["failure"]), {}
    value = result["value"]
    if isinstance(value, CandidateException):
        return False, f"candidate_exception:{value.type_name}", {}
    if not isinstance(value, dict):
        return False, "malformed_worker_value", {}
    if task_id == TASK_IDS[0]:
        output = value.get("output_ids")
        tokens = BLOOM_TOKENS
        expected_length = len(tokens[0]) + BLOOM_MAX_NEW_TOKENS
        if (not isinstance(output, list) or len(output) != len(tokens)
                or any(not isinstance(row, list) or len(row) != expected_length for row in output)
                or output[0][:len(tokens[0])] != tokens[0]):
            return False, "generation_shape_or_prefix_mismatch", {"output_ids": output}
        return True, None, {"output_ids": output, "output_shape": [len(output), expected_length]}

    observations = value.get("observations")
    if not isinstance(observations, list) or len(observations) != 2:
        return False, "missing_window_observations", {}
    comparisons = []
    for observation in observations:
        steps = observation.get("comparisons") if isinstance(observation, dict) else None
        if not isinstance(steps, list) or len(steps) != 4:
            return False, "missing_generation_steps", {}
        for step in steps:
            if not isinstance(step, dict) or not _all_close(
                step.get("cached"), step.get("full_prefix"), rtol=1e-4, atol=1e-4,
            ):
                return False, "cached_full_prefix_logits_mismatch", {}
        comparisons.extend(steps)
    summary = {
        "sliding_windows": [row.get("sliding_window") for row in observations],
        "comparison_count": len(comparisons),
        "rtol": 1e-4,
        "atol": 1e-4,
    }
    return True, None, summary


def grade(task_id: str, workspace: Path, image: str) -> dict[str, Any]:
    protocol, lock, protocol_sha = _load_protocol()
    if image != protocol.get("container_image") or image != IMAGE:
        raise ValueError("runtime image must exactly match the frozen v7 digest")
    task, manifest = _load_task(task_id, protocol, lock)
    spec = protocol["tasks"][task_id]
    files, file_count, directory_count = _scan_candidate(workspace, manifest, spec["target_path"])
    image_id = _inspect_frozen_image(image)
    staging_root = Path.home() / ".cache"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vare-fresh-v7-subject-", dir=staging_root) as folder:
        subject = Path(folder) / "subject"
        subject.mkdir()
        candidate_subject_files = {name: data for name, data in files.items() if name != "test_public.py"}
        _stage_verified_snapshot(subject, manifest, candidate_subject_files)
        shutil.copyfile(WORKER, subject / WORKER.name)
        # No task adapter, expected output, test file, or host grader is staged.
        if (subject / "evaluator").exists() or (subject / "test_public.py").exists():
            raise ValueError("candidate subject unexpectedly contains evaluator or public tests")
        worker_result = _run_candidate(task_id, subject, image, protocol)
    passed, failure, verified = _verify_result(task_id, worker_result)
    target_bytes = files[spec["target_path"]]
    payload = {
        "task_id": task_id,
        "passed": passed,
        "failure": failure,
        "verified": verified,
        "elapsed_seconds": worker_result["elapsed_seconds"],
        "candidate_file_count": file_count,
        "candidate_directory_count": directory_count,
        "candidate_target_sha256": sha256(target_bytes),
        "baseline_manifest_sha256": lock["baseline_manifests"][task_id]["sha256"],
        "task_json_sha256": spec["task_artifact_sha256"]["task_json_sha256"],
        "task_id_revision": task["source"]["base_revision"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "grader_implementation_sha256": lock["implementation_sha256"]["scripts/grade_transformers_fresh_dev_v2.py"],
        "worker_sha256": lock["implementation_sha256"]["scripts/transformers_fresh_dev_worker_v1.py"],
        "grading_image": image,
        "container_image_id": image_id,
        "grader_files_in_candidate_subject": False,
        "candidate_expected_output_access": False,
        "candidate_workspace_inventory_verified": True,
        "candidate_observation": worker_result["value"],
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
        parser.error("evidence must be outside the candidate workspace")
    result = grade(args.task_id, workspace, args.image)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
