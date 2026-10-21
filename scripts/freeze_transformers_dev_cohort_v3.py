"""Freeze the isolated RPC-grader protocol after the adapter-leak audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks/mlsys_repo_eval_v5"
TASK_IDS = (
    "transformers-49481-bloom-static-cache-v1",
    "transformers-49361-phi3-longrope-v1",
)
IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze() -> tuple[dict, dict]:
    old_protocol = json.loads((PACK / "protocol-v6.json").read_text(encoding="utf-8"))
    tasks: dict = {}
    manifests: dict = {}
    for task_id in TASK_IDS:
        previous = old_protocol["tasks"][task_id]
        task_root = PACK / previous["directory"]
        artifact_hashes = {
            "task_json_sha256": digest((task_root / "task.json").read_bytes()),
            "task_lock_sha256": digest((task_root / "protocol.lock.json").read_bytes()),
            "baseline_manifest_sha256": digest((task_root / "baseline-source-manifest-v1.json").read_bytes()),
            "legacy_adapter_sha256": digest((task_root / "evaluator/adapter.py").read_bytes()),
            "legacy_entrypoint_sha256": digest((task_root / "evaluator/grade.py").read_bytes()),
        }
        if artifact_hashes["task_json_sha256"] != previous["task_json_sha256"]:
            raise ValueError(f"task descriptor differs from retained v6 source: {task_id}")
        if artifact_hashes["baseline_manifest_sha256"] != previous["baseline_manifest_sha256"]:
            raise ValueError(f"source manifest differs from retained v6 source: {task_id}")
        tasks[task_id] = {
            "directory": previous["directory"],
            "family": previous["family"],
            "source_revision": previous["source_revision"],
            "fixed_revision": previous["fixed_revision"],
            "target_path": previous["target_path"],
            "known_fix_target_sha256": previous["known_fix_target_sha256"],
            "baseline_manifest_sha256": previous["baseline_manifest_sha256"],
            "baseline_snapshot_sha256": previous["baseline_snapshot_sha256"],
            "public_test_sha256": previous["public_test_sha256"],
            "task_artifact_sha256": artifact_hashes,
        }
        manifests[task_id] = {
            "path": f"benchmarks/mlsys_repo_eval_v5/{previous['directory']}/baseline-source-manifest-v1.json",
            "sha256": previous["baseline_manifest_sha256"],
            "workspace_snapshot_sha256": previous["baseline_snapshot_sha256"],
        }
    protocol = {
        "protocol_id": "vare-transformers-fresh-development-cohort-v8",
        "parent_task_protocol_id": old_protocol["protocol_id"],
        "purpose": "Qualify two new Development tasks with host-held assertions and a candidate-only RPC process; v7 evidence serialization failure repaired.",
        "split": "development",
        "repository": "huggingface/transformers",
        "license": "Apache-2.0",
        "container_image": IMAGE,
        "limits": {"timeout_s": 120, "memory_mb": 1900, "cpus": 1, "pids": 32},
        "candidate_boundary": {
            "git_history": False,
            "network": False,
            "candidate_mount": "source-only snapshot; no public test, evaluator, grader, protocol, expected output, or host path",
            "candidate_visible_worker": "generic RPC dispatcher without task assertions or expected values",
            "expected_values": "host process only; never serialized into candidate requests",
            "candidate_target_allowlist": "exact full source inventory; only the task target may differ",
        },
        "grading": {
            "candidate_code_and_hidden_assertions_share_process": False,
            "candidate_code_and_hidden_assertions_share_namespace": False,
            "method": "candidate returns behavioral observations through authenticated VARE candidate frames; host compares outputs to task criteria",
            "worker_sha256_locked": True,
        },
        "tasks": tasks,
        "limitations": [
            "Both tasks are from one repository; their bug families differ but broader repository overlap remains.",
            "Transformers pull requests are public; the model training-data cutoff and prior exposure are unknown.",
            "This task and grader qualification is CPU PyTorch in an arm64 isolated container, not MLX/Metal task execution.",
            "No model rollouts, policy updates, or capability claims are part of task qualification.",
            "Prior v1-v6 qualification attempts are retained as superseded engineering evidence; v7 is the active protocol.",
        ],
    }
    protocol_path = PACK / "protocol-v8.json"
    lock_path = PACK / "protocol-v8.lock.json"
    if protocol_path.exists() or lock_path.exists():
        raise FileExistsError("refusing to overwrite frozen v7 protocol")
    protocol_bytes = (json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode()
    implementation_paths = [
        "scripts/freeze_transformers_dev_cohort_v3.py",
        "scripts/grade_transformers_fresh_dev_v3.py",
        "scripts/transformers_fresh_dev_worker_v1.py",
        "scripts/grade_transformers_phase_a_v1.py",
        "src/vare/isolated_candidate.py",
        "src/vare/container_isolation.py",
    ]
    lock = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": digest(protocol_bytes),
        "parent_protocol_sha256": digest((PACK / "protocol-v6.json").read_bytes()),
        "implementation_sha256": {
            path: digest((ROOT / path).read_bytes()) for path in implementation_paths
        },
        "baseline_manifests": manifests,
    }
    protocol_path.write_bytes(protocol_bytes)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return protocol, lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    protocol, lock = freeze()
    print(json.dumps({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": lock["protocol_sha256"],
        "tasks": list(protocol["tasks"]),
    }, sort_keys=True))
