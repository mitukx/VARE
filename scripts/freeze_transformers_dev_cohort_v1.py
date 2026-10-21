"""Write immutable hashes for the task-only Transformers v5 cohort."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import subprocess
import argparse
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_transformers_dev_cohort_v1 import TASKS, sha256

PACK = ROOT / "benchmarks/mlsys_repo_eval_v5"
IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)
QUALIFIED_TASK_IDS = (
    "transformers-49481-bloom-static-cache-v1",
    "transformers-49361-phi3-longrope-v1",
)


def freeze(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    tasks: dict[str, Any] = {}
    baseline_manifests: dict[str, Any] = {}
    for task_id in QUALIFIED_TASK_IDS:
        identity = TASKS[task_id]
        task_root = PACK / identity["directory"]
        task_bytes = (task_root / "task.json").read_bytes()
        task = json.loads(task_bytes)
        manifest_bytes = (task_root / "baseline-source-manifest-v1.json").read_bytes()
        manifest_sha = sha256(manifest_bytes)
        manifest = json.loads(manifest_bytes)
        adapter_sha = sha256((task_root / "evaluator/adapter.py").read_bytes())
        task_lock_sha = sha256((task_root / "protocol.lock.json").read_bytes())
        entrypoint_sha = sha256((task_root / "evaluator/grade.py").read_bytes())
        target = identity["target"]
        known_fix_target = subprocess.run(
            ["git", "-C", str(repository), "show", f"{identity['fixed']}:{target}"],
            check=True, capture_output=True, timeout=30,
        ).stdout
        if (task.get("task_id") != task_id
                or task.get("source", {}).get("base_revision") != identity["base"]
                or task.get("source", {}).get("fixed_revision") != identity["fixed"]
                or manifest.get("task_id") != task_id
                or manifest.get("source_revision") != identity["base"]
                or target not in manifest.get("files", {})):
            raise ValueError(f"task provenance is inconsistent: {task_id}")
        baseline_manifests[task_id] = {
            "path": f"benchmarks/mlsys_repo_eval_v5/{identity['directory']}/baseline-source-manifest-v1.json",
            "sha256": manifest_sha,
            "workspace_snapshot_sha256": manifest["workspace_snapshot_sha256"],
        }
        tasks[task_id] = {
            "directory": identity["directory"],
            "family": identity["family"],
            "source_revision": identity["base"],
            "fixed_revision": identity["fixed"],
            "target_path": target,
            "known_fix_target_sha256": sha256(known_fix_target),
            "known_fix_application": "replace only target bytes in the pinned baseline snapshot",
            "baseline_manifest_sha256": manifest_sha,
            "baseline_snapshot_sha256": manifest["workspace_snapshot_sha256"],
            "task_json_sha256": sha256(task_bytes),
            "task_lock_sha256": task_lock_sha,
            "entrypoint_sha256": entrypoint_sha,
            "public_test_sha256": manifest["files"]["test_public.py"]["sha256"],
            "adapter_sha256": adapter_sha,
        }
    protocol = {
        "protocol_id": "vare-transformers-fresh-development-cohort-v6",
        "purpose": "Qualify three source-pinned Development tasks with isolated behavioral grading.",
        "split": "development",
        "repository": "huggingface/transformers",
        "license": "Apache-2.0",
        "container_image": IMAGE,
        "limits": {"timeout_s": 120, "memory_mb": 1900, "cpus": 1, "pids": 32},
        "candidate_boundary": {
            "git_history": False,
            "network": False,
            "editable_files_per_task": 1,
            "public_test_read_only": True,
            "grader_in_candidate_namespace": False,
            "candidate_source_validation": "exact full inventory and SHA-256; only target file may differ",
        },
        "tasks": tasks,
        "limitations": [
            "All tasks come from one repository and may share broad Transformers architecture.",
            "Phi3 LongRoPE and legacy RoPE configuration share the RoPE domain and are not independent families.",
            "The model training-data cutoff and exact pretraining exposure are unknown; PR chronology is not a non-contamination proof.",
            "This protocol qualifies graders and tasks; it does not establish model capability or policy improvement.",
        ],
    }
    protocol_path = PACK / "protocol-v6.json"
    lock_path = PACK / "protocol-v6.lock.json"
    if protocol_path.exists() or lock_path.exists():
        raise FileExistsError("refusing to replace an existing frozen task protocol")
    protocol_bytes = (json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode()
    implementation_paths = [
        "scripts/grade_transformers_fresh_dev_v1.py",
        "scripts/grade_transformers_phase_a_v1.py",
        "src/vare/isolated_candidate.py",
        "src/vare/container_isolation.py",
    ]
    lock = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_bytes),
        "implementation_sha256": {
            name: sha256((ROOT / name).read_bytes()) for name in implementation_paths
        },
        "baseline_manifests": baseline_manifests,
    }
    protocol_path.write_bytes(protocol_bytes)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return protocol, lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    protocol, lock = freeze(args.repository.resolve())
    print(json.dumps({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": lock["protocol_sha256"],
        "tasks": list(protocol["tasks"]),
    }, sort_keys=True))
