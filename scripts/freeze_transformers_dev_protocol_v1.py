"""Freeze the v6 task membership, protocol bytes, manifests and grader code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.prepare_transformers_dev_cohort_v2 import PACK, TASKS

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PACK / "protocol-v1.json"
LOCK = PACK / "protocol-v1.lock.json"
IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze() -> dict[str, object]:
    if PROTOCOL.exists() or LOCK.exists():
        raise FileExistsError("refusing to rewrite frozen v6 protocol")
    tasks: dict[str, dict[str, object]] = {}
    manifests: dict[str, dict[str, str]] = {}
    for task_id, spec in TASKS.items():
        directory = PACK / task_id
        task_path = directory / "task.json"
        manifest_path = directory / "baseline-source-manifest-v1.json"
        public_path = directory / "test_public.py"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = {
            "task_json_sha256": sha256(task_path.read_bytes()),
            "baseline_manifest_sha256": sha256(manifest_path.read_bytes()),
            "public_test_sha256": sha256(public_path.read_bytes()),
        }
        tasks[task_id] = {
            "directory": task_id,
            "source_revision": spec["base"],
            "fixed_revision": spec["fixed"],
            "target_path": spec["target"],
            "worker_function": spec["worker"],
            "task_artifact_sha256": hashes,
        }
        manifests[task_id] = {
            "sha256": hashes["baseline_manifest_sha256"],
            "workspace_snapshot_sha256": manifest["workspace_snapshot_sha256"],
        }
    protocol = {
        "protocol_id": "vare-transformers-fresh-development-v6-2026-10-21",
        "task_pack": "mlsys_repo_eval_v6",
        "split": {"development": list(TASKS), "train": [], "held_out": []},
        "task_membership_frozen": True,
        "prompt_policy": "ordinary issue descriptions; no reference patches or hidden test values",
        "container_image": IMAGE,
        "limits": {"cpus": 2, "memory_mb": 1900, "pids": 32, "timeout_s": 120},
        "seeds": {"model_probe": 0, "watermark": 0},
        "tasks": tasks,
        "limitations": [
            "This cohort is Development-only and is not an independent held-out benchmark.",
            "All tasks are from Transformers; task-family diversity is within one upstream repository.",
            "PR dates do not establish absence from any model training corpus.",
            "The fixed image provides CPU execution; no Transformers task GPU performance claim is made.",
        ],
    }
    protocol_bytes = (json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode()
    PROTOCOL.write_bytes(protocol_bytes)
    implementation_paths = (
        "scripts/grade_transformers_fresh_dev_v6.py",
        "scripts/transformers_fresh_dev_worker_v2.py",
        "scripts/grade_transformers_phase_a_v1.py",
        "scripts/prepare_transformers_dev_cohort_v2.py",
    )
    lock = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_bytes),
        "implementation_sha256": {
            name: sha256((ROOT / name).read_bytes()) for name in implementation_paths
        },
        "baseline_manifests": manifests,
    }
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"protocol_sha256": lock["protocol_sha256"], "task_count": len(tasks)}


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True))
