"""Run candidate-workspace tamper controls against the v4 Development graders."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grade_transformers_fresh_dev_v1 import grade, load_frozen_protocol  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attacks(task_id: str, baseline: Path, image: str) -> list[dict[str, Any]]:
    protocol, lock, protocol_sha = load_frozen_protocol()
    results: list[dict[str, Any]] = []
    task_spec = protocol["tasks"][task_id]
    target = task_spec["target_path"]
    mutations: dict[str, Callable[[Path], None]] = {
        "startup_hook": lambda root: (root / "sitecustomize.py").write_text(
            "raise RuntimeError('candidate startup hook executed')\n", encoding="utf-8",
        ),
        "altered_imported_module": lambda root: (root / "src/transformers/cache_utils.py").write_text(
            "raise RuntimeError('candidate changed imported dependency')\n", encoding="utf-8",
        ),
        "symlink_target": lambda root: (root / target).unlink(),
        "modified_public_test": lambda root: (root / "test_public.py").write_text(
            "print('forged public pass')\n", encoding="utf-8",
        ),
        "forged_grading_evidence": lambda root: (root / "grade.json").write_text(
            '{"passed": true, "task_id": "' + task_id + '"}\n', encoding="utf-8",
        ),
    }
    for name, mutate in mutations.items():
        with tempfile.TemporaryDirectory(prefix="vare-fresh-redteam-") as temporary:
            workspace = Path(temporary) / "workspace"
            shutil.copytree(baseline, workspace)
            mutate(workspace)
            if name == "symlink_target":
                (workspace / target).symlink_to(workspace / "test_public.py")
            try:
                grade(task_id, workspace, image)
            except ValueError as exc:
                rejected = True
                reason = str(exc)
            else:
                rejected = False
                reason = "grader returned a result for a tampered workspace"
            results.append({
                "attack": name,
                "rejected_before_candidate_execution": rejected,
                "reason": reason,
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha,
                "baseline_manifest_sha256": lock["baseline_manifests"][task_id]["sha256"],
            })
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=[
        "transformers-49481-bloom-static-cache-v1",
        "transformers-49361-phi3-longrope-v1",
    ], required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("refusing to overwrite red-team evidence")
    protocol, _, protocol_sha = load_frozen_protocol()
    results = attacks(args.task_id, args.workspace, args.image)
    payload = {
        "task_id": args.task_id,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "baseline_target_sha256": sha256(args.workspace / protocol["tasks"][args.task_id]["target_path"]),
        "attacks": results,
        "all_rejected": all(row["rejected_before_candidate_execution"] for row in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"task_id": args.task_id, "all_rejected": payload["all_rejected"], "attacks": len(results)}))
    return 0 if payload["all_rejected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
