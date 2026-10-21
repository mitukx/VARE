"""Create agent-facing task descriptors for the frozen v6 Development cohort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACK = ROOT / "benchmarks/mlsys_repo_eval_v6"
STUDY = ROOT / "experiments/coding_agent_phase_a_v6"
TASKS = (
    "transformers-49315-watermark-repeat-v1",
    "transformers-49515-rope-setter-v1",
    "transformers-49731-generation-device-type-v1",
    "transformers-49767-janus-background-v1",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare() -> int:
    for task_id in TASKS:
        source_root = SOURCE_PACK / task_id
        frozen_task = json.loads((source_root / "task.json").read_text(encoding="utf-8"))
        manifest = json.loads((source_root / "baseline-source-manifest-v1.json").read_text(encoding="utf-8"))
        target = frozen_task["source"]["editable_files"][0]
        task = {
            "schema": "vare-agent-task-v1",
            "task_id": task_id,
            "metadata": {
                "pack": "mlsys_repo_eval_v6",
                "split": "development",
                "task_type": "correctness",
                "family": frozen_task["metadata"]["family"],
                "editable_files": [target],
            },
            "prompt": frozen_task["prompt"],
            "source": {
                "repository": "huggingface/transformers",
                "base_revision": frozen_task["source"]["base_revision"],
                "calibration_revision": frozen_task["source"]["fixed_revision"],
                "snapshot_sha256": manifest["workspace_snapshot_sha256"],
                "file_sha256": {target: manifest["files"][target]["sha256"]},
                "baseline_target_sha256": manifest["files"][target]["sha256"],
                "fixed_revision_sha256": frozen_task["source"]["fixed_revision"],
            },
            "limits": frozen_task["limits"],
        }
        output_dir = STUDY / "tasks" / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "task.json"
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite frozen agent task: {output_path}")
        output_path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"prepared {len(TASKS)} agent task descriptors under {STUDY / 'tasks'}")
    return len(TASKS)


if __name__ == "__main__":
    prepare()
