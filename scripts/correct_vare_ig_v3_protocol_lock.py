#!/usr/bin/env python3
"""Issue immutable corrected V3b run locks after pre-inference schema validation."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import qwen3_vare_ig_experiment as experiment  # noqa: E402
from scripts import qwen3_function_repair_grpo as runner  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {path}")
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    invalid_master_path = ROOT / "protocols/vare_ig_qwen3_mlx_dev_v3.json"
    invalid_run_dir = ROOT / "protocols/vare_ig_qwen3_mlx_dev_v3"
    master_path = ROOT / "protocols/vare_ig_qwen3_mlx_dev_v3b.json"
    run_dir = ROOT / "protocols/vare_ig_qwen3_mlx_dev_v3b"
    invalid_master = json.loads(invalid_master_path.read_text(encoding="utf-8"))
    if invalid_master.get("schema") != "vare_ig_qwen3_mlx_protocol_v3":
        raise RuntimeError("correction source is not the retained V3 master")
    if invalid_master.get("source_hashes") != runner.source_hashes() \
            or invalid_master.get("vare_ig_source_hashes") != experiment.source_hashes() \
            or invalid_master.get("task_contract_hashes") != experiment.task_hashes():
        raise RuntimeError("V3 protocol code or task contract provenance differs from the current repository")
    invalid_runs = sorted(invalid_run_dir.glob("*.json"))
    if len(invalid_runs) != 4:
        raise RuntimeError("expected all four retained V3 run locks")
    for path in invalid_runs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("execution") != invalid_master.get("execution"):
            raise RuntimeError("V3 run execution extension differs from its master")
        if payload.get("sampling", {}).get("seed") not in (20261116, 20261117):
            raise RuntimeError("V3 run seed changed")

    start_sha = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tool_path = Path(__file__).resolve()
    tool_sha = sha256_file(tool_path)
    amendment = deepcopy(invalid_master["protocol_amendment"])
    amendment.update({
        "protocol_correction": "V3b static-lock correction before model inference",
        "protocol_correction_reason": (
            "V3's run locks omitted the explicit task_contracts.train/development maps required by "
            "the existing runner payload validator; all four were rejected before model loading"
        ),
        "superseded_v3_master_path": invalid_master_path.relative_to(ROOT).as_posix(),
        "superseded_v3_master_sha256": sha256_file(invalid_master_path),
        "superseded_v3_run_locks": {
            path.name: sha256_file(path) for path in invalid_runs
        },
        "protocol_amendment_tool_path": tool_path.relative_to(ROOT).as_posix(),
        "protocol_amendment_tool_sha256": tool_sha,
        "implementation_start_sha": start_sha,
        "frozen_at_utc": timestamp,
    })

    master = deepcopy(invalid_master)
    master["schema"] = "vare_ig_qwen3_mlx_protocol_v3b"
    master["protocol_id"] = "vare-ig-qwen3-4b-mlx-training-amendment-v3b"
    master["planned_run_id"] = "vare-ig-qwen3-4b-mlx-training-v3b"
    master["frozen_at_utc"] = timestamp
    master["implementation_start_sha"] = start_sha
    master["protocol_stage"] = "V3b corrected run-lock schema frozen before model inference"
    master["protocol_amendment"] = amendment
    write_once(master_path, master)
    master_sha = sha256_file(master_path)

    train_hashes = experiment.TRAIN_TASK_HASHES
    dev_hashes = experiment.DEVELOPMENT_TASK_HASHES
    for old_path in invalid_runs:
        payload = deepcopy(json.loads(old_path.read_text(encoding="utf-8")))
        payload["protocol_id"] = payload["protocol_id"].replace("-v3", "-v3b")
        payload["planned_run_id"] = payload["planned_run_id"].replace("-v3", "-v3b")
        payload["frozen_at_utc"] = timestamp
        payload["implementation_start_sha"] = start_sha
        payload["task_contracts"] = {
            "train": {"task_hashes": train_hashes},
            "development": {"task_hashes": dev_hashes},
        }
        payload["base_master_protocol_sha256"] = master_sha
        payload["protocol_amendment"] = amendment
        payload["protocol_correction"] = {
            "supersedes_path": old_path.relative_to(ROOT).as_posix(),
            "supersedes_sha256": sha256_file(old_path),
            "reason": "missing task_contracts map; rejected pre-inference",
        }
        path = run_dir / old_path.name
        write_once(path, payload)
        print(f"frozen {path.relative_to(ROOT)} sha256={sha256_file(path)}")
    print(f"frozen {master_path.relative_to(ROOT)} sha256={master_sha}")
    print(f"correction tool sha256={tool_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
