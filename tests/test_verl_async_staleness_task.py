"""Integrity checks for the pinned public fully-async staleness task."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts import calibrate_task


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "benchmarks/historical/verl_async_staleness_v1"
RESULT_ROOT = ROOT / "results/verl-fully-async-staleness-accounting-v1/cpu-calibration-v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("relative_path", ["TASK.md", "task.json", "evaluator/grade.py"])
def test_calibration_lock_rejects_task_or_evaluator_tampering(tmp_path, relative_path):
    copied = tmp_path / "task"
    shutil.copytree(TASK_ROOT, copied)
    task = json.loads((copied / "task.json").read_text(encoding="utf-8"))
    lock = json.loads((copied / "protocol.lock.json").read_text(encoding="utf-8"))
    target = copied / relative_path
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(SystemExit, match="protocol lock mismatch"):
        calibrate_task.verify_lock(task, lock, copied, copied / "evaluator/grade.py")


def test_retained_source_calibration_is_hash_complete():
    summary = json.loads((RESULT_ROOT / "summary.json").read_text(encoding="utf-8"))
    baseline = json.loads((RESULT_ROOT / "baseline.grade.json").read_text(encoding="utf-8"))
    fixed = json.loads((RESULT_ROOT / "calibration.grade.json").read_text(encoding="utf-8"))
    manifest = json.loads((RESULT_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert summary["status"] == "calibrated"
    assert summary["observed"] == {"baseline_passed": False, "calibration_passed": True}
    assert baseline["passed"] is False
    assert fixed["passed"] is True
    assert baseline["workspace_revision"] == "8ced5d2f4b148c2991217554e0c6f03028fa1470"
    assert fixed["workspace_revision"] == "64812373be25c00b0da1833308d573d45570a364"
    assert all(
        _sha256(RESULT_ROOT / relative_path) == expected
        for relative_path, expected in manifest["files"].items()
    )
