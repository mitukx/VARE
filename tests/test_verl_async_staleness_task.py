"""Integrity checks for the pinned public fully-async staleness task."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import calibrate_task


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "benchmarks/historical/verl_async_staleness_v1"
RESULT_ROOT = ROOT / "results/verl-fully-async-staleness-accounting-v1"
_EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "verl_async_staleness_grader", TASK_ROOT / "evaluator/grade.py"
)
assert _EVALUATOR_SPEC is not None and _EVALUATOR_SPEC.loader is not None
_EVALUATOR = importlib.util.module_from_spec(_EVALUATOR_SPEC)
_EVALUATOR_SPEC.loader.exec_module(_EVALUATOR)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evaluator_supports_repository_integrity_calibration_cli(tmp_path):
    workspace = tmp_path / "empty-candidate"
    workspace.mkdir()
    evaluator = TASK_ROOT / "evaluator/grade.py"
    completed = subprocess.run(
        [sys.executable, str(evaluator), "--workspace", str(workspace), "--task-root", str(TASK_ROOT)],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert result["grader_error"] == "FileNotFoundError"
    assert "candidate checkout is missing source files" in result["message"]


def test_dispatch_order_accepts_assignment_increment_and_rejects_pre_wait_reservation():
    good = ast.parse(
        """
async def _processor_worker(self):
    while True:
        while len(self.active_tasks) >= self.max_concurrent_samples:
            await asyncio.sleep(0)
        async with self.lock:
            task = safe_create_task(self.process())
            self.staleness_samples = self.staleness_samples + 1
            self._total_dispatched_samples = self._total_dispatched_samples + 1
"""
    ).body[0]
    bad = ast.parse(
        """
async def _processor_worker(self):
    while True:
        self.staleness_samples += 1
        while len(self.active_tasks) >= self.max_concurrent_samples:
            await asyncio.sleep(0)
        async with self.lock:
            task = safe_create_task(self.process())
            self.staleness_samples += 0
            self._total_dispatched_samples += 1
"""
    ).body[0]

    assert _EVALUATOR.evaluate_dispatch_order(good)["passed"] is True
    assert _EVALUATOR.evaluate_dispatch_order(bad)["passed"] is False


def test_reset_evaluator_binds_transitively_referenced_class_helpers():
    candidate = ast.parse(
        """
class FullyAsyncRollouter:
    async def reset_staleness(self):
        async with self.lock:
            self.staleness_samples = await self._remaining_samples()
            return {}

    async def _remaining_samples(self):
        stats = await self.message_queue_client.get_statistics()
        return self._outstanding_from_snapshot(stats)

    def _outstanding_from_snapshot(self, stats):
        removed = stats["total_produced"] - stats["queue_size"]
        return max(0, self._total_dispatched_samples - removed)

    async def _processor_worker(self):
        return None
"""
    )
    reset, _processor, class_methods = _EVALUATOR._find_methods(candidate)

    results = _EVALUATOR.evaluate_reset(reset, class_methods)

    assert [result["passed"] for result in results] == [True] * 4
    assert all(result["returned_metrics"] for result in results)


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


@pytest.mark.parametrize(
    ("version", "protocol_id"),
    [("cpu-calibration-v6", "verl-fully-async-staleness-accounting-v1"),
     ("cpu-calibration-v7", "verl-fully-async-staleness-accounting-v2")],
)
def test_source_calibration_is_hash_complete(version, protocol_id):
    evidence = RESULT_ROOT / version
    summary = json.loads((evidence / "summary.json").read_text(encoding="utf-8"))
    baseline = json.loads((evidence / "baseline.grade.json").read_text(encoding="utf-8"))
    fixed = json.loads((evidence / "calibration.grade.json").read_text(encoding="utf-8"))
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))

    assert summary["status"] == "calibrated"
    assert summary["protocol_id"] == protocol_id
    assert summary["observed"] == {"baseline_passed": False, "calibration_passed": True}
    assert baseline["passed"] is False
    assert fixed["passed"] is True
    assert baseline["metrics"]["reset_cases"] == 4
    assert fixed["metrics"]["reset_cases"] == 4
    interleaved_baseline = baseline["checks"]["reset_staleness"][-1]
    interleaved_fixed = fixed["checks"]["reset_staleness"][-1]
    assert interleaved_baseline["case_id"] == "task-publishes-during-queue-snapshot"
    assert interleaved_baseline["passed"] is False
    assert interleaved_fixed["passed"] is True
    assert baseline["workspace_revision"] == "8ced5d2f4b148c2991217554e0c6f03028fa1470"
    assert fixed["workspace_revision"] == "64812373be25c00b0da1833308d573d45570a364"
    assert all(
        _sha256(evidence / relative_path) == expected
        for relative_path, expected in manifest["files"].items()
    )
