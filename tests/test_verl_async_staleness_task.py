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


def test_dispatch_order_rejects_unreachable_accounting_updates():
    for transfer in ("return None", "raise RuntimeError('stop')", "break"):
        worker = ast.parse(
            f"""
async def _processor_worker(self):
    while True:
        while len(self.active_tasks) >= self.max_concurrent_samples:
            await asyncio.sleep(0)
        async with self.lock:
            task = safe_create_task(self.process())
            {transfer}
            self.staleness_samples += 1
            self._total_dispatched_samples += 1
"""
        ).body[0]

        assert _EVALUATOR.evaluate_dispatch_order(worker)["passed"] is False


def test_reset_evaluator_binds_transitive_class_and_static_helpers():
    candidate = ast.parse(
        """
class FullyAsyncRollouter:
    async def reset_staleness(self):
        async with self.lock:
            self.staleness_samples = await self._remaining_samples()
            return {}

    async def _remaining_samples(self):
        stats = await self.message_queue_client.get_statistics()
        return self._outstanding_from_snapshot(stats, self._total_dispatched_samples)

    @classmethod
    def _outstanding_from_snapshot(cls, stats, dispatched):
        removed = stats["total_produced"] - stats["queue_size"]
        return cls._nonnegative_outstanding_count(dispatched, removed)

    @staticmethod
    def _nonnegative_outstanding_count(dispatched, removed):
        return max(0, dispatched - removed)

    async def _processor_worker(self):
        return None
"""
    )
    reset, _processor, class_methods = _EVALUATOR._find_methods(candidate)

    results = _EVALUATOR.evaluate_reset(reset, class_methods)

    assert [result["passed"] for result in results] == [True] * 6
    assert all(result["returned_metrics"] for result in results)


def test_reset_evaluator_rejects_separate_queue_reads_during_publication():
    candidate = ast.parse(
        """
class FullyAsyncRollouter:
    async def reset_staleness(self):
        async with self.lock:
            stats = await self.message_queue_client.get_statistics()
            queue_size = await self.message_queue_client.get_queue_size()
            removed = stats["total_produced"] - queue_size
            self.staleness_samples = max(0, self._total_dispatched_samples - removed)
            return {}

    async def _processor_worker(self):
        return None
"""
    )
    reset, _processor, class_methods = _EVALUATOR._find_methods(candidate)

    results = _EVALUATOR.evaluate_reset(reset, class_methods)
    by_case = {result["case_id"]: result for result in results}

    assert by_case["publication-between-separate-queue-reads"]["observed"] == 8
    assert by_case["publication-between-separate-queue-reads"]["passed"] is False


def test_reset_evaluator_rejects_negative_count_from_control_record():
    candidate = ast.parse(
        """
class FullyAsyncRollouter:
    async def reset_staleness(self):
        async with self.lock:
            stats = await self.message_queue_client.get_statistics()
            removed = stats["total_produced"] - stats["queue_size"]
            self.staleness_samples = self._total_dispatched_samples - removed
            return {}

    async def _processor_worker(self):
        return None
"""
    )
    reset, _processor, class_methods = _EVALUATOR._find_methods(candidate)

    results = _EVALUATOR.evaluate_reset(reset, class_methods)
    control_record = results[-1]

    assert control_record["case_id"] == "consumed-control-record-without-dispatch"
    assert control_record["observed"] == -1
    assert control_record["passed"] is False


def test_reset_evaluator_rejects_lifetime_counter_reset():
    candidate = ast.parse(
        """
class FullyAsyncRollouter:
    async def reset_staleness(self):
        async with self.lock:
            stats = await self.message_queue_client.get_statistics()
            removed = stats["total_produced"] - stats["queue_size"]
            self.staleness_samples = max(0, self._total_dispatched_samples - removed)
            self._total_dispatched_samples = 0
            return {}

    async def _processor_worker(self):
        return None
"""
    )
    reset, _processor, class_methods = _EVALUATOR._find_methods(candidate)

    results = _EVALUATOR.evaluate_reset(reset, class_methods)

    assert results[0]["observed"] == results[0]["expected"]
    assert results[0]["dispatch_counter_before"] == 45
    assert results[0]["dispatch_counter_after"] == 0
    assert results[0]["dispatch_counter_preserved"] is False
    assert results[0]["passed"] is False


def test_dispatch_counter_must_be_initialized_once_in_constructor():
    valid = ast.parse(
        "def __init__(self):\n    self._total_dispatched_samples = 0\n"
    ).body[0]
    missing = ast.parse("def __init__(self):\n    self.other = 0\n").body[0]
    conditional = ast.parse(
        "def __init__(self):\n    if ready:\n        self._total_dispatched_samples = 0\n"
    ).body[0]

    assert _EVALUATOR.evaluate_dispatch_counter_initialization(valid)["passed"] is True
    assert _EVALUATOR.evaluate_dispatch_counter_initialization(missing)["passed"] is False
    assert _EVALUATOR.evaluate_dispatch_counter_initialization(conditional)["passed"] is False


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
    ("version", "protocol_id", "reset_case_count"),
    [
        ("cpu-calibration-v6", "verl-fully-async-staleness-accounting-v1", 4),
        ("cpu-calibration-v7", "verl-fully-async-staleness-accounting-v2", 4),
        ("cpu-calibration-v8", "verl-fully-async-staleness-accounting-v3", 6),
        ("cpu-calibration-v9", "verl-fully-async-staleness-accounting-v4", 6),
    ],
)
def test_source_calibration_is_hash_complete(version, protocol_id, reset_case_count):
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
    assert baseline["metrics"]["reset_cases"] == reset_case_count
    assert fixed["metrics"]["reset_cases"] == reset_case_count
    interleaved_baseline = baseline["checks"]["reset_staleness"][3]
    interleaved_fixed = fixed["checks"]["reset_staleness"][3]
    assert interleaved_baseline["case_id"] == "task-publishes-during-queue-snapshot"
    assert interleaved_baseline["passed"] is False
    assert interleaved_fixed["passed"] is True
    if reset_case_count == 6:
        assert baseline["checks"]["dispatch_counter_initialization"]["passed"] is False
        assert fixed["checks"]["dispatch_counter_initialization"]["passed"] is True
        assert baseline["checks"]["reset_staleness"][4]["passed"] is False
        assert baseline["checks"]["reset_staleness"][5]["observed"] == 0
        assert fixed["checks"]["reset_staleness"][4]["passed"] is True
        assert fixed["checks"]["reset_staleness"][5]["passed"] is True
        if protocol_id.endswith("-v4"):
            assert all(case["dispatch_counter_preserved"] for case in fixed["checks"]["reset_staleness"])
    assert baseline["workspace_revision"] == "8ced5d2f4b148c2991217554e0c6f03028fa1470"
    assert fixed["workspace_revision"] == "64812373be25c00b0da1833308d573d45570a364"
    assert all(
        _sha256(evidence / relative_path) == expected
        for relative_path, expected in manifest["files"].items()
    )
