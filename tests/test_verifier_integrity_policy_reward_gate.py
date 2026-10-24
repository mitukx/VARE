from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _report(failures: list[str | None]) -> dict:
    cases = [
        {
            "case_id": f"case-{index}",
            "passed": failure is None,
            "failure": failure,
            "wall_s": 0.01,
        }
        for index, failure in enumerate(failures)
    ]
    return {
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
        "candidate_source_sha256": "a" * 64,
        "grader_source_sha256": "b" * 64,
        "grader_adapter_sha256": "c" * 64,
    }


def _load_runner():
    name = "policy_gradient_runner_under_test"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/run_verifier_integrity_policy_gradient_v1.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The runner's module-level ML imports are not involved in this pure
    # grader-contract regression. Avoid requiring Torch in the test matrix.
    fake_torch = ModuleType("torch")
    fake_torch.Tensor = object
    fake_psutil = ModuleType("psutil")
    with patch.dict(sys.modules, {"torch": fake_torch, "psutil": fake_psutil}):
        spec.loader.exec_module(module)
    return module


def _grade(monkeypatch, report: dict) -> dict:
    runner = _load_runner()
    monkeypatch.setattr(
        runner.task, "verify_protocol_v4", lambda: {"runtime": {"image": "old"}}
    )
    monkeypatch.setattr(runner.task, "grade_workspace", lambda *args, **kwargs: report)
    protocol = {
        "task": {"trusted_case_count": len(report["cases"])},
        "budgets": {
            "candidate_timeout_seconds": 30,
            "candidate_memory_limit_mib": 1536,
        },
    }
    return runner.grade_action("candidate", Path("unused"), "pinned-image", 0, protocol)


def test_wrong_output_remains_a_valid_zero_reward(monkeypatch):
    result = _grade(monkeypatch, _report([None, "wrong_output"]))
    assert result["task_passed"] is False
    assert result["case_count"] == 2


@pytest.mark.parametrize("failure", ["missing_result", "malformed_result"])
def test_transport_and_protocol_failures_abort_before_reward(monkeypatch, failure):
    with pytest.raises(RuntimeError, match="non-scorable failure"):
        _grade(monkeypatch, _report([None, failure]))


def test_report_validator_rejects_inconsistent_case_and_task_outcomes():
    runner = _load_runner()
    report = _report([None, "wrong_output"])
    report["passed"] = True
    with pytest.raises(runner.ExperimentStop, match="task outcome conflicts"):
        runner.validate_candidate_grade_report(report, 2)


def test_report_validator_rejects_wrong_case_count():
    runner = _load_runner()
    with pytest.raises(runner.ExperimentStop, match="expected 2"):
        runner.validate_candidate_grade_report(_report([None]), 2)


def test_historical_v1_runner_fails_preflight_after_prospective_guard():
    runner = _load_runner()
    with pytest.raises(runner.ExperimentStop, match="locked implementation source hash mismatch"):
        runner.load_frozen_inputs()
