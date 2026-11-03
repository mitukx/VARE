"""Restricted source contract followed by an independent numerical oracle.

The caller-supplied worker runner must execute the checked source bytes in the
locked Seatbelt profile with the locked worker and resource ceilings. This
module never imports candidate code. It invokes the runner only after the
candidate AST matches the pinned known-fix AST.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable

from .source_contract import SourceContractResult, check_candidate_source


VALUE_RTOL = 1e-5
VALUE_ATOL = 1e-8
GRAD_RTOL = 1e-4
GRAD_ATOL = 1e-9
MAX_REPORT_BYTES = 1_048_576
EXPECTED_PYTHON = "3.9.6"
EXPECTED_TORCH = "2.8.0"
EXPECTED_KNOWN_FIX_AST_SHA256 = (
    "12adfb24b211ed808b88e52e7bb1b037cb1f3bbc79515b2ec5a3e2c7d83c96b8"
)


class WorkerRunnerError(RuntimeError):
    """The trusted runner failed and records whether a child was launched."""

    def __init__(self, message: str, *, worker_started: bool):
        super().__init__(message)
        self.worker_started = worker_started


def generate_cases(seed: int) -> list[dict[str, Any]]:
    """Generate two bounded challenge cases after the protocol is frozen."""
    rng = random.Random(seed)
    first = round(rng.uniform(-2.0, 2.0), 8)
    delta_one = round(rng.uniform(0.5, 2.0), 8)
    other_one = round(rng.uniform(-4.0, 4.0), 8)
    case_one = {
        "name": "excluded_global_max",
        "logits": [first, round(first - delta_one, 8), other_one, 200.0],
        "target": 0,
        "support": [0, 1],
        "chunk_size": 2,
    }
    first_support = round(rng.uniform(-8.0, -4.0), 8)
    delta_two = round(rng.uniform(3.0, 5.0), 8)
    case_two = {
        "name": "support_max_rises_across_chunk",
        "logits": [
            first_support,
            round(rng.uniform(-8.0, 0.0), 8),
            round(rng.uniform(-8.0, 0.0), 8),
            round(first_support + delta_two, 8),
        ],
        "target": 3,
        "support": [0, 3],
        "chunk_size": 2,
    }
    return [case_one, case_two]


def reference(case: dict[str, Any]) -> tuple[float, list[float]]:
    """Compute support log-probability and its analytic logit gradient."""
    values = case["logits"]
    support = case["support"]
    maximum = max(values[index] for index in support)
    scaled = {index: math.exp(values[index] - maximum) for index in support}
    denominator = sum(scaled.values())
    logprob = values[case["target"]] - maximum - math.log(denominator)
    gradient = [0.0] * len(values)
    for index in support:
        gradient[index] = scaled[index] / denominator
    gradient[case["target"]] -= 1.0
    return logprob, gradient


def _result(
    status: str,
    contract: SourceContractResult,
    *,
    worker_started: bool | None,
    reason: str,
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "grader_status": status,
        "reason": reason,
        "source_contract": {
            "status": contract.status,
            "source_sha256": contract.source_sha256,
            "ast_sha256": contract.ast_sha256,
        },
        "candidate_code_imported_by_parent": False,
        "worker_started": worker_started,
        "worker_report_authenticated": False,
        "candidate_process_report_trusted": False,
        "cases": cases or [],
    }


def _parse_report(raw: bytes) -> dict[str, Any] | None:
    if not raw or len(raw) > MAX_REPORT_BYTES or not raw.endswith(b"\n"):
        return None
    try:
        report = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict):
        return None
    if set(report) != {"python_version", "torch_version", "rows"}:
        return None
    if report.get("python_version") != EXPECTED_PYTHON:
        return None
    if report.get("torch_version") != EXPECTED_TORCH:
        return None
    if not isinstance(report.get("rows"), list):
        return None
    return report


def _compare_rows(
    report: dict[str, Any], cases: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    rows = report["rows"]
    observed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case"), str):
            return "INFRASTRUCTURE_FAIL", []
        if row["case"] in observed:
            return "INFRASTRUCTURE_FAIL", []
        observed[row["case"]] = row
    if set(observed) != {case["name"] for case in cases}:
        return "INFRASTRUCTURE_FAIL", []

    results: list[dict[str, Any]] = []
    for case in cases:
        row = observed[case["name"]]
        if row.get("status") == "TEST_ERROR":
            if not isinstance(row.get("error_type"), str):
                return "INFRASTRUCTURE_FAIL", results
            results.append(
                {
                    "case": case["name"],
                    "candidate_error": row["error_type"],
                    "pass": False,
                }
            )
            continue
        if row.get("status") != "EXECUTED":
            return "INFRASTRUCTURE_FAIL", results
        logprobs = row.get("logprobs")
        gradients = row.get("weight_grad")
        if (
            not isinstance(logprobs, list)
            or len(logprobs) != 1
            or not isinstance(gradients, list)
            or len(gradients) != len(case["logits"])
            or type(row.get("finite")) is not bool
        ):
            return "INFRASTRUCTURE_FAIL", results
        try:
            actual_value = float(logprobs[0])
            actual_gradient = [float(value) for value in gradients]
        except (TypeError, ValueError, OverflowError):
            return "INFRASTRUCTURE_FAIL", results

        expected_value, expected_gradient = reference(case)
        value_ok = math.isfinite(actual_value) and abs(
            actual_value - expected_value
        ) <= (VALUE_ATOL + VALUE_RTOL * abs(expected_value))
        gradient_ok = all(
            math.isfinite(actual)
            and abs(actual - expected) <= (GRAD_ATOL + GRAD_RTOL * abs(expected))
            for actual, expected in zip(actual_gradient, expected_gradient)
        )
        results.append(
            {
                "case": case["name"],
                "expected_logprob": expected_value,
                "actual_logprob": actual_value,
                "expected_gradient": expected_gradient,
                "actual_gradient": actual_gradient,
                "value_ok": value_ok,
                "gradient_ok": gradient_ok,
                "pass": value_ok and gradient_ok,
            }
        )
    status = "PASS" if all(row["pass"] for row in results) else "TEST_FAIL"
    return status, results


def grade_candidate(
    workspace: Path,
    relative_source_path: str,
    expected_ast_sha256: str,
    seed: int,
    worker_runner: Callable[[bytes, list[dict[str, Any]]], bytes],
) -> dict[str, Any]:
    """Gate candidate source before worker launch, then compare to parent math.

    The runner is called only with the exact bytes that passed the source gate.
    It is responsible for copying them into a sealed scratch file and enforcing
    the protocol's macOS Seatbelt and resource limits.
    """
    contract, checked_source = check_candidate_source(
        workspace, relative_source_path, expected_ast_sha256
    )
    if not contract.accepted or checked_source is None:
        return _result(
            "TEST_FAIL",
            contract,
            worker_started=False,
            reason="candidate does not satisfy the restricted known-fix source contract",
        )

    cases = generate_cases(seed)
    try:
        raw_report = worker_runner(checked_source, cases)
    except WorkerRunnerError as exc:
        return _result(
            "INFRASTRUCTURE_FAIL",
            contract,
            worker_started=exc.worker_started,
            reason=f"worker runner failed: {type(exc).__name__}",
        )
    except Exception as exc:  # runner failures must not become TEST_FAIL
        return _result(
            "INFRASTRUCTURE_FAIL",
            contract,
            worker_started=None,
            reason=f"worker runner failed: {type(exc).__name__}",
        )
    report = _parse_report(raw_report)
    if report is None:
        return _result(
            "INFRASTRUCTURE_FAIL",
            contract,
            worker_started=True,
            reason="worker report schema, runtime identity, or output bound failed",
        )
    status, row_results = _compare_rows(report, cases)
    return _result(
        status,
        contract,
        worker_started=True,
        reason="independent value and gradient comparison completed",
        cases=row_results,
    )
