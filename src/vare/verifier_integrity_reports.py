"""Fail-closed validation for isolated verifier reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class VerifierReportIntegrityError(ValueError):
    """The grader response is incomplete or cannot support a reward."""


def validate_candidate_grade_report(
    report: Any, expected_case_count: int
) -> tuple[list[dict[str, Any]], bool]:
    """Validate case outcomes before a policy consumes the task reward.

    A normal candidate mismatch is represented as ``wrong_output`` and is a
    valid zero reward. Transport and protocol failures such as
    ``missing_result`` or ``malformed_result`` must stop the experiment, not
    become training labels or evaluation outcomes.
    """
    if type(expected_case_count) is not int or expected_case_count <= 0:
        raise VerifierReportIntegrityError("expected case count must be positive")
    if not isinstance(report, Mapping):
        raise VerifierReportIntegrityError("grader report is not a mapping")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_case_count:
        actual = len(cases) if isinstance(cases, list) else "non-list"
        raise VerifierReportIntegrityError(
            f"grader returned {actual} cases; expected {expected_case_count}"
        )

    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise VerifierReportIntegrityError(f"case {index} is not a mapping")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise VerifierReportIntegrityError(
                f"case {index} has a missing or duplicate case id"
            )
        seen_ids.add(case_id)
        passed = case.get("passed")
        if type(passed) is not bool:
            raise VerifierReportIntegrityError(f"case {case_id} has no boolean outcome")
        if "failure" not in case:
            raise VerifierReportIntegrityError(f"case {case_id} has no failure category")
        failure = case["failure"]
        if failure not in (None, "wrong_output"):
            raise VerifierReportIntegrityError(
                f"case {case_id} has non-scorable grader failure: {failure!r}"
            )
        if passed != (failure is None):
            raise VerifierReportIntegrityError(
                f"case {case_id} outcome conflicts with failure category"
            )

    report_passed = report.get("passed")
    if type(report_passed) is not bool or report_passed != all(
        case["passed"] for case in cases
    ):
        raise VerifierReportIntegrityError("task outcome conflicts with case outcomes")
    return [dict(case) for case in cases], report_passed
