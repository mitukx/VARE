"""Unit and integration checks for forward-only Run-003 diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run003_failure_forensics as ff  # noqa: E402


def observation(**overrides: object) -> ff.Observation:
    values: dict[str, object] = {
        "environment_constructed": True,
        "sandbox_violation": False,
        "source_integrity_ok": True,
        "timed_out": False,
        "pytest_exit": 0,
        "requested_count": 1,
        "collected_count": 1,
        "passed_count": 1,
        "failed_count": 0,
        "skipped_count": 0,
        "collection_error_count": 0,
        "requested_absent_count": 0,
    }
    values.update(overrides)
    return ff.Observation(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(("values", "expected"), [
    ({"environment_constructed": False}, ff.FailureKind.ENVIRONMENT_CONSTRUCTION_FAILURE),
    ({"sandbox_violation": True}, ff.FailureKind.SANDBOX_VIOLATION),
    ({"source_integrity_ok": False}, ff.FailureKind.INTEGRITY_ERROR),
    ({"junit_integrity_ok": False}, ff.FailureKind.INTEGRITY_ERROR),
    ({"timed_out": True}, ff.FailureKind.TIMEOUT),
    ({"pytest_exit": 2, "collection_error_count": 1, "passed_count": 0,
      "collected_count": 0}, ff.FailureKind.COLLECTION_IMPORT_FAILURE),
    ({"pytest_exit": 5, "requested_count": 0, "collected_count": 0, "passed_count": 0},
     ff.FailureKind.NO_TESTS_COLLECTED),
    ({"pytest_exit": 4, "passed_count": 0, "collected_count": 0,
      "requested_absent_count": 1}, ff.FailureKind.REQUESTED_NODE_ABSENT),
    ({"pytest_exit": 1, "passed_count": 0, "failed_count": 1},
     ff.FailureKind.EXECUTED_TEST_FAILURE),
    ({}, ff.FailureKind.PASSED),
    # Exit status 4 by itself cannot distinguish a missing node from usage/config errors.
    ({"pytest_exit": 4, "passed_count": 0, "collected_count": 0}, ff.FailureKind.UNKNOWN),
    ({"pytest_exit": 2, "passed_count": 0, "collected_count": 0}, ff.FailureKind.UNKNOWN),
])
def test_typed_classification(values: dict[str, object], expected: ff.FailureKind) -> None:
    assert ff.classify(observation(**values)) is expected


def test_junit_summary_counts_without_returning_names_or_error_text(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite tests="2" failures="1" errors="0" skipped="0">'
        '<testcase classname="private.file" name="private_test" />'
        '<testcase classname="private.file" name="private_bad">'
        '<failure message="private assertion">secret details</failure></testcase>'
        '</testsuite></testsuites>', encoding="utf-8")
    summary = ff.summarize_junit(report)
    assert (summary.collected_count, summary.passed_count, summary.failed_count) == (2, 1, 1)
    assert "private" not in json.dumps(summary.__dict__)
    assert len(summary.sha256) == 64


def test_junit_collection_error_is_not_counted_as_an_executed_test(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite><testcase classname="" name="module">'
        '<error message="collection failure">private traceback</error>'
        '</testcase></testsuite></testsuites>', encoding="utf-8")
    summary = ff.summarize_junit(report)
    assert summary.collection_error_count == 1
    assert summary.collected_count == summary.failed_count == 0


def test_invalid_junit_fails_as_integrity_error(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    report.write_text("<testsuite>", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        ff.summarize_junit(report)


def test_legacy_reason_is_preserved_and_aggregate_omits_identifiers() -> None:
    record = ff.sanitized_record(
        observation(pytest_exit=4, passed_count=0, collected_count=0),
        legacy_reason="reference_tests_missing",
    )
    assert record["legacy_reason"] == "reference_tests_missing"
    assert record["classification"] == "unknown"
    aggregate = ff.aggregate([record])
    text = json.dumps(aggregate)
    assert "reference_tests_missing" not in text and "instance_id" not in text
    assert aggregate["attempts"] == 1


@pytest.mark.parametrize("values", [
    {"requested_count": -1},
    {"passed_count": 2, "collected_count": 1},
    {"requested_count": 1, "requested_absent_count": 2},
])
def test_invalid_observation_counts_are_rejected(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ff.classify(observation(**values))
