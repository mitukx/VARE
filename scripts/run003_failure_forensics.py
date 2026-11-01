#!/usr/bin/env python3
"""Forward-only, sanitized diagnostics for prospective pytest validation.

Run-002's historical ``reference_tests_missing`` records are intentionally
left untouched.  This module classifies only observations collected under a
new diagnostic protocol; missing or ambiguous evidence fails closed as
``unknown``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import re
import xml.etree.ElementTree as ET

SCHEMA = "vare.run003.failure_forensics.observation.v1"
MAX_JUNIT_BYTES = 4 * 1024 * 1024


class FailureKind(str, Enum):
    ENVIRONMENT_CONSTRUCTION_FAILURE = "environment_construction_failure"
    COLLECTION_IMPORT_FAILURE = "collection_import_failure"
    NO_TESTS_COLLECTED = "no_tests_collected"
    REQUESTED_NODE_ABSENT = "requested_node_absent"
    EXECUTED_TEST_FAILURE = "executed_test_failure"
    PASSED = "passed"
    TIMEOUT = "timeout"
    SANDBOX_VIOLATION = "sandbox_violation"
    INTEGRITY_ERROR = "integrity_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class JUnitSummary:
    """Count-only JUnit view. Test names and XML text are never returned."""

    sha256: str
    collected_count: int
    passed_count: int
    failed_count: int
    skipped_count: int
    collection_error_count: int


@dataclass(frozen=True)
class Observation:
    """Trusted supervisor evidence for one validation attempt.

    ``requested_absent_count`` must come from a trusted collector that
    compared exact requested node IDs with collected node IDs. It must not be
    inferred from pytest's exit code alone.
    """

    environment_constructed: bool | None
    sandbox_violation: bool
    source_integrity_ok: bool | None
    timed_out: bool
    pytest_exit: int | None
    requested_count: int
    collected_count: int
    passed_count: int
    failed_count: int
    skipped_count: int
    collection_error_count: int
    requested_absent_count: int = 0
    junit_integrity_ok: bool = True

    def validate(self) -> None:
        for name in ("environment_constructed", "source_integrity_ok"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be true, false, or null")
        for name in ("sandbox_violation", "timed_out", "junit_integrity_ok"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("requested_count", "collected_count", "passed_count", "failed_count",
                     "skipped_count", "collection_error_count", "requested_absent_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.pytest_exit is not None and type(self.pytest_exit) is not int:
            raise ValueError("pytest_exit must be an integer or null")
        if self.requested_absent_count > self.requested_count:
            raise ValueError("absent requested count exceeds requested count")
        if self.passed_count + self.failed_count + self.skipped_count > self.collected_count:
            raise ValueError("observed test outcomes exceed collected count")


def summarize_junit(path: Path) -> JUnitSummary:
    """Read bounded JUnit XML and return only counts plus its content hash."""

    raw = path.read_bytes()
    if not raw or len(raw) > MAX_JUNIT_BYTES:
        raise ValueError("JUnit report is empty or exceeds the 4 MiB diagnostic limit")
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("JUnit declarations are not allowed")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError("JUnit report is malformed") from exc
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError("JUnit root element is unsupported")

    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    collected = passed = failed = skipped = collection_errors = 0
    for suite in suites:
        for case in suite.findall("testcase"):
            errors = case.findall("error")
            if errors and any(
                (err.attrib.get("message", "").strip().lower().startswith("collection failure")
                 or err.attrib.get("message", "").strip().lower().startswith("error collecting"))
                for err in errors
            ):
                collection_errors += 1
                continue
            collected += 1
            if case.find("failure") is not None or errors:
                failed += 1
            elif case.find("skipped") is not None:
                skipped += 1
            else:
                passed += 1
    return JUnitSummary(
        sha256=hashlib.sha256(raw).hexdigest(),
        collected_count=collected,
        passed_count=passed,
        failed_count=failed,
        skipped_count=skipped,
        collection_error_count=collection_errors,
    )


def classify(observation: Observation) -> FailureKind:
    """Classify with explicit precedence; ambiguous outcomes remain unknown."""

    observation.validate()
    if observation.environment_constructed is False:
        return FailureKind.ENVIRONMENT_CONSTRUCTION_FAILURE
    if observation.sandbox_violation:
        return FailureKind.SANDBOX_VIOLATION
    if observation.source_integrity_ok is False or not observation.junit_integrity_ok:
        return FailureKind.INTEGRITY_ERROR
    if observation.timed_out:
        return FailureKind.TIMEOUT
    if observation.pytest_exit is None:
        return FailureKind.UNKNOWN
    if observation.collection_error_count:
        return FailureKind.COLLECTION_IMPORT_FAILURE
    if (observation.pytest_exit == 5 and observation.requested_count == 0
            and observation.collected_count == 0):
        return FailureKind.NO_TESTS_COLLECTED
    if observation.requested_absent_count:
        return FailureKind.REQUESTED_NODE_ABSENT
    if observation.failed_count:
        return FailureKind.EXECUTED_TEST_FAILURE
    if (observation.pytest_exit == 0 and observation.requested_count > 0
            and observation.requested_count == observation.passed_count
            and observation.skipped_count == 0):
        return FailureKind.PASSED
    return FailureKind.UNKNOWN


def sanitized_record(
    observation: Observation, *, legacy_reason: str | None, junit_sha256: str | None = None,
) -> dict[str, object]:
    """Serialize count-only evidence; legacy reason is preserved verbatim."""

    if legacy_reason is not None and not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", legacy_reason):
        raise ValueError("legacy_reason must be a bounded reason-code token")
    kind = classify(observation)
    return {
        "schema": SCHEMA,
        "legacy_reason": legacy_reason,
        "classification": kind.value,
        "requested_count": observation.requested_count,
        "collected_count": observation.collected_count,
        "passed_count": observation.passed_count,
        "failed_count": observation.failed_count,
        "skipped_count": observation.skipped_count,
        "collection_error_count": observation.collection_error_count,
        "requested_absent_count": observation.requested_absent_count,
        "pytest_exit": observation.pytest_exit,
        "junit_sha256": junit_sha256,
    }


def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate sanitized rows without carrying task, repository, or test IDs."""

    counts = {kind.value: 0 for kind in FailureKind}
    for record in records:
        if record.get("schema") != SCHEMA:
            raise ValueError("diagnostic row has an unsupported schema")
        classification = record.get("classification")
        if classification not in counts:
            raise ValueError("diagnostic row has an unsupported classification")
        counts[str(classification)] += 1
    return {"schema": "vare.run003.failure_forensics.aggregate.v1", "attempts": len(records),
            "classifications": counts}
