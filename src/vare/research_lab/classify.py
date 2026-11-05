"""Failure classification for one experiment attempt.

A wall-clock timeout is a resource outcome. It is candidate incorrectness only
when the trusted contract says so. Self-reported success flags never decide
acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .jsonutil import strict_load
from .paths import PathPolicyError, resolve_inside
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MetricsRead:
    status: str
    values: dict[str, float] | None
    self_reported: bool


@dataclass(frozen=True, slots=True)
class Judgment:
    state: str
    failure_class: str
    accepted: bool
    candidate_incorrect: bool


_SELF_REPORT_KEYS = ("accepted", "passed", "success", "ok")


def _self_reported(payload: Mapping[str, object]) -> bool:
    return any(payload.get(key) is True for key in _SELF_REPORT_KEYS)


def read_metrics(workspace: Path, contract) -> MetricsRead:
    if contract is None:
        return MetricsRead("ok", {}, False)
    try:
        path = resolve_inside(workspace, contract.relative_path, expect="file")
    except PathPolicyError as exc:
        status = "missing_metrics" if exc.failure_class == "missing_file" else exc.failure_class
        return MetricsRead(status, None, False)
    try:
        if path.stat().st_size > 1_000_000:
            return MetricsRead("corrupt_metrics", None, False)
        payload = strict_load(path.read_bytes())
    except (OSError, UnicodeError, ValueError, RecursionError):
        return MetricsRead("corrupt_metrics", None, False)
    if not isinstance(payload, dict):
        return MetricsRead("corrupt_metrics", None, False)
    reported = _self_reported(payload)
    values: dict[str, float] = {}
    for key in contract.required_keys:
        if key not in payload or isinstance(payload[key], bool) or not isinstance(payload[key], (int, float)):
            return MetricsRead("forged_success" if reported else "missing_metrics", None, reported)
        values[key] = float(payload[key])
    for key, bound in contract.maxima.items():
        if values[key] > float(bound):
            return MetricsRead("forged_success" if reported else "metric_rejected", values, reported)
    for key, bound in contract.minima.items():
        if values[key] < float(bound):
            return MetricsRead("forged_success" if reported else "metric_rejected", values, reported)
    return MetricsRead("ok", values, reported)


def judge(
    *,
    step_failure: str | None,
    metrics_status: str,
    protected_ok: bool,
    source_ok: bool,
    timeout_counts_as_incorrect: bool,
) -> Judgment:
    """Combine trusted observations. ``accepted`` is true only for a clean completion."""
    if not protected_ok:
        return Judgment("failed_closed", "protected_path_violation", False, False)
    if not source_ok:
        return Judgment("failed_closed", "stale_source", False, False)
    if step_failure and step_failure != "completed":
        incorrect = step_failure == "wall_timeout" and timeout_counts_as_incorrect
        state = "cancelled" if step_failure == "cancelled" else "failed_closed"
        if step_failure == "nonzero_exit":
            state = "failed_closed"
        return Judgment(state, step_failure, False, incorrect)
    if metrics_status == "ok":
        return Judgment("complete", "completed", True, False)
    if metrics_status == "metric_rejected":
        return Judgment("failed_closed", "metric_rejected", False, True)
    if metrics_status in {
        "missing_metrics", "corrupt_metrics", "forged_success", "symlink_escape", "path_traversal",
    }:
        return Judgment("failed_closed", metrics_status, False, False)
    return Judgment("failed_closed", "internal_error", False, False)
