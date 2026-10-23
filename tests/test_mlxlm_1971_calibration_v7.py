"""Calibration-gate regressions for full-forward and cache-only comparisons."""

from __future__ import annotations

from scripts.run_mlxlm_1971_calibration_v7 import _has_semantic_decisions


SCALES = [0.25, 0.5, 0.8]


def _report(passed: bool, outcomes: list[tuple[str, bool]]) -> dict:
    cases = []
    case_ids = [
        (scale, phase)
        for scale in SCALES
        for phase in ("full", "cached")
    ]
    for (scale, phase), (failure, ok) in zip(case_ids, outcomes, strict=True):
        cases.append(
            {
                "case_id": f"value-scale-{scale:g}-{phase}",
                "failure": failure or None,
                "passed": ok,
            }
        )
    return {"passed": passed, "cases": cases}


def test_v6_gate_requires_semantic_baseline_fix_and_constant_patterns():
    baseline = _report(False, [("wrong_output", False)] * 6)
    fixed = _report(True, [("", True)] * 6)
    constant = _report(
        False,
        [
            ("wrong_output", False), ("wrong_output", False),
            ("", True), ("", True),
            ("wrong_output", False), ("wrong_output", False),
        ],
    )
    cache_mutant = _report(
        False,
        [
            ("", True), ("wrong_output", False),
            ("", True), ("wrong_output", False),
            ("", True), ("wrong_output", False),
        ],
    )
    assert _has_semantic_decisions(baseline, fixed, constant, cache_mutant, SCALES)


def test_v6_gate_rejects_full_forward_failure_for_cache_mutant():
    baseline = _report(False, [("wrong_output", False)] * 6)
    fixed = _report(True, [("", True)] * 6)
    constant = _report(
        False,
        [
            ("wrong_output", False), ("wrong_output", False),
            ("", True), ("", True),
            ("wrong_output", False), ("wrong_output", False),
        ],
    )
    unrelated_failure = _report(
        False,
        [
            ("wrong_output", False), ("wrong_output", False),
            ("wrong_output", False), ("wrong_output", False),
            ("wrong_output", False), ("wrong_output", False),
        ],
    )
    assert not _has_semantic_decisions(
        baseline, fixed, constant, unrelated_failure, SCALES
    )


def test_v6_gate_rejects_timeout_or_missing_case():
    baseline = _report(False, [("wrong_output", False)] * 6)
    baseline["cases"][0]["failure"] = "timeout"
    fixed = _report(True, [("", True)] * 6)
    constant = _report(
        False,
        [
            ("wrong_output", False), ("wrong_output", False),
            ("", True), ("", True),
            ("wrong_output", False), ("wrong_output", False),
        ],
    )
    cache_mutant = _report(
        False,
        [
            ("", True), ("wrong_output", False),
            ("", True), ("wrong_output", False),
            ("", True), ("wrong_output", False),
        ],
    )
    cache_mutant["cases"].pop()
    assert not _has_semantic_decisions(
        baseline, fixed, constant, cache_mutant, SCALES
    )
