"""Calibration-gate regressions for full-forward and cache-only comparisons."""

from __future__ import annotations

from scripts.run_mlxlm_1971_calibration_v8 import _has_semantic_decisions


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


def test_v7_gate_requires_semantic_baseline_fix_and_constant_patterns():
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


def test_v7_gate_rejects_full_forward_failure_for_cache_mutant():
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


def test_v7_gate_rejects_timeout_or_missing_case():
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


def test_v8_cache_mutant_preserves_full_forward_rope_branch(tmp_path):
    from scripts.run_mlxlm_1971_calibration_v8 import _mutate_workspace
    from vare.mlxlm_1971_repo_task import TARGET_PATH

    source = tmp_path / "source"
    target = source / TARGET_PATH
    target.parent.mkdir(parents=True)
    target.write_text(
        """        if self.value_scale is not None:
            values = values * self.value_scale

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
""",
        encoding="utf-8",
    )
    destination = tmp_path / "mutant"
    _mutate_workspace(source, destination, "scale_after_cache")
    text = (destination / TARGET_PATH).read_text(encoding="utf-8")
    assert "        else:\n            queries = self.rope(queries)\n            keys = self.rope(keys)" in text
    assert text.index("keys, values = cache.update_and_fetch") < text.index(
        "values = values * self.value_scale"
    )
