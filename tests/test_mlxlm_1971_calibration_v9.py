"""Calibration-gate regressions for full-forward and cache-only comparisons."""

from __future__ import annotations

import pytest

from scripts.run_mlxlm_1971_calibration_v9 import _has_semantic_decisions


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


def test_v8_gate_requires_semantic_baseline_fix_and_constant_patterns():
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


def test_v8_gate_rejects_full_forward_failure_for_cache_mutant():
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


def test_v8_gate_rejects_timeout_or_missing_case():
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


def test_v9_cache_mutant_preserves_full_forward_rope_branch(tmp_path):
    from scripts.run_mlxlm_1971_calibration_v9 import _mutate_workspace
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


@pytest.mark.parametrize("flag", ["S", "h"])
def test_v9_rejects_hidden_index_flags_on_locked_inputs(monkeypatch, flag):
    from types import SimpleNamespace

    from vare import mlxlm_1971_repo_task_v9 as task

    def fake_run(command, **kwargs):
        assert command[command.index("ls-files") + 1] == "-v"
        names = command[command.index("--") + 1 :]
        rows = [f"H {name}" for name in names]
        rows[0] = f"{flag} {names[0]}"
        return SimpleNamespace(stdout="\n".join(rows))

    monkeypatch.setattr(task.subprocess, "run", fake_run)
    with pytest.raises(task.MlxLm1971TaskError, match="hidden Git index flag"):
        task.verify_protocol_v9()


def test_public_grader_dispatches_to_v9(monkeypatch, tmp_path):
    from vare import mlxlm_1971_repo_task as public_task
    from vare import mlxlm_1971_repo_task_v9 as versioned_task

    expected = {"protocol_id": "v9-dispatch-test"}
    monkeypatch.setattr(versioned_task, "grade_workspace_v9", lambda *a, **k: expected)
    assert public_task.grade_workspace(tmp_path) is expected


def test_public_export_dispatches_to_v9(monkeypatch, tmp_path):
    from vare import mlxlm_1971_repo_task as public_task
    from vare import mlxlm_1971_repo_task_v9 as versioned_task

    expected = {"protocol_id": "v9-export-dispatch-test"}
    monkeypatch.setattr(versioned_task, "export_candidate_v9", lambda *a, **k: expected)
    assert public_task.export_candidate(tmp_path, tmp_path / "candidate") is expected
