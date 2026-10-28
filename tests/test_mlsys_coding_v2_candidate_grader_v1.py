from pathlib import Path

import pytest

from scripts import grade_mlsys_coding_v2_candidate_v1 as grader


@pytest.mark.parametrize("task", sorted(grader.boundary.TASKS))
@pytest.mark.parametrize(
    ("variant", "expected"),
    [("baseline", False), ("known-fix", True), ("incorrect-mutant", False)],
)
def test_control_bodies_match_the_narrow_function_contract(task, variant, expected):
    files = {
        "baseline": f"{task}-baseline.py.txt",
        "known-fix": f"{task}-known-fix.py.txt",
        "incorrect-mutant": f"{task}-incorrect-mutant.py.txt",
    }
    source = (grader.ASSETS / files[variant]).read_text(encoding="utf-8")
    implementation, _trusted_module = grader.boundary.load_candidate(task, source)
    checks = (
        grader.evaluate_label_smooth(implementation, public=False)
        if task == "label-smooth"
        else grader.evaluate_ema(implementation, public=False)
    )
    assert all(checks.values()) is expected, checks


@pytest.mark.parametrize("task", sorted(grader.boundary.TASKS))
def test_public_smoke_contracts_remain_separate_from_full_grading(task):
    baseline = grader.ASSETS / f"{task}-baseline.py.txt"
    implementation, _trusted_module = grader.boundary.load_candidate(
        task, baseline.read_text(encoding="utf-8"),
    )
    public_checks = (
        grader.evaluate_label_smooth(implementation, public=True)
        if task == "label-smooth"
        else grader.evaluate_ema(implementation, public=True)
    )
    full_checks = (
        grader.evaluate_label_smooth(implementation, public=False)
        if task == "label-smooth"
        else grader.evaluate_ema(implementation, public=False)
    )
    assert all(public_checks.values()) is True
    assert all(full_checks.values()) is False


def test_candidate_admission_rejects_changes_outside_the_target(tmp_path: Path):
    baseline = (grader.ASSETS / "label-smooth-baseline.py.txt").read_text()
    changed = baseline.replace("import math", "import os")
    source = tmp_path / "candidate.py"
    source.write_text(changed)
    result = grader.grade("label-smooth", source)
    assert result["admitted"] is False
    assert result["success"] is False
