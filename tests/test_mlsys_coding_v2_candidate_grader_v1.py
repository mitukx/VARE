from pathlib import Path

import pytest

from scripts import grade_mlsys_coding_v2_candidate_v1 as grader


@pytest.mark.parametrize("task", sorted(grader.boundary.TASKS))
@pytest.mark.parametrize(
    ("variant", "expected"),
    [("baseline", False), ("known-fix", True), ("incorrect-mutant", False)],
)
def test_qualified_controls_match_the_independent_contract(task, variant, expected):
    files = {
        "baseline": f"{task}-baseline.py.txt",
        "known-fix": f"{task}-known-fix.py.txt",
        "incorrect-mutant": f"{task}-incorrect-mutant.py.txt",
    }
    result = grader.grade(task, grader.ASSETS / files[variant])
    assert result["admitted"] is True
    assert result["success"] is expected


@pytest.mark.parametrize("task", sorted(grader.boundary.TASKS))
def test_public_smoke_contracts_remain_separate_from_full_grading(task):
    baseline = grader.ASSETS / f"{task}-baseline.py.txt"
    public = grader.grade(task, baseline, public=True)
    full = grader.grade(task, baseline)
    assert public["public"] is True and public["success"] is True
    assert full["public"] is False and full["success"] is False


def test_candidate_admission_rejects_changes_outside_the_target(tmp_path: Path):
    baseline = (grader.ASSETS / "label-smooth-baseline.py.txt").read_text()
    changed = baseline.replace("import math", "import os")
    source = tmp_path / "candidate.py"
    source.write_text(changed)
    result = grader.grade("label-smooth", source)
    assert result["admitted"] is False
    assert result["success"] is False
