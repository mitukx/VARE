from __future__ import annotations

import os

from scripts.sprint_viii_transformers_49703_grader_v1 import grade_bytes, read_source


def _module_with_initializer(line: str, *, append_branches: bool = True) -> bytes:
    appends = (
        "        if condition:\n            self.input_projections.append(nn.Conv2d(2, 2, 1))\n"
        "        else:\n            self.input_projections.append(nn.Sequential())\n"
        if append_branches else ""
    )
    return (
        "class Mask2FormerTransformerModule(nn.Module):\n"
        "    def __init__(self):\n"
        f"        {line}\n"
        f"{appends}"
    ).encode()


def test_grader_calibration_baseline_known_fix_and_wrong_mutant() -> None:
    baseline = grade_bytes(_module_with_initializer("self.input_projections = []"))
    known_fix = grade_bytes(_module_with_initializer("self.input_projections = nn.ModuleList()"))
    wrong_mutant = grade_bytes(_module_with_initializer("self.input_projections = nn.ModuleDict()"))
    assert baseline["status"] == "FAIL"
    assert known_fix["status"] == "PASS"
    assert wrong_mutant["status"] == "FAIL"


def test_grader_accepts_empty_modulelist_and_requires_both_append_paths() -> None:
    equivalent = grade_bytes(_module_with_initializer("self.input_projections = nn.ModuleList([])"))
    missing_branch = grade_bytes(
        _module_with_initializer("self.input_projections = nn.ModuleList()", append_branches=False)
    )
    assert equivalent["status"] == "PASS"
    assert missing_branch["reason"] == "projection_branches_do_not_register_both_paths"


def test_grader_reads_only_regular_non_symlink_candidate_files(tmp_path) -> None:
    source = tmp_path / "target.py"
    source.write_bytes(_module_with_initializer("self.input_projections = nn.ModuleList()"))
    os.symlink(source.name, tmp_path / "linked.py")
    data, error = read_source(str(tmp_path), "target.py")
    assert data == source.read_bytes()
    assert error is None
    data, error = read_source(str(tmp_path), "linked.py")
    assert data is None
    assert error == "source_symlink_rejected"
