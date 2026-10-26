"""Integration controls for the newly versioned authored-task isolation path."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import grade_mlsys_coding_v2_candidate_v2 as grader


pytestmark = pytest.mark.skipif(
    os.environ.get("VARE_RUN_AUTHORED_MLSYS_ISOLATION_TESTS") != "1",
    reason="run the container-backed authored-task controls on the designated CI lane",
)


@pytest.mark.parametrize("task", sorted(grader.TASK_SPECS))
@pytest.mark.parametrize(
    ("variant_index", "expected"),
    [(0, False), (1, True), (2, False)],
)
def test_controls_are_independently_graded_in_the_pinned_container(
    task: str, variant_index: int, expected: bool,
) -> None:
    spec = grader.TASK_SPECS[task]
    source = grader.ROOT / "benchmarks/corrected_template_agent_v2" / spec["control_names"][variant_index]
    result = grader.grade(task, source)
    assert result.get("admitted") is True, json.dumps(result, sort_keys=True)
    assert result.get("isolated_candidate_execution") is True, json.dumps(result, sort_keys=True)
    isolated = result.get("isolated_grade", {}).get("decision", {})
    assert isolated.get("success") is expected, json.dumps(result, sort_keys=True)
    assert result.get("success") is expected, json.dumps(result, sort_keys=True)


def test_isolation_protocol_and_task_pack_are_hash_pinned() -> None:
    protocol = grader._load_protocol()
    grader._verify_task_pack(protocol)
    assert protocol["resource_limits"]["network"] == "none"
    assert protocol["verifier"]["expected_outputs_mounted_into_candidate"] is False


@pytest.mark.parametrize("task", sorted(grader.TASK_SPECS))
def test_public_smoke_is_separate_from_full_isolated_grading(task: str) -> None:
    spec = grader.TASK_SPECS[task]
    baseline = grader.ROOT / "benchmarks/corrected_template_agent_v2" / spec["control_names"][0]
    public = grader.grade(task, baseline, public=True)
    assert public["public"] is True
    assert public["success"] is True
    assert public["isolated_candidate_execution"] is False


def test_candidate_admission_rejects_changes_outside_the_target(tmp_path: Path) -> None:
    baseline = (grader.ROOT / "benchmarks/corrected_template_agent_v2/label-smooth-baseline.py.txt").read_text()
    changed = baseline.replace("import math", "import os")
    source = tmp_path / "candidate.py"
    source.write_text(changed, encoding="utf-8")
    result = grader.grade("label-smooth", source)
    assert result["admitted"] is False
    assert result["success"] is False
