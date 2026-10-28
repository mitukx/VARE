import json

import pytest

from scripts import grade_corrected_template_candidate_v1 as grader


@pytest.mark.parametrize("task", ["checkpoint-shards", "gqa-repeat"])
@pytest.mark.parametrize("variant", ["known-fix", "incorrect-mutant"])
def test_predeclared_gqa_and_upstream_control_sources_are_structurally_admitted(task, variant):
    task_record = next(item for item in json.loads(
        (grader.ASSETS / "tasks.json").read_text()
    )["tasks"] if item["id"] == task)
    asset_key = "known_fix_asset" if variant == "known-fix" else "incorrect_mutant_asset"
    source = (grader.ASSETS / task_record[asset_key]).read_text()

    _, function = grader.boundary.admit(task, source)

    assert function.name == task_record["target_function"]


def test_admission_rejects_changes_outside_the_target_function():
    source_path = grader.ASSETS / "gqa-repeat-upstream-fixed.py.txt"
    changed = source_path.read_text().replace("import math", "import os")

    with pytest.raises(ValueError, match="outside the declared function body"):
        grader.boundary.admit("gqa-repeat", changed)
