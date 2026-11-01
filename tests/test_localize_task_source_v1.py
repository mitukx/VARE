from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.localize_task_source_v1 import choose_window, localize


def test_localizer_selects_a_bounded_window_by_prompt_overlap_with_stable_ties():
    lines = [f"value_{index} = {index}\n" for index in range(200)]
    lines[100] = "targetneedle = 1\n"
    lines[101] = "other targetneedle = 2\n"

    result = choose_window("Repair targetneedle behavior", "".join(lines), "module.py")

    assert result["algorithm"] == "task_prompt_lexical_windows_v1"
    assert result["window_lines"] == 80
    assert result["start_line"] <= 101 <= result["end_line"]
    assert result["end_line"] - result["start_line"] + 1 == 80
    assert result["lexical_overlap_score"] >= 2


def test_localizer_fails_closed_without_source_overlap():
    with pytest.raises(ValueError, match="no lexical match"):
        choose_window("repair an optimizer", "class Cache:\n    pass\n", "module.py")


def test_localize_uses_only_task_text_and_candidate_source(tmp_path: Path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "TASK.md").write_text("Repair alpha_route handling in the model.\n", encoding="utf-8")
    (task / "task.json").write_text(json.dumps({
        "task_id": "fixture",
        "metadata": {"editable_files": ["module.py", "test_public.py"], "split": "development"},
        "prompt": "Repair alpha_route handling in the model.",
        "source": {"base_revision": "a" * 40, "calibration_revision": "b" * 40},
    }), encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "class Model:\n    def alpha_route(self):\n        return 1\n"
    (workspace / "module.py").write_text(source, encoding="utf-8")
    (workspace / "test_public.py").write_text("SECRET_GRADER_CONTENT\n", encoding="utf-8")

    result = localize(task, workspace)

    assert result["task_id"] == "fixture"
    assert result["path"] == "module.py"
    assert "alpha_route" in result["content"]
    assert "SECRET_GRADER_CONTENT" not in json.dumps(result)
    assert result["known_fix_or_grader_read"] is False
