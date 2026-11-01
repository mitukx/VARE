from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.localize_task_source_v2 import choose_definition_window, localize
from scripts.run_local_agent import load_localization_record


def test_ast_locator_selects_the_explicit_class_definition():
    source = "\n".join(
        ["class Earlier:", "    pass"]
        + [f"value_{index} = {index}" for index in range(100)]
        + ["class ReplicatedWithGradAllReduce:", "    def install_forward(self, module, mesh):", "        return module"]
        + [f"value_{index} = {index}" for index in range(100)]
    )

    result = choose_definition_window(
        "Repair `ReplicatedWithGradAllReduce` in `src/module.py`.", source, "src/module.py",
    )

    assert result["algorithm"] == "task_prompt_ast_entity_window_v1"
    assert result["matched_entity"] == "ReplicatedWithGradAllReduce"
    assert result["matched_node_kind"] == "ClassDef"
    assert "class ReplicatedWithGradAllReduce" in result["content"]
    assert result["window_lines"] == 80


def test_ast_locator_supports_explicit_qualified_method_names():
    result = choose_definition_window(
        "Repair `Cache.update`.",
        "class Cache:\n    def update(self):\n        return 1\n",
        "src/cache.py",
    )
    assert result["matched_entity"] == "Cache.update"
    assert result["matched_node_kind"] == "FunctionDef"
    assert "def update" in result["content"]


def test_ast_locator_fails_closed_on_missing_or_ambiguous_entities():
    with pytest.raises(ValueError, match="does not match"):
        choose_definition_window("Repair `MissingLayer`.", "class Other:\n    pass\n", "x.py")
    with pytest.raises(ValueError, match="multiple"):
        choose_definition_window(
            "Repair `Cache`.", "class Cache:\n    pass\nclass Cache:\n    pass\n", "x.py",
        )


def test_localize_uses_only_prompt_and_candidate_source(tmp_path: Path):
    task = tmp_path / "task"
    task.mkdir()
    task_text = "Repair `Layer.install_forward` in `src/module.py`.\n"
    (task / "TASK.md").write_text(task_text, encoding="utf-8")
    (task / "task.json").write_text(json.dumps({
        "task_id": "fixture",
        "metadata": {"split": "development", "editable_files": ["src/module.py", "test_public.py"]},
        "prompt": task_text,
        "source": {"base_revision": "a" * 40, "calibration_revision": "b" * 40},
    }), encoding="utf-8")
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src/module.py").write_text(
        "class Layer:\n    def install_forward(self, module, mesh):\n        return module\n",
        encoding="utf-8",
    )
    (workspace / "test_public.py").write_text("secret grader text\n", encoding="utf-8")

    record = localize(task, workspace)

    assert record["task_id"] == "fixture"
    assert record["matched_entity"] == "Layer.install_forward"
    assert record["known_fix_or_grader_read"] is False
    assert "secret grader" not in json.dumps(record)
    record_path = task / "localization.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    loaded = load_localization_record(record_path, workspace, {"src/module.py"})
    assert loaded["content_sha256"] == record["content_sha256"]
