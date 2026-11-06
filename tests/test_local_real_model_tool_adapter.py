from __future__ import annotations

import runpy
from pathlib import Path


def _runner_namespace(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VARE_EXPERIMENT_ROOT", str(tmp_path))
    runner = Path(__file__).resolve().parents[1] / "experiments" / (
        "run_vare_pydantic_14138_full_trajectory_continuation.py"
    )
    return runpy.run_path(str(runner))


def test_task_arm_map_uses_qualified_13994_and_excludes_exposed_13972(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)

    assert runner["TASK_ARM_DIRS"][
        "pydantic-13994-generic-model-cross-module-pickling"
    ] == tmp_path / "work/fresh-train-dev/task-arms-v14/pydantic-13994/baseline"
    assert "pydantic-13972-decimal-whole-digit-after-validator" not in runner[
        "TASK_ARM_DIRS"
    ]


def test_baseline_source_hashes_normalizes_single_file_task_manifest(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)

    normalized = runner["baseline_source_hashes"](
        {
            "editable_path": "pydantic/_internal/_mock_val_ser.py",
            "baseline_source_sha256": "abc123",
            "source_hashes": {"baseline": "abc123"},
        }
    )

    assert normalized == {"pydantic/_internal/_mock_val_ser.py": "abc123"}


def test_recover_recorded_mlx_hunk_and_unescaped_patch_quotes(tmp_path, monkeypatch):
    runner = _runner_namespace(tmp_path, monkeypatch)
    raw = r'''{"tool":"apply_patch","arguments":{"path":"pydantic/example.py",'''
    raw += r'''"hunk":"@@ -1 +1 @@\n-old = 1\n+new = """value"""\n"}}'''

    action, error = runner["action_from_text"](raw)

    assert error is None
    assert action is not None
    assert action["output_normalization"] == "malformed_apply_patch_json_salvaged"
    path, patch, normalizations = runner["normalize_patch_arguments"](
        action["arguments"], {"pydantic/example.py"}
    )
    runner["validate_patch"](patch, {"pydantic/example.py"})
    assert path == "pydantic/example.py"
    assert normalizations == [
        "hunk_argument_aliased_to_patch",
        "unified_diff_file_headers_synthesized_for_allowlisted_path",
        "unified_diff_hunk_line_counts_recomputed",
    ]
    assert 'old = 1' in patch
    assert 'new = """value"""' in patch


def test_infer_omitted_path_only_for_single_allowlisted_file(tmp_path, monkeypatch):
    runner = _runner_namespace(tmp_path, monkeypatch)
    allowed = {"pydantic/example.py"}

    path, patch, normalizations = runner["normalize_patch_arguments"](
        {"patch": "@@ -1,9 +1,9 @@\n-old\n+new\n"}, allowed
    )

    runner["validate_patch"](patch, allowed)
    assert path == "pydantic/example.py"
    assert normalizations == [
        "single_editable_path_inferred",
        "unified_diff_file_headers_synthesized_for_allowlisted_path",
        "unified_diff_hunk_line_counts_recomputed",
    ]


def test_salvage_truncated_patch_json_requires_allowlisted_path(tmp_path, monkeypatch):
    runner = _runner_namespace(tmp_path, monkeypatch)
    raw = r'''{"tool":"apply_patch","arguments":{"path":"pydantic/example.py","patch":"@@ -1 +1 @@\n-old\n+new'''

    action, error = runner["action_from_text"](raw)

    assert error is None
    assert action is not None
    assert action["output_normalization"] == (
        "malformed_apply_patch_json_salvaged_incomplete"
    )
    path, patch, _ = runner["normalize_patch_arguments"](
        action["arguments"], {"pydantic/example.py"}
    )
    runner["validate_patch"](patch, {"pydantic/example.py"})
    assert path == "pydantic/example.py"


def test_read_file_marks_the_twenty_line_cap_as_truncated(tmp_path, monkeypatch):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("".join(f"value_{i} = {i}\n" for i in range(30)))

    result = runner["read_source"](
        workspace,
        {"path": "pydantic/example.py", "start_line": 1, "end_line": 25},
        {"pydantic/example.py"},
    )

    assert result["end_line"] == 20
    assert result["content_truncated"] is True
    assert result["truncation_reasons"] == ["line_limit"]
    assert result["next_start_line"] == 21


def test_search_code_falls_back_to_bare_identifier_and_guides_after_miss(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("JsonValue = int | str\n")
    allowed = {"pydantic/example.py"}

    fallback = runner["search_source"](workspace, "class JsonValue", allowed)
    assert fallback["query_fallback"] is True
    assert fallback["requested_query"] == "class JsonValue"
    assert fallback["matched_query"] == "JsonValue"
    assert fallback["matches"][0]["line"] == 1
    assert "focused source window" in fallback["next_step_hint"]

    missing = runner["search_source"](workspace, "NoSuchSymbol", allowed)
    assert missing["matches"] == []
    assert "read the most relevant declaration" in missing["next_step_hint"].lower()
    assert "do not page consecutively" in missing["next_step_hint"].lower()


def test_search_code_no_match_returns_allowlisted_symbol_hints(tmp_path, monkeypatch):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class MockValSer:\n"
        "    def __deepcopy__(self):\n"
        "        return self\n"
        "\n"
        "def rebuild_validator():\n"
        "    return None\n"
    )

    result = runner["search_source"](
        workspace, "defer_build", {"pydantic/example.py"}
    )

    assert result["matches"] == []
    assert result["symbol_hints"] == [
        {"path": "pydantic/example.py", "line": 1, "symbol": "MockValSer"},
        {"path": "pydantic/example.py", "line": 2, "symbol": "__deepcopy__"},
        {"path": "pydantic/example.py", "line": 5, "symbol": "rebuild_validator"},
    ]
    assert "symbol_hints" in result["next_step_hint"]
    assert all(row["path"] == "pydantic/example.py" for row in result["symbol_hints"])


def test_no_match_symbol_hints_rank_against_task_prompt(tmp_path, monkeypatch):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "_internal" / "_generics.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class PydanticGenericMetadata:\n"
        "    pass\n"
        "\n"
        "def create_generic_submodel():\n"
        "    return None\n"
        "\n"
        "def get_model_typevars_map():\n"
        "    return {}\n"
    )

    result = runner["search_source"](
        workspace,
        "NoSuchSymbol",
        {"pydantic/_internal/_generics.py"},
        task_prompt="Fix pickling for generic Pydantic model classes in another module",
    )

    assert result["matches"] == []
    assert result["symbol_hints"][0]["symbol"] == "create_generic_submodel"
    assert result["symbol_hints"][0]["task_relevance_score"] > result["symbol_hints"][1][
        "task_relevance_score"
    ]
    assert "task-ranked" in result["next_step_hint"]


def test_repeated_oversized_read_auto_pages_instead_of_repeating(tmp_path, monkeypatch):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("".join(f"value_{i} = {i}\n" for i in range(50)))
    allowed = {"pydantic/example.py"}
    action = {
        "tool": "read_file",
        "arguments": {
            "path": "pydantic/example.py",
            "start_line": 1,
            "end_line": 45,
        },
    }
    cursors = {}

    first = runner["execute_tool"](
        workspace, workspace, allowed, action, read_cursors=cursors
    )
    second = runner["execute_tool"](
        workspace, workspace, allowed, action, read_cursors=cursors
    )
    third = runner["execute_tool"](
        workspace, workspace, allowed, action, read_cursors=cursors
    )
    repeated_after_completion = runner["execute_tool"](
        workspace, workspace, allowed, action, read_cursors=cursors
    )

    assert [
        (first["start_line"], first["end_line"]),
        (second["start_line"], second["end_line"]),
        (third["start_line"], third["end_line"]),
    ] == [(1, 20), (21, 40), (41, 45)]
    assert first["content_truncated"] is True
    assert second["auto_paged"] is True
    assert third["content_truncated"] is False
    assert repeated_after_completion["already_paged"] is True


def test_overlapping_oversized_reads_return_next_unseen_page(tmp_path, monkeypatch):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("".join(f"value_{i} = {i}\n" for i in range(80)))
    allowed = {"pydantic/example.py"}
    cursors = {}

    def read(start, end):
        return runner["execute_tool"](
            workspace,
            workspace,
            allowed,
            {
                "tool": "read_file",
                "arguments": {
                    "path": "pydantic/example.py",
                    "start_line": start,
                    "end_line": end,
                },
            },
            read_cursors=cursors,
        )

    first = read(1, 21)
    second = read(21, 41)
    overlapping = read(10, 61)
    final_unseen = read(10, 61)
    already_read = read(10, 61)
    narrow_reread = read(10, 15)

    assert [
        (first["start_line"], first["end_line"]),
        (second["start_line"], second["end_line"]),
        (overlapping["start_line"], overlapping["end_line"]),
        (final_unseen["start_line"], final_unseen["end_line"]),
    ] == [(1, 20), (21, 40), (41, 60), (61, 61)]
    assert overlapping["requested_start_line"] == 10
    assert overlapping["auto_paged"] is True
    assert already_read["already_read"] is True
    assert already_read["content_replayed"] is True
    assert (already_read["start_line"], already_read["end_line"]) == (10, 29)
    assert "10: value_9 = 9" in already_read["content"]
    compacted = runner["compact_action_summary"](
        [
            {
                "turn": 5,
                "action": {
                    "tool": "read_file",
                    "arguments": {
                        "path": "pydantic/example.py",
                        "start_line": 10,
                        "end_line": 61,
                    },
                },
                "tool_result": already_read,
            }
        ]
    )
    assert "Last visible source page" in compacted
    assert "10: value_9 = 9" in compacted
    assert narrow_reread["start_line"] == 10
    assert narrow_reread["end_line"] == 15


def test_patch_normalization_adds_missing_terminal_newline(tmp_path, monkeypatch):
    runner = _runner_namespace(tmp_path, monkeypatch)

    path, patch, normalizations = runner["normalize_patch_arguments"](
        {
            "path": "pydantic/example.py",
            "patch": "--- a/pydantic/example.py\n+++ b/pydantic/example.py\n"
            "@@ -1 +1 @@\n-old = 1\n+new = 1",
        },
        {"pydantic/example.py"},
    )

    runner["validate_patch"](patch, {"pydantic/example.py"})
    assert path == "pydantic/example.py"
    assert patch.endswith("\n")
    assert "patch_final_newline_added" in normalizations


def test_replace_text_applies_only_an_exact_unique_source_span(tmp_path, monkeypatch):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("def load():\n    return old_value\n")
    allowed = {"pydantic/example.py"}

    result = runner["execute_tool"](
        workspace,
        workspace,
        allowed,
        {
            "tool": "replace_text",
            "arguments": {
                "path": "pydantic/example.py",
                "old_text": "    return old_value",
                "new_text": "    return new_value",
            },
        },
    )

    assert result["applied"] is True
    assert result["changed_files"] == ["pydantic/example.py"]
    assert source.read_text() == "def load():\n    return new_value\n"


def test_replace_text_rejects_missing_or_ambiguous_spans_without_writing(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "example.py"
    source.parent.mkdir(parents=True)
    original = "value = 1\nvalue = 1\n"
    source.write_text(original)
    allowed = {"pydantic/example.py"}

    result = runner["execute_tool"](
        workspace,
        workspace,
        allowed,
        {
            "tool": "replace_text",
            "arguments": {
                "path": "pydantic/example.py",
                "old_text": "value = 1",
                "new_text": "value = 2",
            },
        },
    )

    assert "exactly once" in result["error"]
    assert "found 2 matches" in result["error"]
    assert source.read_text() == original
