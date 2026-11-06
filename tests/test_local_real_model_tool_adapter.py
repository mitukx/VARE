from __future__ import annotations

import runpy
from pathlib import Path


def _runner_namespace(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VARE_EXPERIMENT_ROOT", str(tmp_path))
    runner = Path(__file__).resolve().parents[1] / "experiments" / (
        "run_vare_pydantic_14138_full_trajectory_continuation.py"
    )
    return runpy.run_path(str(runner))


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
