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
