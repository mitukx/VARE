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


def test_natural_language_search_does_not_fallback_to_unrelated_token(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "_internal" / "_mock_val_ser.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "code: PydanticErrorCodes\n"
        "\n"
        "class MockValSer:\n"
        "    def __deepcopy__(self):\n"
        "        return self\n"
    )
    allowed = {"pydantic/_internal/_mock_val_ser.py"}

    result = runner["search_source"](
        workspace,
        "copy or deep-copy Pydantic models configured with defer_build",
        allowed,
        task_prompt="Copying validators or serializers should preserve deferred state.",
    )

    assert result["matches"] == []
    assert result.get("query_fallback") is not True
    assert "MockValSer" in {row["symbol"] for row in result["symbol_hints"]}


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
    assert [
        (row["path"], row["line"], row["symbol"])
        for row in result["symbol_hints"]
    ] == [
        ("pydantic/example.py", 1, "MockValSer"),
        ("pydantic/example.py", 2, "__deepcopy__"),
        ("pydantic/example.py", 5, "rebuild_validator"),
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


def test_deferred_validator_task_ranks_mock_valser_from_ast_context(tmp_path, monkeypatch):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "_internal" / "_mock_val_ser.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class AttemptRebuild(Protocol):\n"
        "    def __call__(self): ...\n"
        "\n"
        "class MockValSer(Generic[ValSer]):\n"
        '    """Mocker for SchemaValidator or SchemaSerializer. Copying its '
        'validator or serializer should preserve deferred state."""\n'
        "    __slots__ = ('_val_or_ser',)\n"
        "    def __getattr__(self, item):\n"
        "        return item\n"
        "    def rebuild(self):\n"
        "        return None\n"
    )
    allowed = {"pydantic/_internal/_mock_val_ser.py"}
    prompt = (
        "Fix RecursionError when copying or deep-copying Pydantic models configured "
        "with defer_build. Copying their validators or serializers should preserve "
        "deferred state without recursing."
    )

    result = runner["search_source"](
        workspace, "defer_build", allowed, task_prompt=prompt
    )

    assert result["matches"] == []
    assert result["symbol_hints"][0]["symbol"] == "MockValSer"
    assert result["symbol_hints"][0]["task_relevance_score"] > 0


def test_task_ranked_search_returns_full_symbol_and_stops_unrelated_reads(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "_internal" / "_mock_val_ser.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class MockCoreSchema:\n"
        '    """Mocker for CoreSchema."""\n'
        "    def rebuild(self):\n"
        "        return None\n"
        "\n"
        "class MockValSer:\n"
        '    """Copying a SchemaValidator or SchemaSerializer should preserve '
        'deferred state without recursion."""\n'
        "    def __getattr__(self, item):\n"
        "        return item\n"
        "    def rebuild(self):\n"
        "        return None\n"
        + "".join(f"value_{i} = {i}\n" for i in range(40))
    )
    allowed = {"pydantic/_internal/_mock_val_ser.py"}
    prompt = (
        "Fix RecursionError when copying or deep-copying Pydantic models configured "
        "with defer_build. Copying their validators or serializers should preserve "
        "deferred state without recursing."
    )
    targets = {}
    cursors = {}

    search = runner["execute_tool"](
        workspace,
        workspace,
        allowed,
        {"tool": "search_code", "arguments": {"query": "rebuild"}},
        task_prompt=prompt,
        read_cursors=cursors,
        search_targets=targets,
    )
    first_relevant_hit = search["matches"][0]["line"]
    assert first_relevant_hit == 10
    assert search["ranked_declaration_context"]["symbol"] == "MockValSer"
    redirect = runner["execute_tool"](
        workspace,
        workspace,
        allowed,
        {
            "tool": "read_file",
            "arguments": {
                "path": "pydantic/_internal/_mock_val_ser.py",
                "start_line": 1,
                "end_line": 5,
            },
        },
        task_prompt=prompt,
        read_cursors=cursors,
        search_targets=targets,
    )
    after_consuming_hit = runner["execute_tool"](
        workspace,
        workspace,
        allowed,
        {
            "tool": "read_file",
            "arguments": {
                "path": "pydantic/_internal/_mock_val_ser.py",
                "start_line": first_relevant_hit + 20,
                "end_line": first_relevant_hit + 30,
            },
        },
        task_prompt=prompt,
        read_cursors=cursors,
        search_targets=targets,
    )

    assert redirect["navigation_stop"]["symbol"] == "MockValSer"
    assert redirect["content"] == ""
    assert after_consuming_hit["navigation_stop"]["symbol"] == "MockValSer"


def test_task_ranked_declaration_stops_broad_paging_after_full_read(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "_internal" / "_mock_val_ser.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "".join(f"value_{i} = {i}\n" for i in range(20))
        + "class MockValSer:\n"
        + '    """SchemaValidator and SchemaSerializer mocks preserve deferred state."""\n'
        + "    def __getattr__(self, item):\n"
        + "        return item\n"
    )
    allowed = {"pydantic/_internal/_mock_val_ser.py"}
    prompt = (
        "Fix RecursionError when copying deferred SchemaValidator and SchemaSerializer "
        "mocks. Preserve deferred state without recursing."
    )
    targets = {}
    cursors = {}

    search = runner["execute_tool"](
        workspace,
        workspace,
        allowed,
        {"tool": "search_code", "arguments": {"query": "defer_build"}},
        task_prompt=prompt,
        read_cursors=cursors,
        search_targets=targets,
    )
    declaration = search["symbol_hints"][0]
    assert search["ranked_declaration_context"]["symbol"] == "MockValSer"
    assert targets == {}
    assert "def __getattr__" in search["ranked_declaration_context"]["content"]
    assert targets == {}
    unrelated_read = runner["execute_tool"](
        workspace,
        workspace,
        allowed,
        {
            "tool": "read_file",
            "arguments": {
                "path": declaration["path"],
                "start_line": 1,
                "end_line": 20,
            },
        },
        task_prompt=prompt,
        read_cursors=cursors,
        search_targets=targets,
    )
    assert unrelated_read["navigation_stop"]["reason"] == (
        "task_ranked_declaration_read_complete"
    )
    assert unrelated_read["content"] == ""


def test_compact_action_summary_retains_full_ranked_declaration_after_navigation_stop(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)
    declaration = "\n".join(
        [
            "76: class MockValSer:",
            "81:     __slots__ = '_error_message', '_code', '_val_or_ser', '_attempt_rebuild'",
            "96:     def __getattr__(self, item):",
            "98:         if self._attempt_rebuild:",
            "104:         getattr(self._val_or_ser, item)",
            "107:     def rebuild(self):",
        ]
    )
    turns = [
        {
            "turn": 1,
            "action": {"tool": "search_code", "arguments": {"query": "defer_build"}},
            "tool_result": {
                "matches": [],
                "ranked_declaration_context": {
                    "path": "pydantic/_internal/_mock_val_ser.py",
                    "symbol": "MockValSer",
                    "start_line": 76,
                    "end_line": 114,
                    "content": declaration,
                },
            },
        },
        {
            "turn": 2,
            "action": {
                "tool": "read_file",
                "arguments": {"path": "pydantic/_internal/_mock_val_ser.py", "start_line": 75, "end_line": 115},
            },
            "tool_result": {
                "content": "",
                "navigation_stop": {
                    "symbol": "MockValSer",
                    "start_line": 76,
                    "end_line": 114,
                },
            },
        },
    ]

    summary = runner["compact_action_summary"](turns)

    assert declaration in summary
    assert "__slots__" in summary and "def __getattr__" in summary
    assert "getattr(self._val_or_ser, item)" in summary


def test_ranked_declaration_tool_result_keeps_context_out_of_escaped_preview(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)
    content = "source line\n" * 192
    result = {
        "matches": [{"line": index, "text": f"match_{index}"} for index in range(1, 7)],
        "symbol_hints": [
            {"line": index, "symbol": f"TaskRelevantSymbol{index}", "path": "pydantic/a.py"}
            for index in range(1, 7)
        ],
        "search_loop_redirect": {
            "reason": "repeated_related_searches_returned_no_matches",
            "related_empty_searches": 3,
        },
        "ranked_declaration_context": {
            "path": "pydantic/a.py",
            "symbol": "MockValSer",
            "start_line": 1,
            "end_line": 160,
            "content": content,
        },
    }
    assert len(__import__("json").dumps(result, ensure_ascii=False)) > 3200

    bounded = runner["bounded_result"](result, max_chars=3200)

    assert bounded["ranked_declaration_context"]["content"] == content
    assert bounded["search_loop_redirect"]["related_empty_searches"] == 3
    assert bounded["matches"] == result["matches"][:1]
    assert bounded["matches_total"] == 6
    assert bounded["symbol_hints"] == result["symbol_hints"][:1]
    assert bounded["symbol_hints_total"] == 6
    assert "preview" not in bounded


def test_prompt_treats_included_ranked_context_as_already_read(tmp_path, monkeypatch):
    runner = _runner_namespace(tmp_path, monkeypatch)

    for prompt in (runner["SYSTEM_PROMPT"], runner["TOOL_HELP"]):
        assert "ranked_declaration_context" in prompt
        assert "already" in prompt
        assert "do not reread" in prompt or "do not call read_file" in prompt


def test_recorded_deferred_search_variants_redirect_after_three_misses(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "_internal" / "_mock_val_ser.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class MockValSer:\n"
        '    """Mocker for SchemaValidator and SchemaSerializer."""\n'
        "    def rebuild(self):\n"
        "        return None\n"
    )
    allowed = {"pydantic/_internal/_mock_val_ser.py"}
    prompt = "Copying validators or serializers should preserve deferred state."
    guard = runner["SearchLoopGuard"](empty_search_limit=3)
    cursors = {}
    actions = ["defer_build", "deferred_build_state", "DeferredBuildState"]

    results = []
    for query in actions:
        results.append(
            runner["execute_tool"](
                workspace,
                workspace,
                allowed,
                {"tool": "search_code", "arguments": {"query": query}},
                task_prompt=prompt,
                read_cursors=cursors,
                search_loop_guard=guard,
            )
        )
    redirected = runner["execute_tool"](
        workspace,
        workspace,
        allowed,
        {"tool": "search_code", "arguments": {"query": "build_deferred_state"}},
        task_prompt=prompt,
        read_cursors=cursors,
        search_loop_guard=guard,
    )

    assert all(result["matches"] == [] for result in results)
    assert results[-1]["search_loop_redirect"]["related_empty_searches"] == 3
    assert results[-1]["ranked_declaration_context"]["start_line"] == 1
    assert "SchemaValidator" in results[-1]["ranked_declaration_context"]["content"]
    assert redirected["search_loop_redirect"]["executed"] is False
    assert redirected["symbol_hints"][0]["symbol"] == "MockValSer"


def test_historical_14115_search_trace_is_stopped_by_controller_guard():
    import gzip
    import json
    from vare.agent_navigation import SearchLoopGuard

    root = Path(__file__).resolve().parents[1]
    trace_path = root / (
        "results/vare-pydantic-14115-full-trajectory-v1/attempt-001/"
        "tool-traces.jsonl.gz"
    )
    guard = SearchLoopGuard(empty_search_limit=3)
    search_path = "pydantic/_internal/_mock_val_ser.py"
    searches = executed = intercepted = redirects = 0

    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("sample_index") != 0:
                continue
            action = row.get("action") or {}
            tool = action.get("tool")
            if tool == "read_file":
                guard.reset()
                continue
            if tool != "search_code":
                continue
            searches += 1
            query = action.get("arguments", {}).get("query", "")
            assert not (row.get("tool_result") or {}).get("matches")
            if guard.intercept(query, search_path) is not None:
                intercepted += 1
                continue
            executed += 1
            if guard.observe(query, 0, search_path) is not None:
                redirects += 1

    assert searches == 15
    assert executed == 6
    assert intercepted == 9
    assert redirects == 1


def test_recorded_jsonvalue_declaration_prefix_replay_finds_bare_alias(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "types.py"
    source.parent.mkdir(parents=True)
    source.write_text("Other = int\nJsonValue = int | str\n")

    result = runner["search_source"](
        workspace, "class JsonValue", {"pydantic/types.py"}
    )

    assert result["query_fallback"] is True
    assert result["matched_query"] == "JsonValue"
    assert result["matches"][0]["line"] == 2


def test_search_match_redirects_misleading_read_to_runtime_declaration(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "types.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from typing import TYPE_CHECKING\n"
        + "".join(f"value_{i} = {i}\n" for i in range(20))
        +
        "if TYPE_CHECKING:\n"
        "    JsonValue: TypeAlias = 'int | str'\n"
        "else:\n"
        "    JsonValue = TypeAliasType('JsonValue', custom_serializer())\n"
    )
    allowed = {"pydantic/types.py"}
    prompt = "Fix Python-mode serialization of JsonValue values."
    targets = {}

    search = runner["execute_tool"](
        workspace,
        workspace,
        allowed,
        {"tool": "search_code", "arguments": {"query": "class JsonValue"}},
        task_prompt=prompt,
        search_targets=targets,
    )
    read = runner["execute_tool"](
        workspace,
        workspace,
        allowed,
        {
            "tool": "read_file",
            "arguments": {"path": "pydantic/types.py", "start_line": 1, "end_line": 5},
        },
        task_prompt=prompt,
        search_targets=targets,
    )

    assert search["matches"][0]["line"] == 25
    assert read["navigation_redirect"]["matched_line"] == 25
    assert read["start_line"] > 1
    assert "JsonValue = TypeAliasType" in read["content"]
    assert read["requested_start_line"] == 1


def test_recorded_generic_model_search_ranks_factory_before_unrelated_declarations(
    tmp_path, monkeypatch
):
    runner = _runner_namespace(tmp_path, monkeypatch)
    workspace = tmp_path / "candidate"
    source = workspace / "pydantic" / "_internal" / "_generics.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class LimitedDict(dict):\n"
        "    pass\n"
        "\n"
        "class PydanticGenericMetadata: pass\n"
        "\n"
        "def create_generic_submodel():\n"
        "    return None\n"
    )

    result = runner["search_source"](
        workspace,
        "get_model_type_hints",
        {"pydantic/_internal/_generics.py"},
        task_prompt=(
            "Fix pickling for generic Pydantic model classes parametrized in a "
            "different module."
        ),
    )

    assert result["matches"] == []
    assert result["symbol_hints"][0]["symbol"] == "create_generic_submodel"


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
