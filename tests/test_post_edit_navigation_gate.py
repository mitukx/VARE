import hashlib

from scripts.run_local_agent import (
    source_read_progress_key,
    tools_for_no_progress_guard,
    tools_for_post_edit_navigation_gate,
)


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def test_post_edit_gate_is_inactive_until_an_edit_applies() -> None:
    tools = [_tool(name) for name in (
        "list_files", "search_source", "read_lines", "list_functions",
        "replace_lines_transactional", "check_syntax", "finish",
    )]

    result = tools_for_post_edit_navigation_gate(tools, edit_applied=False, enabled=True)

    assert result == tools


def test_post_edit_gate_hides_inventories_but_keeps_repair_actions() -> None:
    tools = [_tool(name) for name in (
        "list_files", "search_source", "read_lines", "list_functions",
        "replace_lines_transactional", "check_syntax", "run_public_tests", "finish",
    )]

    result = tools_for_post_edit_navigation_gate(tools, edit_applied=True, enabled=True)

    names = [tool["function"]["name"] for tool in result]
    assert names == [
        "search_source", "read_lines", "replace_lines_transactional",
        "check_syntax", "run_public_tests", "finish",
    ]


def test_post_edit_gate_can_be_disabled_after_an_edit() -> None:
    tools = [_tool("list_files"), _tool("finish")]

    result = tools_for_post_edit_navigation_gate(tools, edit_applied=True, enabled=False)

    assert result == tools


def test_no_progress_guard_hides_only_repeated_actions_for_one_turn() -> None:
    tools = [_tool(name) for name in (
        "list_files", "search_source", "read_lines", "replace_lines", "finish",
    )]

    result = tools_for_no_progress_guard(
        tools, blocked_tools={"search_source", "list_files"}, enabled=True,
    )

    assert [tool["function"]["name"] for tool in result] == [
        "read_lines", "replace_lines", "finish",
    ]
    assert tools_for_no_progress_guard(
        tools, blocked_tools={"search_source"}, enabled=False,
    ) == tools


def test_duplicate_read_key_uses_returned_range_when_lease_hides_file_hash() -> None:
    result = {
        "status": "read_complete",
        "path": "src/model.py",
        "start_line": 10,
        "end_line": 12,
        "content": "one\ntwo\nthree\n",
        "lease": "opaque-one-time-token",
    }

    key = source_read_progress_key(result, edit_attempt_count=0)

    assert key == (
        "src/model.py", 10, 12,
        hashlib.sha256(result["content"].encode("utf-8")).hexdigest(), 0,
    )
    assert "sha256" not in result


def test_duplicate_read_key_changes_after_source_or_edit_progress() -> None:
    result = {
        "status": "read_complete",
        "path": "src/model.py",
        "start_line": 10,
        "end_line": 12,
        "content": "before\n",
        "lease": "lease-a",
    }
    initial = source_read_progress_key(result, edit_attempt_count=0)
    changed_source = {**result, "content": "after\n", "lease": "lease-b"}

    assert source_read_progress_key(changed_source, edit_attempt_count=0) != initial
    assert source_read_progress_key(result, edit_attempt_count=1) != initial


def test_duplicate_read_key_rejects_incomplete_or_untrusted_ranges() -> None:
    assert source_read_progress_key({"status": "read_rejected"}, edit_attempt_count=0) is None
    assert source_read_progress_key({
        "status": "read_complete", "path": "src/model.py", "start_line": True,
        "end_line": 2, "content": "x",
    }, edit_attempt_count=0) is None
