from scripts.run_local_agent import (
    normalize_tool_call_format, read_file_range, refresh_public_tests_after_edit,
    replace_text, tools_for,
)
from scripts.run_separated_coding_eval import (
    _attempted_task_run_count,
    _planned_task_run_count,
    verified_task_success,
)


def test_normalizes_a_fenced_qwen_json_tool_call_without_changing_raw_text():
    raw = '```json\n{"name":"list_files","arguments":{}}\n```'

    normalized, recovered = normalize_tool_call_format(raw, {"list_files", "read_file"})

    assert recovered is True
    assert normalized == '<tool_call>{"name":"list_files","arguments":{}}</tool_call>'
    assert raw.startswith("```json")


def test_normalizes_a_bare_qwen_json_tool_call():
    raw = '{"name":"read_file","arguments":{"path":"grad_accum.py"}}'
    assert normalize_tool_call_format(raw, {"read_file"}) == (
        '<tool_call>{"name":"read_file","arguments":{"path":"grad_accum.py"}}</tool_call>',
        True,
    )


def test_normalizes_a_chatml_wrapped_json_tool_call():
    raw = '<|im_start|>\n{"name":"write_file","arguments":{"path":"x.py","content":"pass\\n"}}\n<|im_end|>'
    assert normalize_tool_call_format(raw, {"write_file"}) == (
        '<tool_call>{"name":"write_file","arguments":{"path":"x.py","content":"pass\\n"}}</tool_call>',
        True,
    )


def test_normalizes_json_before_a_trailing_chatml_generation_marker():
    raw = '<|im_start|>\n{"name":"list_files","arguments":{}}\n<|im_start|>\n'
    assert normalize_tool_call_format(raw, {"list_files"}) == (
        '<tool_call>{"name":"list_files","arguments":{}}</tool_call>', True,
    )


def test_does_not_promote_unknown_or_malformed_fenced_json_to_tool_calls():
    for raw in (
        '```json\n{"name":"run_shell","arguments":{}}\n```',
        '```json\n{"name":"read_file","arguments":[]}\n```',
        '```python\nprint("not a tool")\n```',
        '{"name":"run_shell","arguments":{}}',
    ):
        assert normalize_tool_call_format(raw, {"read_file"}) == (raw, False)


def test_preserves_native_tool_call_format():
    raw = '<tool_call>{"name":"finish","arguments":{"summary":"done"}}</tool_call>'
    assert normalize_tool_call_format(raw, {"finish"}) == (raw, False)


def test_structured_interface_exposes_bounded_reads_and_unique_replacements():
    names = {tool["function"]["name"] for tool in tools_for("structured_replace", public_tests=True)}
    assert {"read_file_range", "replace_text", "run_public_tests", "finish"} <= names
    assert "apply_patch" not in names
    assert "write_file" not in names


def test_bounded_read_rejects_oversized_or_disallowed_ranges(tmp_path):
    source = tmp_path / "src.py"
    source.write_text("one\ntwo\nthree\n", encoding="utf-8")
    assert read_file_range(tmp_path, "src.py", 2, 3, {"src.py"})["content"] == "two\nthree"
    assert "error" in read_file_range(tmp_path, "src.py", 1, 121, {"src.py"})
    assert "error" in read_file_range(tmp_path, "../src.py", 1, 1, {"src.py"})


def test_structured_replacement_requires_one_exact_match_and_allowlisted_path(tmp_path):
    source = tmp_path / "src.py"
    source.write_text("value = 1\n", encoding="utf-8")
    assert replace_text(tmp_path, "src.py", "value = 1", "value = 2", {"src.py"})["applied"]
    assert source.read_text(encoding="utf-8") == "value = 2\n"
    source.write_text("x = 1\nx = 1\n", encoding="utf-8")
    ambiguous = replace_text(tmp_path, "src.py", "x = 1", "x = 2", {"src.py"})
    assert ambiguous["applied"] is False
    assert ambiguous["matches"] == 2
    assert replace_text(tmp_path, "../outside.py", "x", "y", {"src.py"})["applied"] is False


def test_verified_success_requires_public_test_and_independent_hidden_grade():
    agent = {
        "source_hashes_before": {"x.py": "before"},
        "source_hashes_after": {"x.py": "after"},
        "public_test_result": {"passed": True},
    }
    assert verified_task_success(agent, {"success": True}) is True
    assert verified_task_success({**agent, "public_test_result": None}, {"success": True}) is False
    assert verified_task_success(agent, {"success": False}) is False


def test_applied_recovered_edit_requires_a_fresh_public_test(monkeypatch, tmp_path):
    calls = []

    def fail_fresh_test(workspace, image, test_hash):
        calls.append((workspace, image, test_hash))
        return {"passed": False, "returncode": 1}

    monkeypatch.setattr("scripts.run_local_agent.run_public_tests", fail_fresh_test)
    stale_result = {"passed": True, "returncode": 0}
    refreshed = refresh_public_tests_after_edit(
        {"applied": True}, tmp_path, "pinned-image", "test-hash",
    )
    assert refreshed == {"passed": False, "returncode": 1}
    assert refreshed is not stale_result
    assert calls == [(tmp_path, "pinned-image", "test-hash")]


def test_attempted_task_run_count_includes_every_arm_and_secondary_model():
    arm_reports = [
        {"name": "greedy", "rows": [{}, {}]},
        {"name": "sampled", "rows": [{}, {}]},
    ]
    secondary = {"rows": [{}], "coder_1_5b_rows": [{}, {}]}
    assert _attempted_task_run_count(arm_reports, secondary) == 7


def test_planned_task_run_count_matches_admitted_secondary_models():
    protocol = {"models": {
        "secondary_if_feasible": {"seeds": [201]},
        "not_run_unless": {"seeds": [301, 302]},
    }}
    arms = [{"seeds": [101, 102]}]
    assert _planned_task_run_count(
        protocol, arms, 3, {"coder_0_5b": "eligible", "coder_1_5b": "ran"},
    ) == 15
    assert _planned_task_run_count(
        protocol, arms, 3, {"coder_0_5b": "skipped", "coder_1_5b": "skipped"},
    ) == 6
