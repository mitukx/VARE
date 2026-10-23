from scripts.run_local_agent import (
    normalize_tool_call_format,
    parse_tool_call_blocks,
    refresh_public_tests_after_edit,
    tools_for,
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


def test_recovers_exact_duplicate_qwen_wrapper_and_parses_nested_arguments():
    raw = '<tool_call>\n{{"name": "read_file", "arguments": {"path": "/mlx_lm/sample_utils.py"}}\n}}\n</tool_call>'
    tools = tools_for("unified_diff")
    names = {item["function"]["name"] for item in tools}

    normalized, recovered = normalize_tool_call_format(raw, names)
    blocks, parsed, failures = parse_tool_call_blocks(normalized, tools)

    assert recovered is True
    assert len(blocks) == 1
    assert failures == []
    assert [(name, arguments) for _, name, arguments in parsed] == [
        ("read_file", {"path": "/mlx_lm/sample_utils.py"}),
    ]


def test_parses_nested_json_in_native_tool_call_blocks():
    raw = '<tool_call>{"name":"read_file","arguments":{"path":"src/module.py"}}</tool_call>'
    blocks, parsed, failures = parse_tool_call_blocks(raw, tools_for("unified_diff"))
    assert len(blocks) == 1
    assert [(name, arguments) for _, name, arguments in parsed] == [
        ("read_file", {"path": "src/module.py"}),
    ]
    assert failures == []


def test_does_not_recover_malformed_patch_or_unbalanced_suffix():
    tools = tools_for("unified_diff")
    names = {item["function"]["name"] for item in tools}
    malformed_patch = (
        '<tool_call>\n{{"name": "apply_patch", "arguments": '
        '{"patch": "diff --git a/x.py b/x.py\\n" +\n"+++ b/x.py\\n"}}\n}}'
        '</tool_call>'
    )
    ambiguous = '<tool_call>{{"name":"list_files","arguments":{}}} extra }}</tool_call>'

    assert normalize_tool_call_format(malformed_patch, names) == (malformed_patch, False)
    assert normalize_tool_call_format(ambiguous, names) == (ambiguous, False)


def test_rejects_tool_arguments_outside_the_declared_schema():
    tools = tools_for("unified_diff", public_tests=True)
    blocks, parsed, failures = parse_tool_call_blocks(
        '<tool_call>{"name":"run_public_tests","arguments":{"timeout":600}}</tool_call>',
        tools,
    )
    assert len(blocks) == 1
    assert parsed == []
    assert len(failures) == 1
    assert "declared schema" in failures[0][1]


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
