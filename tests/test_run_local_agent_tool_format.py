from scripts.run_local_agent import (
    generate_turn,
    mlx_resource_stop_reason,
    normalize_tool_call_format,
    public_test_command,
    refresh_public_tests_after_edit,
    tools_for,
)
from vare.agent_edit_format import ParseFailure, parse_submission
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


def test_structured_tools_use_short_hash_checked_edits_and_vare_diff():
    names = {tool["function"]["name"] for tool in tools_for("structured", public_tests=True)}
    assert {"read_file", "search_text", "replace_text", "insert_text", "run_public_tests", "inspect_diff", "finish"} <= names
    assert "apply_patch" not in names
    assert "write_file" not in names
    read_file = next(tool for tool in tools_for("structured") if tool["function"]["name"] == "read_file")
    assert read_file["function"]["parameters"]["properties"]["start_line"]["type"] == "integer"


def test_structured_mode_does_not_accept_free_form_patch_recovery():
    result = parse_submission("--- a/file.py\n+++ b/file.py\n", edit_format="structured")
    assert isinstance(result, ParseFailure)
    assert result.reason == "no_edit"


def test_public_tests_add_archived_transformers_source_to_import_path():
    command = public_test_command()
    assert command[:2] == ["python", "-c"]
    assert "sys.path.insert(0,'/workspace/src')" in command[2]
    assert "runpy.run_path('/workspace/test_public.py')" in command[2]
    assert "name.startswith('test_')" in command[2]


def test_mlx_resource_budget_fails_closed_on_each_frozen_limit():
    good = {
        'available_bytes': 5 * 1024**3,
        'swap_free_bytes': 700 * 1024**2,
        'swap_used_bytes': 6 * 1024**3,
    }
    limits = {
        'minimum_available_bytes': 4 * 1024**3,
        'minimum_free_swap_bytes': 512 * 1024**2,
        'maximum_swap_used_bytes': 7 * 1024**3,
    }
    assert mlx_resource_stop_reason(good, **limits) is None
    assert mlx_resource_stop_reason({}, **limits) == 'resource_probe_unavailable'
    assert mlx_resource_stop_reason({**good, 'swap_free_bytes': 100}, **limits) == 'minimum_free_swap_breached'
    assert mlx_resource_stop_reason({**good, 'swap_used_bytes': 8 * 1024**3}, **limits) == 'maximum_swap_used_breached'
    assert mlx_resource_stop_reason(good, **limits, peak_bytes=17 * 1024**3,
                                    maximum_peak_bytes=16 * 1024**3) == 'maximum_mlx_peak_memory_breached'


def test_mlx_turn_resets_peak_and_records_peak_in_bytes(monkeypatch):
    import sys
    import types
    from argparse import Namespace
    from types import SimpleNamespace

    calls = []
    mx = types.SimpleNamespace(
        reset_peak_memory=lambda: calls.append('reset'),
        get_peak_memory=lambda: calls.append('peak') or 123456,
    )
    mlx_package = types.ModuleType('mlx')
    mlx_package.__path__ = []
    mlx_package.core = mx
    generated = SimpleNamespace(
        text='answer', generation_tokens=2, prompt_tps=3.0, generation_tps=4.0,
    )
    mlx_lm = types.ModuleType('mlx_lm')
    mlx_lm.stream_generate = lambda *args, **kwargs: iter([generated])
    sample_utils = types.ModuleType('mlx_lm.sample_utils')
    sample_utils.make_sampler = lambda **kwargs: 'sampler'
    monkeypatch.setitem(sys.modules, 'mlx', mlx_package)
    monkeypatch.setitem(sys.modules, 'mlx.core', mx)
    monkeypatch.setitem(sys.modules, 'mlx_lm', mlx_lm)
    monkeypatch.setitem(sys.modules, 'mlx_lm.sample_utils', sample_utils)

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return 'prompt'

        def encode(self, text, **kwargs):
            return [1, 2, 3]

    args = Namespace(
        max_context_tokens=100, do_sample=False, temperature=0.0, top_p=1.0, top_k=0,
    )
    result = generate_turn('mlx', object(), FakeTokenizer(), [], [], limit=5, args=args)
    assert result[0:3] == ('answer', 2, 3)
    assert result[3]['mlx_peak_memory_bytes'] == 123456
    assert calls == ['reset', 'peak']


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

    def fail_fresh_test(
        workspace, image, test_hash, memory_mb, *, expected_workspace_files,
        expected_file_hashes=None, allowed_modified_files=None,
    ):
        calls.append((
            workspace, image, test_hash, memory_mb, expected_workspace_files,
            expected_file_hashes, allowed_modified_files,
        ))
        return {"passed": False, "returncode": 1}

    monkeypatch.setattr("scripts.run_local_agent.run_public_tests", fail_fresh_test)
    stale_result = {"passed": True, "returncode": 0}
    refreshed = refresh_public_tests_after_edit(
        {"applied": True}, tmp_path, "pinned-image", "test-hash", 1536,
        expected_workspace_files={"src/module.py", "test_public.py"},
    )
    assert refreshed == {"passed": False, "returncode": 1}
    assert refreshed is not stale_result
    assert calls == [(
        tmp_path, "pinned-image", "test-hash", 1536,
        {"src/module.py", "test_public.py"}, None, None,
    )]


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
