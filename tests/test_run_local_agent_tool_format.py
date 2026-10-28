import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_local_agent
from scripts.run_local_agent import (
    DurableEventJournal, capture_rendered_prompt, load_localization_record, main, normalize_tool_call_format,
    parse_tool_output_v1, pinned_public_contract_grader_sources, read_file_range,
    python_syntax_failures, refresh_public_tests_after_edit, replace_text, run_public_contract_tests,
    run_with_public_test_budget, tools_for, tools_for_source_read_gate,
)
from scripts.run_local_agent import supports_localization_context, transactional_edit_guidance
from vare.agent_edit_v2 import SourceEditLeaseStoreV1
from vare.container_isolation import DockerIsolationRunner, IsolationResult
from scripts.calibrate_agent_edit_interface_v1 import FIXTURES, _validate_fixture, tool_output
from scripts.run_separated_coding_eval import (
    _attempted_task_run_count,
    _planned_task_run_count,
    verified_task_success,
)

PINNED_TEST_IMAGE = (
    "python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258"
)


def test_public_contract_grader_registration_is_task_and_hash_pinned(monkeypatch):
    grader = run_local_agent.ROOT / 'scripts/grade_mlsys_coding_v2_candidate_v2.py'
    sources = pinned_public_contract_grader_sources(grader, 'label-smooth')
    assert sources['scripts/grade_mlsys_coding_v2_candidate_v2.py'] == (
        '9bdcdf7c2550d542de4ac520bbc3a0b72c3bc6ca53021ce6bda16693e841df05')
    assert 'scripts/grade_mlsys_coding_v2_candidate_v1.py' in sources

    with pytest.raises(ValueError, match='path or task is not pinned'):
        pinned_public_contract_grader_sources(grader, 'unregistered-task')

    actual_sha = run_local_agent.sha256

    def drifted_sha(path: Path) -> str:
        if path.resolve() == grader.resolve():
            return '0' * 64
        return actual_sha(path)

    monkeypatch.setattr(run_local_agent, 'sha256', drifted_sha)
    with pytest.raises(ValueError, match='grader hash mismatch'):
        pinned_public_contract_grader_sources(grader, 'label-smooth')


def test_source_read_gate_hides_edits_until_a_fresh_lease_is_read():
    tools = tools_for(
        'structured_replace', public_tests=True,
        tool_interface='lease_bound_transactional_navigation_v1',
    )
    gated = tools_for_source_read_gate(tools, lease_read_succeeded=False, enabled=True)
    assert {tool['function']['name'] for tool in gated} == {
        'list_files', 'read_lines', 'search_source',
    }
    assert {tool['function']['name'] for tool in tools_for_source_read_gate(
        tools, lease_read_succeeded=True, enabled=True,
    )} == {tool['function']['name'] for tool in tools}
    assert tools_for_source_read_gate(
        tools, lease_read_succeeded=False, enabled=False,
    ) is tools


def test_legacy_public_contract_grader_registration_remains_available():
    grader = run_local_agent.ROOT / 'scripts/grade_corrected_template_candidate_v1.py'
    sources = pinned_public_contract_grader_sources(grader, 'gqa-repeat')
    assert 'scripts/grade_corrected_template_candidate_v1.py' in sources
    assert 'scripts/grade_vare_development_candidate_v1.py' in sources


def test_main_accepts_registered_authored_grader_before_model_path_check(
        tmp_path, monkeypatch, capsys):
    grader = run_local_agent.ROOT / 'scripts/grade_mlsys_coding_v2_candidate_v2.py'
    monkeypatch.setattr(sys, 'argv', [
        'run_local_agent.py', '--workspace', str(tmp_path / 'workspace'),
        '--model-path', str(tmp_path / 'missing-model'), '--task-root', str(tmp_path / 'task'),
        '--output', str(tmp_path / 'agent.json'), '--instruction', 'inspect and repair',
        '--seed', '101', '--public-contract-grader', str(grader),
        '--public-contract-task', 'label-smooth',
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert 'local model snapshot directory does not exist' in capsys.readouterr().err


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


def test_public_contract_runner_accepts_only_strict_grader_success(tmp_path, monkeypatch):
    from types import SimpleNamespace

    source = tmp_path / "candidate.py"
    source.write_text("pass\n")
    monkeypatch.setattr("scripts.run_local_agent.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='{"success":true}', stderr=""))
    assert run_public_contract_tests(tmp_path / "grader.py", "task", source)["passed"] is True
    monkeypatch.setattr("scripts.run_local_agent.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='not-json', stderr=""))
    assert run_public_contract_tests(tmp_path / "grader.py", "task", source)["passed"] is False
    monkeypatch.setattr("scripts.run_local_agent.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout='{"success":true}', stderr="failure"))
    assert run_public_contract_tests(tmp_path / "grader.py", "task", source)["passed"] is False


def test_public_test_budget_is_shared_and_does_not_execute_after_exhaustion():
    calls = []

    def runner():
        calls.append("ran")
        return {"passed": True}

    result, used = run_with_public_test_budget(0, 1, runner)
    assert result == {"passed": True} and used == 1
    result, used = run_with_public_test_budget(used, 1, runner)
    assert result["error"] == "public test run budget exhausted"
    assert used == 1 and calls == ["ran"]


def test_rendered_prompt_is_retained_with_exact_utf8_hash():
    import argparse
    import hashlib

    args = argparse.Namespace()
    rendered = "<|im_start|>system\nExact prompt with tools\n"

    assert capture_rendered_prompt(args, rendered) == hashlib.sha256(rendered.encode()).hexdigest()
    assert args._last_rendered_prompt == rendered
    assert args._last_rendered_prompt_sha256 == hashlib.sha256(rendered.encode()).hexdigest()


def test_lease_bound_interface_exposes_only_the_versioned_atomic_edit():
    names = {tool["function"]["name"] for tool in tools_for(
        "structured_replace", tool_interface="lease_bound_lines_v1",
    )}
    assert {"read_lines", "replace_lines", "list_files", "finish"} <= names
    assert "replace_text" not in names
    assert "apply_patch" not in names


def test_transactional_navigation_and_localized_tools_share_the_same_edit_contract():
    autonomous = tools_for(
        'structured_replace', public_tests=True,
        tool_interface='lease_bound_transactional_navigation_v1',
    )
    localized = tools_for(
        'structured_replace', public_tests=True,
        tool_interface='lease_bound_transactional_localized_v1',
    )
    auto_names = {tool['function']['name'] for tool in autonomous}
    localized_names = {tool['function']['name'] for tool in localized}
    assert auto_names == {
        'list_files', 'read_lines', 'search_source', 'replace_lines_transactional',
        'check_syntax', 'run_public_tests', 'finish',
    }
    assert localized_names == {
        'read_lines', 'replace_lines_transactional', 'check_syntax',
        'run_public_tests', 'finish',
    }
    for tools in (autonomous, localized):
        edit = next(tool for tool in tools
                    if tool['function']['name'] == 'replace_lines_transactional')
        assert edit['function']['parameters']['required'] == ['lease', 'text', 'normalization']
        assert edit['function']['parameters']['additionalProperties'] is False


def test_localization_context_can_be_compared_without_changing_tool_capabilities():
    autonomous = tools_for(
        'structured_replace', public_tests=True,
        tool_interface='lease_bound_transactional_navigation_v1',
    )
    localized_context = tools_for(
        'structured_replace', public_tests=True,
        tool_interface='lease_bound_transactional_navigation_v1',
    )

    assert autonomous == localized_context
    names = {tool['function']['name'] for tool in localized_context}
    assert names == {
        'list_files', 'read_lines', 'search_source', 'replace_lines_transactional',
        'check_syntax', 'run_public_tests', 'finish',
    }
    assert supports_localization_context('lease_bound_transactional_navigation_v1')
    assert supports_localization_context('lease_bound_transactional_localized_v1')
    assert 'search tool remains available' in transactional_edit_guidance(
        localized=True, search_available=True,
    )
    assert 'search tool remains available' not in transactional_edit_guidance(
        localized=True, search_available=False,
    )


def test_transactional_navigation_guidance_v2_is_explicit_and_keeps_edit_contract():
    v1 = transactional_edit_guidance(localized=False, search_available=True)
    v2 = transactional_edit_guidance(
        localized=False, search_available=True, navigation_guidance='v2',
    )

    assert v1.startswith('Search candidate-visible source, then inspect a narrow range using read_lines.')
    assert 'one short, distinctive, unquoted identifier or term' in v2
    assert 'every query term must occur on one source line' in v2
    assert 'retry once with a shorter identifier' in v2
    assert 'Never repeat an identical query after a duplicate_query result' in v2
    assert 'Do not read the beginning of the file when a search misses' in v2
    assert 'read_lines to obtain a one-time lease' in v2
    assert 'replace_lines_transactional with the complete replacement' in v2
    assert 'required explicit normalization choice' in v2
    assert 'syntax-invalid proposal is rejected without changing the candidate' in v2
    assert 'Semantic code is never inferred or repaired by the controller.' in v2

    legacy_tools = tools_for(
        'structured_replace', public_tests=True,
        tool_interface='lease_bound_transactional_navigation_v1',
    )
    assert {tool['function']['name'] for tool in legacy_tools} == {
        'list_files', 'read_lines', 'search_source', 'replace_lines_transactional',
        'check_syntax', 'run_public_tests', 'finish',
    }


def test_transactional_guidance_matches_available_verification_tool():
    tools_without_public_tests = tools_for(
        'structured_replace',
        public_tests=False,
        tool_interface='lease_bound_transactional_localized_v1',
    )
    without_public_tests = transactional_edit_guidance(
        localized=False,
        search_available=True,
        navigation_guidance='v2',
        public_tests_available=False,
    )
    with_public_tests = transactional_edit_guidance(
        localized=False,
        search_available=True,
        navigation_guidance='v2',
        public_tests_available=True,
    )

    assert 'call check_syntax on the edited path' in without_public_tests
    assert 'run_public_tests' not in without_public_tests
    assert 'call run_public_tests' in with_public_tests
    exposed_names = {tool['function']['name'] for tool in tools_without_public_tests}
    assert 'check_syntax' in exposed_names
    assert 'run_public_tests' not in exposed_names


def test_function_scoped_interface_uses_closed_schemas_and_only_function_edits():
    tools = tools_for(
        'structured_replace', public_tests=True,
        tool_interface='lease_bound_function_edit_v1',
    )
    by_name = {tool['function']['name']: tool['function']['parameters'] for tool in tools}
    assert set(by_name) == {
        'list_functions', 'read_function', 'replace_function_body_transactional',
        'check_syntax', 'run_public_tests', 'finish',
    }
    assert by_name['list_functions']['required'] == ['path']
    assert by_name['read_function']['required'] == ['path', 'qualified_name']
    assert by_name['replace_function_body_transactional']['required'] == [
        'lease', 'body', 'normalization',
    ]
    assert by_name['replace_function_body_transactional']['properties']['normalization']['enum'] == [
        'none', 'strip_single_python_fence', 'append_final_newline',
        'strip_single_python_fence_append_final_newline',
    ]
    assert all(schema['additionalProperties'] is False for schema in by_name.values())


def test_function_scoped_parser_rejects_unsupported_arguments_without_repair():
    tools = tools_for(
        'structured_replace', tool_interface='lease_bound_function_edit_v1',
    )
    valid = '<tool_call>{"name":"list_functions","arguments":{"path":"module.py"}}</tool_call>'
    invalid = '<tool_call>{"name":"list_functions","arguments":{"path":"module.py","pattern":"*"}}</tool_call>'

    assert parse_tool_output_v1(valid, tools)[0] == 'complete_tool_call'
    status, call = parse_tool_output_v1(invalid, tools)
    assert status == 'invalid_json_or_schema'
    assert call['arguments']['pattern'] == '*'


def test_certified_single_line_function_interface_has_the_same_closed_actions():
    tools = tools_for(
        'structured_replace', public_tests=True,
        tool_interface='lease_bound_single_line_function_edit_v1',
    )
    names = {tool['function']['name'] for tool in tools}
    assert names == {
        'list_functions', 'read_function', 'replace_function_body_transactional',
        'check_syntax', 'run_public_tests', 'finish',
    }
    edit = next(tool for tool in tools
                if tool['function']['name'] == 'replace_function_body_transactional')
    assert edit['function']['parameters']['additionalProperties'] is False


def test_localization_record_is_bound_to_exact_candidate_source(tmp_path):
    from scripts.run_local_agent import sha256

    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    source = workspace / 'module.py'
    source.write_text('def call():\n    return 1\n', encoding='utf-8')
    record_path = tmp_path / 'localization.json'
    record = {
        'algorithm': 'task_prompt_lexical_windows_v1',
        'path': 'module.py', 'start_line': 1, 'end_line': 2,
        'source_sha256': sha256(source), 'content': 'def call():\n    return 1\n',
    }
    record_path.write_text(json.dumps(record), encoding='utf-8')

    assert load_localization_record(record_path, workspace, {'module.py'}) == record
    source.write_text('def call():\n    return 2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='does not match'):
        load_localization_record(record_path, workspace, {'module.py'})


def test_versioned_parser_distinguishes_complete_truncated_invalid_and_policy_rejected():
    tools = tools_for("structured_replace", tool_interface="lease_bound_lines_v1")
    complete = '<tool_call>{"name":"list_files","arguments":{}}</tool_call>'
    assert parse_tool_output_v1(complete, tools)[0] == "complete_tool_call"
    truncated = '<tool_call>{"name":"replace_lines","arguments":{"lease":"12ab34cd56ef","text":"'
    assert parse_tool_output_v1(truncated, tools) == ("truncated_tool_call", None)
    invalid = '<tool_call>{"name":"list_files","arguments":[]}</tool_call>'
    assert parse_tool_output_v1(invalid, tools)[0] == "invalid_json_or_schema"
    denied = '<tool_call>{"name":"write_file","arguments":{"path":"src/value.py"}}</tool_call>'
    assert parse_tool_output_v1(denied, tools)[0] == "valid_call_rejected_by_policy"
    assert parse_tool_output_v1("I will inspect the code first.", tools) == ("no_tool_call", None)


def test_versioned_parser_rejects_bad_schema_without_repairing_it():
    tools = tools_for("structured_replace", tool_interface="lease_bound_lines_v1")
    bad_type = (
        '<tool_call>{"name":"read_lines","arguments":'
        '{"path":"src/value.py","start":true,"end":1}}</tool_call>'
    )
    extra = '<tool_call>{"name":"list_files","arguments":{"surprise":1}}</tool_call>'
    assert parse_tool_output_v1(bad_type, tools)[0] == "invalid_json_or_schema"
    assert parse_tool_output_v1(extra, tools)[0] == "invalid_json_or_schema"
    # The transport parser must leave the old truncated action byte-for-byte untouched.
    raw = '<tool_call>{"name":"replace_lines","arguments":{"lease":"12ab34cd56ef","text":"partial'
    assert parse_tool_output_v1(raw, tools) == ("truncated_tool_call", None)


def test_corrected_template_calls_use_the_strict_parser_without_repair():
    tools = tools_for("structured_replace", public_tests=True,
                      tool_interface="structured_replace_v1")
    valid = '<tool_call>{"name":"list_files","arguments":{}}</tool_call>'
    doubled_braces = '<tool_call>{{"name":"list_files","arguments":{}}}</tool_call>'

    assert parse_tool_output_v1(valid, tools)[0] == "complete_tool_call"
    assert parse_tool_output_v1(doubled_braces, tools)[0] == "invalid_json_or_schema"


def test_token_calibration_fixtures_have_valid_unique_legacy_preimages():
    for fixture in FIXTURES:
        _validate_fixture(fixture)
    assert len({fixture["name"] for fixture in FIXTURES}) == len(FIXTURES)


def test_token_calibration_rejects_ambiguous_legacy_preimage():
    fixture = dict(FIXTURES[1], legacy_old_text="    return x\n")
    with pytest.raises(ValueError, match="legacy old_text is not unique"):
        _validate_fixture(fixture)


def test_tool_output_uses_compact_complete_envelope():
    assert tool_output("finish", {"summary": "ok"}) == (
        '<tool_call>{"name":"finish","arguments":{"summary":"ok"}}</tool_call>'
    )


def test_durable_event_journal_fsyncs_a_non_overwriting_hash_chain(tmp_path):
    path = tmp_path / "attempt.events.jsonl"
    journal = DurableEventJournal(path)
    journal.append("model_output_completed", turn=1, model_output="<tool_call>{}</tool_call>")
    last_hash = journal.append("run_finished", stop_reason="no_tool_call")
    journal.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["event"]["event_type"] for item in records] == [
        "journal_created", "model_output_completed", "run_finished",
    ]
    previous = "0" * 64
    for record in records:
        assert record["previous_event_sha256"] == previous
        canonical = json.dumps(
            record["event"], ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert record["event_sha256"] == digest
        previous = digest
    assert last_hash == previous
    with pytest.raises(FileExistsError):
        DurableEventJournal(path)


def test_navigation_runs_require_durable_event_journal_before_model_loading(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [
        "run_local_agent.py", "--workspace", str(tmp_path / "workspace"),
        "--model-path", str(tmp_path / "missing-model"), "--task-root", str(tmp_path / "task"),
        "--output", str(tmp_path / "agent.json"), "--instruction", "inspect and repair",
        "--seed", "7", "--tool-interface", "lease_bound_navigation_v1",
        "--edit-format", "structured_replace",
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--event-journal is required for lease_bound_navigation_v1 runs" in capsys.readouterr().err


def test_navigation_runs_accept_a_supplied_event_journal_before_model_loading(
        tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "value.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "config", "user.name", "VARE test"], check=True)
    subprocess.run([
        "git", "-C", str(workspace), "config", "user.email", "vare-test@example.invalid",
    ], check=True)
    subprocess.run(["git", "-C", str(workspace), "add", "src/value.py"], check=True)
    subprocess.run(["git", "-C", str(workspace), "commit", "-qm", "candidate baseline"], check=True)
    task_root = tmp_path / "task"
    task_root.mkdir()
    (task_root / "TASK.md").write_text("Fix the value function.\n", encoding="utf-8")
    (task_root / "task.json").write_text(json.dumps({
        "task_id": "journal-model-load-failure-fixture",
        "source": {"base_revision": "base-sha", "calibration_revision": "fix-sha"},
        "metadata": {"editable_files": ["src/value.py"], "split": "development"},
        "prompt": "Change the value.",
    }), encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_file = model_dir / "weights.safetensors"
    model_file.write_bytes(b"pinned fixture checkpoint\n")
    tokenizer_config = model_dir / "tokenizer_config.json"
    tokenizer_config.write_text(json.dumps({"chat_template": "fixture-template"}), encoding="utf-8")
    journal_path = tmp_path / "events.jsonl"

    fake_mlx = ModuleType("mlx")
    fake_mlx.__path__ = []
    fake_mlx.__version__ = "fixture"
    fake_core = ModuleType("mlx.core")
    selected_devices = []
    fake_core.cpu = "fixture-cpu"
    fake_core.gpu = "fixture-gpu"
    fake_core.set_default_device = selected_devices.append
    fake_core.random = SimpleNamespace(seed=lambda seed: None)
    fake_mlx.core = fake_core
    fake_mlx_lm = ModuleType("mlx_lm")

    def fail_model_load(path):
        raise OSError("fixture model load failed")

    fake_mlx_lm.load = fail_model_load
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setattr("scripts.run_local_agent.platform.system", lambda: "Darwin")
    monkeypatch.setattr("scripts.run_local_agent.platform.machine", lambda: "arm64")
    monkeypatch.setattr("scripts.run_local_agent.run_git", lambda *args: "a" * 40)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr("sys.argv", [
        "run_local_agent.py", "--workspace", str(workspace),
        "--model-path", str(model_dir), "--task-root", str(task_root),
        "--diff-base", "HEAD",
        "--output", str(tmp_path / "agent.json"), "--event-journal", str(journal_path),
        "--instruction", "inspect and repair", "--seed", "7",
        "--backend", "mlx", "--model-repository", "fixture/model",
        "--tool-interface", "lease_bound_navigation_v1", "--edit-format", "structured_replace",
    ])
    with pytest.raises(OSError, match="fixture model load failed"):
        main()
    assert selected_devices == [fake_core.gpu]
    records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert [item["event"]["event_type"] for item in records] == [
        "journal_created", "invocation_started", "run_context",
        "model_load_started", "model_load_failed",
    ]
    invocation = records[1]["event"]
    assert invocation["task_id"] == "journal-model-load-failure-fixture"
    assert invocation["task_prompt"] == "Fix the value function.\n"
    assert invocation["instruction"] == "inspect and repair"
    assert invocation["model_repository"] == "fixture/model"
    model_files = {
        model_file.name: hashlib.sha256(model_file.read_bytes()).hexdigest(),
        tokenizer_config.name: hashlib.sha256(tokenizer_config.read_bytes()).hexdigest(),
    }
    assert invocation["model_files_sha256"] == model_files
    assert invocation["tokenizer_configuration"]["effective_template_source"] == "snapshot"
    assert invocation["tokenizer_configuration"]["original_config_sha256"] == model_files[tokenizer_config.name]
    assert records[3]["event"]["model_files_sha256"] == model_files
    assert records[-1]["event"]["exception_type"] == "OSError"
    previous = "0" * 64
    for record in records:
        assert record["previous_event_sha256"] == previous
        canonical = json.dumps(
            record["event"], ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert record["event_sha256"] == expected
        previous = expected
    assert not (tmp_path / "agent.json").exists()


def test_mocked_full_edit_reaches_an_independent_fixture_oracle(tmp_path):
    import json
    import runpy

    source = tmp_path / "src" / "value.py"
    source.parent.mkdir()
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    allowed = {"src/value.py"}
    leases = SourceEditLeaseStoreV1(
        tmp_path, allowed, token_factory=lambda: "12ab34cd56ef",
    )
    observed = leases.read_lines("src/value.py", 2, 2)
    request = {
        "name": "replace_lines",
        "arguments": {
            "lease": observed["lease"], "text": "    return 2\n",
        },
    }
    raw = "<tool_call>" + json.dumps(request, separators=(",", ":")) + "</tool_call>"
    status, call = parse_tool_output_v1(
        raw, tools_for("structured_replace", tool_interface="lease_bound_lines_v1"),
    )
    assert status == "complete_tool_call"
    assert call is not None
    assert "sha256" not in observed
    applied = leases.replace_lines(call["arguments"]["lease"], call["arguments"]["text"])
    assert applied["status"] == "edit_applied"
    # The oracle is separately specified and reads the committed candidate file.
    assert runpy.run_path(str(source))["value"]() == 2
    assert leases.audit_records()[0]["expected_sha256"] == applied["before_sha256"]


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


def test_public_test_runner_executes_function_tests_and_fails_on_assertion(
        monkeypatch, tmp_path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    test_file = workspace / "test_public.py"
    test_file.write_text(
        "from pathlib import Path\n"
        "def test_candidate_syntax_is_valid():\n"
        "    import ast\n"
        "    ast.parse(Path('candidate.py').read_text())\n",
        encoding="utf-8",
    )
    (workspace / "candidate.py").write_text("if True:\n    pass\n", encoding="utf-8")
    expected_hash = run_local_agent.sha256(test_file)

    async def execute_test_driver(_self, _limits, *, workspace, command,
                                  allowed_workspace_root):
        assert allowed_workspace_root == tmp_path
        assert command[:2] == ["python", "-c"]
        completed = subprocess.run(
            [sys.executable, "-c", command[2], str(workspace / "test_public.py")],
            cwd=workspace, capture_output=True, text=True, check=False,
        )
        return IsolationResult(completed.returncode, completed.stdout, completed.stderr)

    monkeypatch.setattr(DockerIsolationRunner, "run_candidate", execute_test_driver)
    passed = run_local_agent.run_public_tests(workspace, PINNED_TEST_IMAGE, expected_hash)
    assert passed["passed"] is True
    assert "1 public test(s) passed" in passed["stdout"]

    (workspace / "candidate.py").write_text("if True\n    pass\n", encoding="utf-8")
    failed = run_local_agent.run_public_tests(workspace, PINNED_TEST_IMAGE, expected_hash)
    assert failed["passed"] is False
    assert failed["returncode"] == 1
    assert "SyntaxError" in failed["stderr"]


def test_public_test_runner_rejects_unexecuted_parameterized_tests(
        monkeypatch, tmp_path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    test_file = workspace / "test_public.py"
    test_file.write_text("def test_requires_fixture(fixture):\n    assert fixture\n", encoding="utf-8")

    async def execute_test_driver(_self, _limits, *, workspace, command,
                                  allowed_workspace_root):
        completed = subprocess.run(
            [sys.executable, "-c", command[2], str(workspace / "test_public.py")],
            cwd=workspace, capture_output=True, text=True, check=False,
        )
        return IsolationResult(completed.returncode, completed.stdout, completed.stderr)

    monkeypatch.setattr(DockerIsolationRunner, "run_candidate", execute_test_driver)
    result = run_local_agent.run_public_tests(
        workspace, PINNED_TEST_IMAGE, run_local_agent.sha256(test_file),
    )
    assert result["passed"] is False
    assert "fixtures are unsupported" in result["stderr"]


def test_public_test_runner_executes_module_assertions_and_rejects_noop(
        monkeypatch, tmp_path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    test_file = workspace / "test_public.py"
    test_file.write_text(
        "from pathlib import Path\n"
        "assert Path('candidate.py').read_text() == 'value = 2\\n'\n",
        encoding="utf-8",
    )
    (workspace / "candidate.py").write_text("value = 2\n", encoding="utf-8")

    async def execute_test_driver(_self, _limits, *, workspace, command,
                                  allowed_workspace_root):
        completed = subprocess.run(
            [sys.executable, "-c", command[2], str(workspace / "test_public.py")],
            cwd=workspace, capture_output=True, text=True, check=False,
        )
        return IsolationResult(completed.returncode, completed.stdout, completed.stderr)

    monkeypatch.setattr(DockerIsolationRunner, "run_candidate", execute_test_driver)
    result = run_local_agent.run_public_tests(
        workspace, PINNED_TEST_IMAGE, run_local_agent.sha256(test_file),
    )
    assert result["passed"] is True
    assert "module-level public assertion(s) passed" in result["stdout"]

    test_file.write_text("# no assertions or test functions\n", encoding="utf-8")
    result = run_local_agent.run_public_tests(
        workspace, PINNED_TEST_IMAGE, run_local_agent.sha256(test_file),
    )
    assert result["passed"] is False
    assert "no executable public assertions" in result["stderr"]


def test_finish_syntax_gate_covers_only_allowlisted_python_sources(tmp_path):
    (tmp_path / "good.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("if True\n    pass\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not Python\n", encoding="utf-8")
    assert set(python_syntax_failures(tmp_path, {"good.py", "bad.py", "notes.txt"})) == {"bad.py"}


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


def test_function_edit_v2_tool_schema_is_line_array_and_closed():
    tools = tools_for(
        "structured_replace", public_tests=True,
        tool_interface="lease_bound_function_edit_v2",
    )
    by_name = {tool["function"]["name"]: tool["function"] for tool in tools}
    assert list(by_name) == [
        "list_functions", "read_function", "replace_function_body_transactional",
        "check_syntax", "run_public_tests", "finish",
    ]
    schema = by_name["replace_function_body_transactional"]["parameters"]
    assert schema["required"] == ["lease", "body_lines"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"lease", "body_lines"}
    assert schema["properties"]["body_lines"] == {
        "type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 120,
    }


def test_function_edit_v2_parser_enforces_array_item_and_length_constraints():
    tools = tools_for("structured_replace", tool_interface="lease_bound_function_edit_v2")

    def parse(body_lines):
        payload = {
            "name": "replace_function_body_transactional",
            "arguments": {"lease": "lease-1", "body_lines": body_lines},
        }
        return parse_tool_output_v1(f"<tool_call>{json.dumps(payload)}</tool_call>", tools)[0]

    assert parse(["    return value"]) == "complete_tool_call"
    assert parse([]) == "invalid_json_or_schema"
    assert parse([1]) == "invalid_json_or_schema"
    assert parse(["line"] * 121) == "invalid_json_or_schema"


def test_localized_function_editor_omits_navigation_but_keeps_leased_array_edit():
    autonomous = tools_for(
        "structured_replace", public_tests=True,
        tool_interface="lease_bound_function_edit_v2",
    )
    localized = tools_for(
        "structured_replace", public_tests=True,
        tool_interface="lease_bound_function_edit_localized_v1",
    )
    auto_names = {tool["function"]["name"] for tool in autonomous}
    localized_names = {tool["function"]["name"] for tool in localized}
    assert auto_names == {
        "list_functions", "read_function", "replace_function_body_transactional",
        "check_syntax", "run_public_tests", "finish",
    }
    assert localized_names == {
        "read_function", "replace_function_body_transactional",
        "check_syntax", "run_public_tests", "finish",
    }
    auto_edit = next(tool for tool in autonomous
                     if tool["function"]["name"] == "replace_function_body_transactional")
    localized_edit = next(tool for tool in localized
                          if tool["function"]["name"] == "replace_function_body_transactional")
    assert localized_edit == auto_edit
    rejected = '<tool_call>{"name":"list_functions","arguments":{"path":"module.py"}}</tool_call>'
    assert parse_tool_output_v1(rejected, localized)[0] == "valid_call_rejected_by_policy"
