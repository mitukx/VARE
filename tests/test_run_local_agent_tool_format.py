import hashlib
import json

import pytest

from scripts.run_local_agent import (
    DurableEventJournal, main, normalize_tool_call_format, parse_tool_output_v1, read_file_range,
    refresh_public_tests_after_edit, replace_text, run_public_contract_tests,
    run_with_public_test_budget, tools_for,
)
from vare.agent_edit_v2 import SourceEditLeaseStoreV1
from scripts.calibrate_agent_edit_interface_v1 import FIXTURES, _validate_fixture, tool_output
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


def test_lease_bound_interface_exposes_only_the_versioned_atomic_edit():
    names = {tool["function"]["name"] for tool in tools_for(
        "structured_replace", tool_interface="lease_bound_lines_v1",
    )}
    assert {"read_lines", "replace_lines", "list_files", "finish"} <= names
    assert "replace_text" not in names
    assert "apply_patch" not in names


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
    model_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
    assert invocation["model_files_sha256"] == {model_file.name: model_hash}
    assert records[3]["event"]["model_files_sha256"] == {model_file.name: model_hash}
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
