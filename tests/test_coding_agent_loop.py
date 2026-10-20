"""Harness regressions for the MLSys coding-agent loop.

These tests use a scripted generator. A passing hidden grade here is a
known-fix control, not a model result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vare.coding_agent_loop import (
    CandidateWorker,
    DurableEpisode,
    Generation,
    InjectedFault,
    WorkspaceTools,
    grade_separated,
    parse_tool_calls,
    run_episode,
    run_jail,
    system_prompt,
)
from vare.environments.runner import Workspace
from vare.environments.spec import EngineeringTaskSpec
from vare.trajectory_store import PolicyCheckpoint, TrajectoryStore
from vare.types import Task

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "benchmarks" / "mlsys_coding_v1" / "tasks" / "grad_accum_tail"
MALFORMED = (
    '<tool_call>\n{"name": "apply_patch", "arguments": {"patch": '
    '"def average_microbatch_grads(self, batch_size=1):\\n    pass"}}\n</tool_call>'
)
FIX = (
    '<tool_call>\n{"name": "replace_text", "arguments": {'
    '"path": "grad_accum.py", '
    '"old": "    scale = float(accumulation_steps)", '
    '"new": "    scale = float(len(microbatches))"}}\n</tool_call>'
)
FINISH = '<tool_call>\n{"name": "finish", "arguments": {"summary": "mean of the supplied rows"}}\n</tool_call>'
XML = """<tool_call>
<function=read_file>
<parameter=path>
grad_accum.py
</parameter>
</function>
</tool_call>"""


def _scripted(outputs: list[str]):
    def generate(prompt: str, limit: int) -> Generation:
        index = prompt.count("<tool_response>")
        return Generation(outputs[index], 8)

    return generate


def _open_task():
    spec = EngineeringTaskSpec.load(TASK_DIR / "task.json")
    workspace = Workspace(spec)
    tools = WorkspaceTools(workspace.path, set(spec.metadata["editable_files"]))
    return spec, workspace, tools


def _run(tools, outputs, **kwargs):
    spec = EngineeringTaskSpec.load(TASK_DIR / "task.json")
    return run_episode(
        tools=tools,
        task_prompt=spec.prompt,
        generate=_scripted(outputs),
        max_turns=6,
        max_new_tokens=64,
        max_total_new_tokens=256,
        max_no_tool_retries=2,
        wall_seconds=60,
        close_think=True,
        **kwargs,
    )


def test_jailed_rpc_preserves_tuples():
    root = Path("/tmp/vare-tuple-subject")
    if root.exists():
        for child in root.glob("*"):
            child.unlink()
    else:
        root.mkdir()
    (root / "pairs.py").write_text(
        "def identify(items):\n"
        "    return [type(item).__name__ for item in items]\n",
        encoding="utf-8",
    )
    worker = CandidateWorker(root)
    try:
        result = worker.call("pairs", "identify", [[(1, 2), [3, 4]]], {})
    finally:
        worker.close()
    assert result["ok"] is True
    assert result["value"] == ["tuple", "list"]


PROBE_BARE = (
    '{"name": "replace_text", "arguments": {"path": "sample.py", '
    '"old": "VALUE = 0", "new": "VALUE = 1"}}'
)


def test_recovery_protocol_loads_without_a_pack_block():
    import importlib.util

    path = ROOT / "scripts" / "run_mlsys_coding_agent_e2e.py"
    spec = importlib.util.spec_from_file_location("mlsys_e2e_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    protocol = module._load_protocol(ROOT / "protocols" / "mlsys_coding_agent_recovery_v1.json")
    assert protocol["kind"] == "recovery"
    assert protocol["task"] == "mlsys-v1-grad-accum-tail"


def test_parser_accepts_bare_tool_json_from_the_format_probe():
    calls = parse_tool_calls(PROBE_BARE)
    assert calls == [{
        "name": "replace_text",
        "arguments": {"path": "sample.py", "old": "VALUE = 0", "new": "VALUE = 1"},
    }]
    fenced = "```json\n" + PROBE_BARE + "\n```"
    assert parse_tool_calls(fenced)[0]["name"] == "replace_text"
    assert parse_tool_calls("example only\n" + PROBE_BARE + "\nthanks") == []


def test_parser_accepts_recorded_json_and_qwen_xml():
    calls = parse_tool_calls(MALFORMED)
    assert calls[0]["name"] == "apply_patch"
    assert "pass" in calls[0]["arguments"]["patch"]
    xml = parse_tool_calls(XML)
    assert xml == [{"name": "read_file", "arguments": {"path": "grad_accum.py"}}]


def test_malformed_patch_continues_and_exact_replace_passes_hidden_grade():
    spec, workspace, tools = _open_task()
    try:
        result = _run(tools, [MALFORMED, FIX, FINISH])
        assert result.stop_reason == "agent_finished"
        assert result.transcript[0]["tool_calls"][0]["result"]["error"] == "patch is not a unified diff"
        assert "replace_text" in result.transcript[0]["tool_calls"][0]["result"]["hint"]
        assert "short window was scaled" not in system_prompt(spec.prompt)
        grade = grade_separated(task_dir=TASK_DIR, workspace=workspace.path, timeout_s=30)
        assert grade["passed"] is True
        assert grade["provenance"]["oracle_mounted"] is False
    finally:
        workspace.close()


def test_unpatched_subject_fails_separated_grade_and_oracle_bytes_pass():
    _spec, workspace, _tools = _open_task()
    try:
        failed = grade_separated(task_dir=TASK_DIR, workspace=workspace.path, timeout_s=30)
        assert failed["passed"] is False
        oracle = TASK_DIR / "evaluator" / "oracle" / "grad_accum.py"
        (workspace.path / "grad_accum.py").write_bytes(oracle.read_bytes())
        passed = grade_separated(task_dir=TASK_DIR, workspace=workspace.path, timeout_s=30)
        assert passed["passed"] is True
    finally:
        workspace.close()


def test_candidate_jail_cannot_read_host_canary_or_open_a_socket():
    secret = Path("/tmp/vare-canary-secret")
    secret.write_text("CANARY-VALUE-do-not-leak", encoding="utf-8")
    root = Path("/tmp/vare-jail-subject")
    if root.exists():
        for child in root.glob("*"):
            child.unlink()
    else:
        root.mkdir()
    (root / "reveal.py").write_text(
        "def reveal(path):\n"
        "    try:\n"
        "        return open(path, encoding='utf-8').read()\n"
        "    except Exception as exc:\n"
        "        return type(exc).__name__\n"
        "def ping():\n"
        "    import socket\n"
        "    sock = socket.socket()\n"
        "    sock.settimeout(2)\n"
        "    try:\n"
        "        sock.connect(('1.1.1.1', 443))\n"
        "        return 'open'\n"
        "    except Exception as exc:\n"
        "        return type(exc).__name__\n",
        encoding="utf-8",
    )
    hidden = ROOT / "benchmarks/mlsys_coding_v1/tasks/grad_accum_tail/evaluator/test_hidden.py"
    worker = CandidateWorker(root)
    try:
        host_file = worker.call("reveal", "reveal", [str(secret)], {})
        hidden_file = worker.call("reveal", "reveal", [str(hidden)], {})
        network = worker.call("reveal", "ping", [], {})
    finally:
        worker.close()
    assert host_file["ok"] is True
    assert "CANARY-VALUE" not in str(host_file["value"])
    assert hidden_file["ok"] is True
    assert "short window was scaled" not in str(hidden_file["value"])
    assert network["value"] != "open"
    jailed = run_jail(
        mounts={"workspace": root},
        argv=["/usr/bin/python3", "-c", "print(open('/tmp/vare-canary-secret').read())"],
        cwd="/workspace",
        timeout=20,
        max_output=2000,
    )
    assert "CANARY-VALUE" not in jailed["stdout"]


def test_vare_fault_before_commit_reapplies_once_and_baseline_restart_discards_the_edit():
    spec, workspace, tools = _open_task()
    try:
        now = {"t": 1_000_000.0}
        store = TrajectoryStore.create(
            workspace.path.parent / "state.db",
            run_id="coding-agent-test",
            record_bytes=262144,
        )
        episode = DurableEpisode(
            store=store,
            trajectory_id="t-grad-accum-s1",
            policy=PolicyCheckpoint("scripted-agent", 1, hashlib.sha256(b"scripted").hexdigest()),
            worker_id="worker-1",
            snapshot_dir=workspace.path.parent / "snapshots",
            clock=lambda: now["t"],
            heartbeat=False,
        )
        episode.enroll(
            Task(id=spec.id, prompt=spec.prompt, family=spec.family),
            expected_steps=6,
        )
        with pytest.raises(InjectedFault):
            _run(
                tools,
                [FIX, FINISH],
                durable=episode,
                fault_after_steps=1,
                fault_phase="before_commit",
            )
        assert "len(microbatches)" in (workspace.path / "grad_accum.py").read_text(encoding="utf-8")
        now["t"] += 1000
        resumed = _run(tools, [FIX, FINISH], durable=episode)
        text = (workspace.path / "grad_accum.py").read_text(encoding="utf-8")
        assert text.count("scale = float(len(microbatches))") == 1
        assert "scale = float(accumulation_steps)" not in text
        assert resumed.stop_reason == "agent_finished"
        assert resumed.resumed_from_step == 0
        tools.restore(None)
        with pytest.raises(InjectedFault):
            _run(tools, [FIX, FINISH], fault_after_steps=1, fault_phase="before_commit")
        tools.restore(None)
        assert "accumulation_steps" in (workspace.path / "grad_accum.py").read_text(encoding="utf-8")
        restarted = _run(tools, [FIX, FINISH])
        assert restarted.stop_reason == "agent_finished"
        assert (workspace.path / "grad_accum.py").read_text(encoding="utf-8").count("scale = float(len(microbatches))") == 1
    finally:
        workspace.close()
