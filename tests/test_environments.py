from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import time

import pytest

from vare.environments import (
    CommandSpec,
    EngineeringTaskSpec,
    ExecutableEvaluator,
    ResourceLimits,
    Workspace,
)
from vare.environments.runner import run_command


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "benchmarks/smoke/stable_logsumexp/task.json"
ORACLE = ROOT / "tests/fixtures/oracle/fix_stable_logsumexp.py"


def test_task_spec_roundtrip_and_baseline_fails():
    spec = EngineeringTaskSpec.load(SPEC)
    assert spec.id == "smoke-stable-logsumexp"
    with Workspace(spec) as ws:
        result = ExecutableEvaluator(spec).evaluate(ws.path)
    assert not result.passed
    assert result.integrity_ok
    assert result.correctness < 1.0


def test_oracle_patch_passes_and_has_diff():
    spec = EngineeringTaskSpec.load(SPEC)
    with Workspace(spec) as ws:
        subprocess.run(["python", str(ORACLE), str(ws.path)], check=True)
        result = ExecutableEvaluator(spec).evaluate(ws.path)
    assert result.passed
    assert result.correctness == 1.0
    assert "subject.py" in result.workspace_diff


def test_protected_evaluator_tampering_fails_closed():
    spec = EngineeringTaskSpec.load(SPEC)
    with Workspace(spec) as ws:
        (ws.path / "test_subject.py").write_text("def test_fake(): assert True\n", encoding="utf-8")
        result = ExecutableEvaluator(spec).evaluate(ws.path)
    assert not result.passed
    assert not result.integrity_ok
    assert any("protected file modified" in x for x in result.integrity_failures)


def test_relative_local_source_is_portable_and_not_serialized(monkeypatch, tmp_path):
    spec = EngineeringTaskSpec.load(SPEC)
    assert spec.to_dict()["source"]["local_path"] == "."
    assert "_source_base" not in spec.to_dict()
    monkeypatch.chdir(tmp_path)
    with Workspace(spec) as ws:
        assert (ws.path / "subject.py").is_file()


def test_command_output_budget_is_enforced_while_streaming(tmp_path):
    marker = tmp_path / "descendant-finished"
    child = (
        "import pathlib,sys,time; time.sleep(.5); "
        "pathlib.Path(sys.argv[1]).write_text('alive')"
    )
    parent = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); "
        "sys.stdout.write('x'*1048576); sys.stdout.flush(); "
        "sys.stderr.write('y'*1048576); sys.stderr.flush()"
    )
    command = CommandSpec(
        argv=(sys.executable, "-c", parent, child, str(marker)),
        name="output-flood",
    )
    limits = ResourceLimits(
        wall_time_s=3,
        cpu_time_s=None,
        memory_mb=None,
        max_output_bytes=128,
    )

    result = run_command(tmp_path, command, limits)

    assert result.output_limited
    assert not result.timed_out
    assert not result.passed
    assert len(result.stdout.encode("utf-8")) <= limits.max_output_bytes
    assert len(result.stderr.encode("utf-8")) <= limits.max_output_bytes
    if os.name == "posix":
        time.sleep(.6)
        assert not marker.exists()


def test_command_below_output_budget_passes(tmp_path):
    result = run_command(
        tmp_path,
        CommandSpec(argv=(sys.executable, "-c", "print('ok')")),
        ResourceLimits(
            wall_time_s=3,
            cpu_time_s=None,
            memory_mb=None,
            max_output_bytes=128,
        ),
    )

    assert result.passed
    assert not result.output_limited
    assert result.stdout == "ok\n"


def test_command_at_output_budget_is_not_limited(tmp_path):
    limit = 128
    result = run_command(
        tmp_path,
        CommandSpec(
            argv=(sys.executable, "-c", f"import sys; sys.stdout.write('x'*{limit})")
        ),
        ResourceLimits(
            wall_time_s=3,
            cpu_time_s=None,
            memory_mb=None,
            max_output_bytes=limit,
        ),
    )

    assert result.passed
    assert not result.output_limited
    assert len(result.stdout.encode("utf-8")) == limit


def test_command_timeout_remains_distinct_from_output_overflow(tmp_path):
    result = run_command(
        tmp_path,
        CommandSpec(argv=(sys.executable, "-c", "import time; print('start',flush=True); time.sleep(5)")),
        ResourceLimits(
            wall_time_s=.15,
            cpu_time_s=None,
            memory_mb=None,
            max_output_bytes=128,
        ),
    )

    assert result.timed_out
    assert not result.output_limited
    assert not result.passed
    assert result.returncode == 124
    assert result.stdout == "start\n"
