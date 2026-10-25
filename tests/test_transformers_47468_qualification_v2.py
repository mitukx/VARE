"""Deterministic unit tests for v2 identity checks and interrupted-attempt recovery.

These tests use temporary records and fake subprocess results. They never launch
the Transformers grader or a model.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_transformers_47468_qualification_v2 as runner


def _identity(executable: Path) -> dict[str, object]:
    return {
        "executable": str(executable.resolve()),
        "sys_version": "3.12.12 test-build",
        "version_info": [3, 12, 12],
        "implementation": "cpython",
        "cache_tag": "cpython-312",
    }


class FakeRun:
    def __init__(self, identity: dict[str, object], host: bytes, overlay: bytes):
        self.identity = identity
        self.host = host
        self.overlay = overlay
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        command = list(command)
        self.commands.append(command)
        if "-c" in command:
            stdout = json.dumps(self.identity, sort_keys=True)
        elif "--path" in command:
            stdout = self.overlay.decode()
        else:
            stdout = self.host.decode()
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def _write_lock(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def test_runtime_identity_accepts_exact_python_and_inventory(tmp_path):
    python = tmp_path / "python"
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    host = b"pip==25.0.1\npytest==8.4.2\n"
    overlay_bytes = b"numpy==2.5.4\ntokenizers==0.23.3\n"
    fake = FakeRun(_identity(python), host, overlay_bytes)

    result = runner.validate_runtime_identity(
        python, overlay, _identity(python), _write_lock(tmp_path / "host.txt", host),
        _write_lock(tmp_path / "overlay.txt", overlay_bytes), run=fake,
    )

    assert result["inventories"]["host"]["actual_sha256"] == runner.sha_bytes(host)
    assert result["inventories"]["overlay"]["actual_sha256"] == runner.sha_bytes(overlay_bytes)
    assert ["--all" in command for command in fake.commands[1:]] == [True, True]


def test_runtime_identity_rejects_python_version_drift(tmp_path):
    python = tmp_path / "python"
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    expected = _identity(python)
    changed = {**expected, "sys_version": "3.12.13 different-build"}
    fake = FakeRun(changed, b"", b"")

    with pytest.raises(RuntimeError, match="runtime Python identity drift"):
        runner.validate_runtime_identity(
            python, overlay, expected, _write_lock(tmp_path / "host.txt", b""),
            _write_lock(tmp_path / "overlay.txt", b""), run=fake,
        )
    assert len(fake.commands) == 1


def test_runtime_identity_rejects_host_dependency_drift(tmp_path):
    python = tmp_path / "python"
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    expected_host = b"pip==25.0.1\npytest==8.4.2\n"
    fake = FakeRun(_identity(python), b"pip==25.0.1\npytest==9.0.0\n", b"")

    with pytest.raises(RuntimeError, match="host dependency inventory differs"):
        runner.validate_runtime_identity(
            python, overlay, _identity(python), _write_lock(tmp_path / "host.txt", expected_host),
            _write_lock(tmp_path / "overlay.txt", b""), run=fake,
        )
    assert len(fake.commands) == 2


def test_runtime_identity_rejects_overlay_dependency_drift(tmp_path):
    python = tmp_path / "python"
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    host = b"pip==25.0.1\npytest==8.4.2\n"
    expected_overlay = b"numpy==2.5.4\ntokenizers==0.23.3\n"
    fake = FakeRun(_identity(python), host, b"numpy==2.5.4\ntokenizers==0.23.4\n")

    with pytest.raises(RuntimeError, match="overlay dependency inventory differs"):
        runner.validate_runtime_identity(
            python, overlay, _identity(python), _write_lock(tmp_path / "host.txt", host),
            _write_lock(tmp_path / "overlay.txt", expected_overlay), run=fake,
        )
    assert len(fake.commands) == 3


def _seed_manifest(manifest: Path) -> None:
    event = {
        "sequence": 1,
        "event_type": "protocol_frozen",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "previous_event_sha256": "0" * 64,
    }
    event["event_sha256"] = runner.event_hash(event)
    runner.write_new(manifest, (json.dumps(event, sort_keys=True) + "\n").encode())


def test_interrupted_attempt_reconstructs_partial_artifacts_and_blocks_rerun(tmp_path):
    experiment = tmp_path / "experiment"
    run_dir = experiment / "run-v2"
    records = experiment / "records"
    logs = run_dir / "logs"
    resources = run_dir / "resources"
    records.mkdir(parents=True)
    logs.mkdir(parents=True)
    resources.mkdir(parents=True)
    manifest = records / "manifest.jsonl"
    _seed_manifest(manifest)
    invocation_id = "attempt-0001"
    artifacts = {
        "stdout": "logs/baseline-attempt.stdout.log",
        "stderr": "logs/baseline-attempt.stderr.log",
        "resources": "resources/baseline-attempt.jsonl",
        "record": "records/baseline-attempt.json",
    }
    source_identity = {"target_path": "src/transformers/utils/generic.py", "target_sha256": "a" * 64}
    runner.append_event("phase_admitted", {
        "invocation_id": invocation_id,
        "phase": "baseline",
        "phase_order_index": 0,
        "source_identity": source_identity,
        "command_sha256": "b" * 64,
        "artifacts": artifacts,
    }, manifest_path=manifest)
    runner.append_event("phase_started", {
        "invocation_id": invocation_id,
        "phase": "baseline",
        "started_at_utc": "2026-01-01T00:00:01+00:00",
        "command_argv": ["fake-test-command"],
    }, manifest_path=manifest)
    runner.append_event("phase_process_started", {
        "invocation_id": invocation_id,
        "phase": "baseline",
        "pid": 31415,
        "pid_create_time": 1234.5,
    }, manifest_path=manifest)
    (run_dir / artifacts["stdout"]).write_bytes(b"partial output\n")
    (run_dir / artifacts["stderr"]).write_bytes(b"")
    (run_dir / artifacts["resources"]).write_bytes(b'{"sample":1}\n')

    recovered = runner.recover_interrupted_attempts(manifest, run_dir, experiment)

    assert len(recovered) == 1
    record = recovered[0]["record"]
    assert record["status"] == "INTERRUPTED_AFTER_PROCESS_START_RECONSTRUCTED"
    assert record["process_started_event"]["pid"] == 31415
    assert record["artifact_observations"]["stdout"]["sha256"] == runner.sha_bytes(b"partial output\n")
    assert record["artifact_observations"]["record"]["exists"] is False
    events = runner.load_manifest(manifest)
    assert events[-1]["event_type"] == "phase_interrupted_recovered"
    assert runner.recover_interrupted_attempts(manifest, run_dir, experiment) == []
    with pytest.raises(RuntimeError, match="cannot rerun"):
        runner.require_no_prior_attempt(events)


def test_admitted_but_not_started_attempt_is_reconstructed_and_terminal(tmp_path):
    experiment = tmp_path / "experiment"
    run_dir = experiment / "run-v2"
    (experiment / "records").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "resources").mkdir()
    manifest = experiment / "records/manifest.jsonl"
    _seed_manifest(manifest)
    invocation_id = "attempt-not-started"
    artifacts = {
        "stdout": "logs/out.log",
        "stderr": "logs/err.log",
        "resources": "resources/samples.jsonl",
        "record": "records/phase.json",
    }
    runner.append_event("phase_admitted", {
        "invocation_id": invocation_id,
        "phase": "baseline",
        "phase_order_index": 0,
        "artifacts": artifacts,
        "source_identity": {},
        "command_sha256": "c" * 64,
    }, manifest_path=manifest)

    recovered = runner.recover_interrupted_attempts(manifest, run_dir, experiment)

    assert recovered[0]["record"]["status"] == "INTERRUPTED_BEFORE_START_RECONSTRUCTED"
    assert runner.prior_attempt_exists(runner.load_manifest(manifest))


def test_phase_started_without_process_record_is_marked_launch_uncertain(tmp_path):
    experiment = tmp_path / "experiment"
    run_dir = experiment / "run-v2"
    records = experiment / "records"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "resources").mkdir()
    records.mkdir(parents=True)
    manifest = records / "manifest.jsonl"
    _seed_manifest(manifest)
    invocation_id = "attempt-launch-uncertain"
    artifacts = {
        "stdout": "logs/out.log",
        "stderr": "logs/err.log",
        "resources": "resources/samples.jsonl",
        "record": "records/phase.json",
    }
    runner.append_event("phase_admitted", {
        "invocation_id": invocation_id,
        "phase": "baseline",
        "phase_order_index": 0,
        "artifacts": artifacts,
        "source_identity": {},
        "command_sha256": "e" * 64,
    }, manifest_path=manifest)
    runner.append_event("phase_started", {
        "invocation_id": invocation_id,
        "phase": "baseline",
    }, manifest_path=manifest)

    recovered = runner.recover_interrupted_attempts(manifest, run_dir, experiment)

    record = recovered[0]["record"]
    assert record["status"] == "INTERRUPTED_LAUNCH_STATE_UNCERTAIN_RECONSTRUCTED"
    assert record["process_started_event"] is None


def test_recovery_is_idempotent_if_interrupted_before_manifest_append(tmp_path, monkeypatch):
    experiment = tmp_path / "experiment"
    run_dir = experiment / "run-v2"
    records = experiment / "records"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "resources").mkdir()
    records.mkdir(parents=True)
    manifest = records / "manifest.jsonl"
    _seed_manifest(manifest)
    invocation_id = "attempt-recovery-crash"
    artifacts = {
        "stdout": "logs/out.log",
        "stderr": "logs/err.log",
        "resources": "resources/samples.jsonl",
        "record": "records/phase.json",
    }
    runner.append_event("phase_admitted", {
        "invocation_id": invocation_id,
        "phase": "baseline",
        "phase_order_index": 0,
        "artifacts": artifacts,
        "source_identity": {},
        "command_sha256": "d" * 64,
    }, manifest_path=manifest)
    runner.append_event("phase_started", {
        "invocation_id": invocation_id,
        "phase": "baseline",
    }, manifest_path=manifest)
    for relpath in artifacts.values():
        if relpath.endswith("phase.json"):
            continue
        (run_dir / relpath).write_bytes(b"partial\n")

    append = runner.append_event
    monkeypatch.setattr(runner, "append_event", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("simulated crash")))
    with pytest.raises(OSError, match="simulated crash"):
        runner.recover_interrupted_attempts(manifest, run_dir, experiment)
    monkeypatch.setattr(runner, "append_event", append)

    recovered = runner.recover_interrupted_attempts(manifest, run_dir, experiment)
    assert len(recovered) == 1
    assert recovered[0]["record"]["status"] == "INTERRUPTED_LAUNCH_STATE_UNCERTAIN_RECONSTRUCTED"
    assert runner.load_manifest(manifest)[-1]["event_type"] == "phase_interrupted_recovered"
    with pytest.raises(RuntimeError, match="cannot rerun"):
        runner.require_no_prior_attempt(runner.load_manifest(manifest))
