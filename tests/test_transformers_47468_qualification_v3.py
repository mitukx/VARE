"""Deterministic unit tests for v3 identity, recovery, and journal integrity.

These tests use temporary records, fake inventory subprocess results, and one
isolated fake writer process. They never launch the Transformers grader or model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

import pytest

from scripts import run_transformers_47468_qualification_v3 as runner


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


def test_recovery_requires_the_exact_frozen_checkout_and_source_hashes(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    event = {
        "sequence": 1,
        "event_type": "protocol_frozen",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "previous_event_sha256": "0" * 64,
        "protocol_commit": "a" * 40,
        "protocol_tree": "b" * 40,
        "protocol_sha256": "c" * 64,
        "runner_sha256": "d" * 64,
        "lock_sha256": "e" * 64,
    }
    event["event_sha256"] = runner.event_hash(event)
    runner.write_new(manifest, (json.dumps(event, sort_keys=True) + "\n").encode())
    expected = {
        "protocol_commit": "a" * 40,
        "protocol_tree": "b" * 40,
        "protocol_sha256": "c" * 64,
        "runner_sha256": "d" * 64,
        "lock_sha256": "e" * 64,
    }

    runner.validate_recovery_identity(manifest, **expected)
    with pytest.raises(RuntimeError, match="checkout commit differs"):
        runner.validate_recovery_identity(manifest, **{**expected, "protocol_commit": "f" * 40})
    with pytest.raises(RuntimeError, match="source lock differs"):
        runner.validate_recovery_identity(manifest, **{**expected, "lock_sha256": "f" * 64})
    assert manifest.read_bytes().endswith(b"\n")


def _prepare_pending_tail(manifest: Path, suffix: bytes) -> tuple[bytes, bytes]:
    rows = runner.load_manifest(manifest)
    prefix = manifest.read_bytes()
    row = {
        "sequence": len(rows) + 1,
        "event_type": "phase_admitted",
        "timestamp_utc": "2026-01-01T00:00:02+00:00",
        "previous_event_sha256": rows[-1]["event_sha256"] if rows else "0" * 64,
        "invocation_id": "pending-admission",
        "phase": "baseline",
    }
    row["event_sha256"] = runner.event_hash(row)
    event_line = (json.dumps(row, sort_keys=True) + "\n").encode()
    pending_path = runner._manifest_pending_path(manifest)
    runner.write_json_new(pending_path, runner._pending_record(row, event_line, prefix))
    fd = os.open(manifest, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(fd, suffix)
        os.fsync(fd)
    finally:
        os.close(fd)
    return prefix, event_line


def test_interrupted_attempt_reconstructs_partial_artifacts_and_blocks_rerun(tmp_path):
    experiment = tmp_path / "experiment"
    run_dir = experiment / "run-v3"
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
        "command_argv": [sys.executable, "-c", "fake-writer"],
    }, manifest_path=manifest)
    stdout_file = run_dir / artifacts["stdout"]
    writer_program = (
        "import os,sys,time\n"
        "p=sys.argv[1]\n"
        "for i in range(100):\n"
        "    with open(p,'ab') as f:\n"
        "        f.write((str(i)+'\\n').encode()); f.flush(); os.fsync(f.fileno())\n"
        "    time.sleep(0.025)\n"
    )
    pid_file = run_dir / "logs/writer.pid"
    parent_program = (
        "import subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[3],sys.argv[1]])\n"
        "open(sys.argv[2],'w').write(str(child.pid))\n"
        "time.sleep(0.12)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", parent_program, str(stdout_file), str(pid_file), writer_program],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        child_info = psutil.Process(child.pid)
        runner.append_event("phase_process_started", {
            "invocation_id": invocation_id,
            "phase": "baseline",
            "pid": child.pid,
            "pid_create_time": child_info.create_time(),
            "process_group_id": os.getpgid(child.pid),
        }, manifest_path=manifest)
        deadline = time.monotonic() + 2
        while (not stdout_file.exists() or stdout_file.stat().st_size == 0) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stdout_file.exists() and stdout_file.stat().st_size > 0
        child.wait(timeout=2)
        assert runner._group_has_live_processes(child.pid), "detached writer must survive its recorded group leader"
        (run_dir / artifacts["stderr"]).write_bytes(b"")
        (run_dir / artifacts["resources"]).write_bytes(b'{"sample":1}\n')

        # Simulate parent loss: the detached fake writer is still active when recovery begins.
        recovered = runner.recover_interrupted_attempts(manifest, run_dir, experiment)
        record = recovered[0]["record"]
        assert record["status"] == "INTERRUPTED_AFTER_PROCESS_START_RECONSTRUCTED"
        assert record["process_started_event"]["pid"] == child.pid
        assert "SIGTERM" in record["process_quiescence"]["signals_sent"]
        before = (stdout_file.stat().st_size, runner.sha_file(stdout_file))
        time.sleep(0.12)
        after = (stdout_file.stat().st_size, runner.sha_file(stdout_file))
        assert after == before
        assert record["artifact_observations"]["stdout"]["sha256"] == before[1]
        assert record["artifact_observations"]["record"]["exists"] is False
        events = runner.load_manifest(manifest)
        assert events[-1]["event_type"] == "phase_interrupted_recovered"
        assert runner.recover_interrupted_attempts(manifest, run_dir, experiment) == []
        with pytest.raises(RuntimeError, match="cannot rerun"):
            runner.require_no_prior_attempt(events)
    finally:
        if child.poll() is None:
            try:
                os.killpg(child.pid, 9)
            except ProcessLookupError:
                pass
            child.wait(timeout=2)
        else:
            try:
                os.killpg(child.pid, 9)
            except ProcessLookupError:
                pass


def test_admitted_but_not_started_attempt_is_reconstructed_and_terminal(tmp_path):
    experiment = tmp_path / "experiment"
    run_dir = experiment / "run-v3"
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


def test_pid_create_time_mismatch_refuses_to_signal_process_group():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        created = psutil.Process(child.pid).create_time()
        event = {
            "pid": child.pid,
            "pid_create_time": created + 10,
            "process_group_id": child.pid,
        }
        with pytest.raises(RuntimeError, match="create-time mismatch"):
            runner.quiesce_recorded_process(event)
        assert child.poll() is None
    finally:
        if child.poll() is None:
            try:
                os.killpg(child.pid, 9)
            except ProcessLookupError:
                pass
            child.wait(timeout=2)


def test_normal_phase_completion_quiesces_orphan_writer_before_hashing(tmp_path):
    target = tmp_path / "phase-output.log"
    pid_file = tmp_path / "writer.pid"
    writer_program = (
        "import os,sys,time\n"
        "p=sys.argv[1]\n"
        "for i in range(1000):\n"
        "    with open(p,'ab') as f:\n"
        "        f.write((str(i)+'\\n').encode()); f.flush(); os.fsync(f.fileno())\n"
        "    time.sleep(0.02)\n"
    )
    leader_program = (
        "import subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[3],sys.argv[1]])\n"
        "open(sys.argv[2],'w').write(str(child.pid))\n"
        "time.sleep(0.12)\n"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_program, str(target), str(pid_file), writer_program],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 2
        while (not target.exists() or target.stat().st_size == 0) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert target.exists() and target.stat().st_size > 0
        assert pid_file.read_text().isdigit()
        event = {
            "pid": leader.pid,
            "pid_create_time": psutil.Process(leader.pid).create_time(),
            "process_group_id": os.getpgid(leader.pid),
        }

        return_code, quiescence = runner.wait_and_quiesce_phase_process(leader, event)

        assert return_code == 0
        assert "SIGTERM" in quiescence["signals_sent"]
        assert not runner._group_has_live_processes(leader.pid)
        size_hash_after = (target.stat().st_size, runner.sha_file(target))
        time.sleep(0.12)
        assert (target.stat().st_size, runner.sha_file(target)) == size_hash_after
    finally:
        if runner._group_has_live_processes(leader.pid):
            try:
                os.killpg(leader.pid, 9)
            except ProcessLookupError:
                pass
            if leader.poll() is None:
                leader.wait(timeout=2)


def test_failed_process_identity_journal_quiesces_unrecorded_launch(tmp_path, monkeypatch):
    target = tmp_path / "phase-output.log"
    pid_file = tmp_path / "writer.pid"
    writer_program = (
        "import os,sys,time\n"
        "p=sys.argv[1]\n"
        "for i in range(1000):\n"
        "    with open(p,'ab') as f:\n"
        "        f.write((str(i)+'\\n').encode()); f.flush(); os.fsync(f.fileno())\n"
        "    time.sleep(0.02)\n"
    )
    leader_program = (
        "import subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[3],sys.argv[1]])\n"
        "open(sys.argv[2],'w').write(str(child.pid))\n"
        "time.sleep(5)\n"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_program, str(target), str(pid_file), writer_program],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 2
        while (not target.exists() or target.stat().st_size == 0) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert target.exists() and target.stat().st_size > 0
        assert pid_file.read_text().isdigit()
        monkeypatch.setattr(
            runner, "append_event",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("journal unavailable")),
        )

        with pytest.raises(OSError, match="journal unavailable"):
            runner.record_phase_process_start(leader, "attempt-journal-failure", "baseline")

        assert leader.poll() is not None
        assert not runner._group_has_live_processes(leader.pid)
        size_hash_after = (target.stat().st_size, runner.sha_file(target))
        time.sleep(0.12)
        assert (target.stat().st_size, runner.sha_file(target)) == size_hash_after
    finally:
        if runner._group_has_live_processes(leader.pid):
            try:
                os.killpg(leader.pid, 9)
            except ProcessLookupError:
                pass
            if leader.poll() is None:
                leader.wait(timeout=2)


def test_phase_started_without_process_record_defers_hashing(tmp_path):
    experiment = tmp_path / "experiment"
    run_dir = experiment / "run-v3"
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

    assert recovered[0]["deferred"] is True
    assert recovered[0]["event"]["artifact_hashes_finalized"] is False
    assert not (records / f"recovered-phase-{invocation_id}.json").exists()
    assert runner.recover_interrupted_attempts(manifest, run_dir, experiment)[0]["deferred"] is True


def test_deferred_recovery_is_idempotent_if_append_is_interrupted(tmp_path, monkeypatch):
    experiment = tmp_path / "experiment"
    run_dir = experiment / "run-v3"
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
    assert recovered[0]["deferred"] is True
    assert runner.load_manifest(manifest)[-1]["event_type"] == "phase_recovery_deferred"
    with pytest.raises(RuntimeError, match="cannot rerun"):
        runner.require_no_prior_attempt(runner.load_manifest(manifest))


def test_torn_append_is_repaired_from_matching_pending_and_quarantined(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _seed_manifest(manifest)
    prefix = manifest.read_bytes()
    complete_event = json.loads(prefix)
    suffix = b'{"event_sha256":'
    prior, event_line = _prepare_pending_tail(manifest, suffix)

    rows = runner.load_manifest(manifest)

    assert rows[0] == complete_event
    assert rows[-2]["event_sha256"] == json.loads(event_line)["event_sha256"]
    assert rows[-1]["event_type"] == "journal_tail_repaired"
    assert rows[-1]["verified_prior_prefix_sha256"] == runner.sha_bytes(prior)
    quarantined = manifest.parent / rows[-1]["damaged_suffix_path"]
    assert quarantined.read_bytes() == suffix
    assert rows[-1]["damaged_suffix_sha256"] == runner.sha_bytes(suffix)
    assert manifest.read_bytes().startswith(prior)
    repaired_bytes = manifest.read_bytes()
    assert runner.load_manifest(manifest) == rows
    assert manifest.read_bytes() == repaired_bytes


def test_torn_repair_is_idempotent_if_interrupted_after_atomic_replace(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    _seed_manifest(manifest)
    suffix = b'{"event_sha256":'
    _prepare_pending_tail(manifest, suffix)
    remove_pending = runner._remove_pending
    monkeypatch.setattr(runner, "_remove_pending", lambda path: (_ for _ in ()).throw(OSError("crash after replace")))

    with pytest.raises(OSError, match="crash after replace"):
        runner.load_manifest(manifest)
    assert manifest.read_bytes().endswith(b"\n")
    assert runner._manifest_pending_path(manifest).exists()
    monkeypatch.setattr(runner, "_remove_pending", remove_pending)

    rows = runner.load_manifest(manifest)
    assert sum(row["event_type"] == "journal_tail_repaired" for row in rows) == 1
    assert not runner._manifest_pending_path(manifest).exists()


def test_torn_suffix_without_pending_is_preserved_and_rejected(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _seed_manifest(manifest)
    with manifest.open("ab") as stream:
        stream.write(b'{"partial":')
    damaged = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="without a matching pending-event record"):
        runner.load_manifest(manifest)

    assert manifest.read_bytes() == damaged
    assert not list(tmp_path.glob("*.quarantine-*.suffix"))


def test_mismatched_pending_suffix_is_rejected_without_rewriting_prefix(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _seed_manifest(manifest)
    prior = manifest.read_bytes()
    _prepare_pending_tail(manifest, b'{"attacker":')
    damaged = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="does not match the durable pending-event prefix"):
        runner.load_manifest(manifest)

    assert manifest.read_bytes() == damaged
    assert manifest.read_bytes().startswith(prior)
    assert not list(tmp_path.glob("*.quarantine-*.suffix"))


def test_corruption_before_torn_final_line_is_rejected(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _seed_manifest(manifest)
    runner.append_event("prior-valid-event", {"value": 1}, manifest_path=manifest)
    original = manifest.read_bytes()
    _prepare_pending_tail(manifest, b'{"sequence":3,"event_type":')
    damaged = manifest.read_bytes()
    first_line = damaged.splitlines(keepends=True)[0]
    tampered_first_line = first_line.replace(b"protocol_frozen", b"tampered_event", 1)
    manifest.write_bytes(tampered_first_line + b"".join(damaged.splitlines(keepends=True)[1:]))
    corrupted = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="manifest event hash mismatch at line 1"):
        runner.load_manifest(manifest)

    assert manifest.read_bytes() == corrupted
    assert not list(tmp_path.glob("*.quarantine-*.suffix"))
    assert original.startswith(first_line)


def test_pending_prior_prefix_mismatch_is_rejected(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _seed_manifest(manifest)
    _prepare_pending_tail(manifest, b'{"sequence":2')
    pending_path = runner._manifest_pending_path(manifest)
    pending = json.loads(pending_path.read_text())
    pending["prior_prefix_sha256"] = "f" * 64
    pending.pop("pending_record_sha256")
    pending["pending_record_sha256"] = runner.sha_bytes(json.dumps(pending, sort_keys=True, separators=(",", ":")).encode())
    pending_path.write_text(json.dumps(pending, sort_keys=True, indent=2) + "\n")
    damaged = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="pending-event prior prefix mismatch"):
        runner.load_manifest(manifest)

    assert manifest.read_bytes() == damaged
    assert not list(tmp_path.glob("*.quarantine-*.suffix"))
