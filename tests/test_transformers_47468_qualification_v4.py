"""No-grader regressions for v4's process-creation boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

import pytest

from scripts import run_transformers_47468_qualification_v4 as runner


def _lock() -> dict[str, str]:
    return {
        "protocol_id": "transformers-47468-generic-scalar-float-development-qualification-v4",
        "runner_path": "scripts/run_transformers_47468_qualification_v4.py",
        "protocol_sha256": runner.sha_file(runner.PROTOCOL),
        "runner_sha256": runner.sha_file(runner.SCRIPT),
        "core_runner_path": "scripts/run_transformers_47468_qualification_v3.py",
        "core_runner_sha256": runner.sha_file(runner.CORE_SCRIPT),
        "tests_path": "tests/test_transformers_47468_qualification_v4.py",
        "tests_sha256": runner.sha_file(Path(__file__).resolve()),
        "readme_sha256": runner.sha_file(runner.EXPERIMENT / "README.md"),
        "gitignore_path": ".gitignore",
        "gitignore_sha256": runner.sha_file(runner.ROOT / ".gitignore"),
    }


def test_v4_source_lock_pins_wrapper_and_unchanged_v3_core():
    runner.validate_locked_source_identity(_lock())
    runner.validate_core_before_import(_lock())
    changed = _lock()
    changed["core_runner_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="immutable v3 core runner SHA mismatch"):
        runner.validate_locked_source_identity(changed)
    with pytest.raises(RuntimeError, match="immutable v3 core runner SHA mismatch before import"):
        runner.validate_core_before_import(changed)
    assert runner.sha_file(runner.CORE_SCRIPT) == (
        "148e105cc2c24fa4a24b47b476360484b95f9093f97082fc251e3360c03f7705"
    )


def test_v4_profile_denies_process_creation_and_network():
    assert runner.PROFILE == "(version 1) (allow default) (deny network*) (deny process-fork)"


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        ({"network_denied": True, "fork_denied": False, "spawn_denied": True}, "process-fork"),
        ({"network_denied": True, "fork_denied": True, "spawn_denied": False}, "subprocess-spawn"),
        ({"network_denied": False, "fork_denied": True, "spawn_denied": True}, "network-deny"),
    ],
)
def test_boundary_probe_rejects_each_unblocked_capability(monkeypatch, checks, message):
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(checks), stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match=message):
        runner.validate_execution_boundary(Path(sys.executable))


def test_boundary_probe_accepts_only_all_three_denials(monkeypatch):
    checks = {"network_denied": True, "fork_denied": True, "spawn_denied": True}
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(checks), stderr=""
        ),
    )
    result = runner.validate_execution_boundary(Path(sys.executable))
    assert result["fork_denied"] and result["spawn_denied"] and result["network_denied"]
    assert result["profile"] == runner.PROFILE
    assert len(result["probe_sha256"]) == 64


@pytest.mark.skipif(platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file(),
                    reason="the authenticated execution-boundary probe requires macOS sandbox-exec")
def test_actual_sandbox_denies_network_fork_and_subprocess_spawn():
    result = runner.validate_execution_boundary(Path(sys.executable))
    assert result["network_denied"] is True
    assert result["fork_denied"] is True
    assert result["spawn_denied"] is True


def test_v4_does_not_change_v3_runner_bytes():
    assert runner.sha_file(runner.CORE_SCRIPT) == (
        "148e105cc2c24fa4a24b47b476360484b95f9093f97082fc251e3360c03f7705"
    )


def _append_event(manifest: Path, event_type: str, payload: dict[str, object]) -> None:
    runner.core.append_event(event_type, payload, manifest_path=manifest)


def _seed_frozen_manifest(manifest: Path) -> None:
    event = {
        "sequence": 1,
        "event_type": "protocol_frozen",
        "timestamp_utc": "2026-10-26T00:00:00+00:00",
        "previous_event_sha256": "0" * 64,
    }
    event["event_sha256"] = runner.core.event_hash(event)
    runner.core.write_new(manifest, (json.dumps(event, sort_keys=True) + "\n").encode())


def _seed_ambiguous_launch(manifest: Path, invocation_id: str, expected_test: str,
                           cwd: Path, *, launch_token: bool,
                           command: list[str] | None = None) -> dict[str, str]:
    command = command or [sys.executable, expected_test]
    command_sha256 = runner.core.sha_bytes(json.dumps(command, separators=(",", ":")).encode())
    artifacts = {
        "stdout": "logs/fake.stdout.log",
        "stderr": "logs/fake.stderr.log",
        "resources": "resources/fake.jsonl",
        "record": "records/fake.json",
    }
    _append_event(manifest, "phase_admitted", {
        "invocation_id": invocation_id,
        "phase": "baseline",
        "phase_order_index": 0,
        "command_sha256": command_sha256,
        "artifacts": artifacts,
    })
    _append_event(manifest, "phase_started", {
        "invocation_id": invocation_id,
        "phase": "baseline",
        "started_at_utc": "2026-10-26T00:00:01+00:00",
        "command_argv": command,
        "cwd": str(cwd),
        "environment": {"VARE_PHASE_INVOCATION_ID": invocation_id},
    })
    if launch_token:
        _append_event(manifest, "phase_launch_token", {
            "invocation_id": invocation_id,
            "marker_name": "VARE_PHASE_INVOCATION_ID",
            "marker_value": invocation_id,
            "command_sha256": command_sha256,
            "recorded_before_process_creation": True,
        })
    return artifacts


def test_launch_token_recovers_and_stops_writer_before_hashing(tmp_path):
    experiment = tmp_path / "experiment"
    run_dir = tmp_path / "run"
    manifest = experiment / "records" / "manifest.jsonl"
    (experiment / "records").mkdir(parents=True)
    for name in ("logs", "resources", "records"):
        (run_dir / name).mkdir(parents=True)
    _seed_frozen_manifest(manifest)
    invocation_id = str(uuid.uuid4())
    test_source = tmp_path / "fake_test.py"
    test_source.write_text("import time\nwhile True:\n print('writer', flush=True)\n time.sleep(0.01)\n")
    artifacts = _seed_ambiguous_launch(manifest, invocation_id, str(test_source), tmp_path,
                                       launch_token=True)
    stdout_path = run_dir / artifacts["stdout"]
    stderr_path = run_dir / artifacts["stderr"]
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        child_env = {"PATH": os.environ.get("PATH", ""), "VARE_PHASE_INVOCATION_ID": invocation_id}
        child = runner._ORIGINAL_POPEN(
            [sys.executable, str(test_source)], cwd=tmp_path, env=child_env,
            stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, start_new_session=True,
        )
        time.sleep(0.05)
        recovered = runner.core.recover_interrupted_attempts(manifest, run_dir, experiment)
        stdout.flush()
        stderr.flush()
        child.wait(timeout=2)
    events = runner.core.load_manifest(manifest)
    process_event = next(row for row in events if row.get("event_type") == "phase_process_started")
    assert process_event["recovered_from_launch_token"] is True
    assert process_event["pid"] == child.pid
    assert len(recovered) == 1
    assert recovered[0]["record"]["status"] == "INTERRUPTED_AFTER_PROCESS_START_RECONSTRUCTED"
    assert recovered[0]["event"]["event_type"] == "phase_interrupted_recovered"
    before = (stdout_path.stat().st_size, runner.sha_file(stdout_path))
    time.sleep(0.1)
    after = (stdout_path.stat().st_size, runner.sha_file(stdout_path))
    assert after == before
    assert child.poll() is not None


def test_missing_launch_process_stays_fail_closed_without_artifact_hashes(tmp_path):
    experiment = tmp_path / "experiment"
    run_dir = tmp_path / "run"
    manifest = experiment / "records" / "manifest.jsonl"
    (experiment / "records").mkdir(parents=True)
    for name in ("logs", "resources", "records"):
        (run_dir / name).mkdir(parents=True)
    _seed_frozen_manifest(manifest)
    invocation_id = str(uuid.uuid4())
    fake_test = tmp_path / "absent_test.py"
    fake_test.write_text("pass\n")
    artifacts = _seed_ambiguous_launch(manifest, invocation_id, str(fake_test), tmp_path,
                                       launch_token=True)
    (run_dir / artifacts["stdout"]).write_bytes(b"partial\n")
    recovered = runner.core.recover_interrupted_attempts(manifest, run_dir, experiment)
    events = runner.core.load_manifest(manifest)
    assert recovered and recovered[0]["deferred"] is True
    assert any(row.get("event_type") == "phase_launch_scan_completed" for row in events)
    assert any(row.get("event_type") == "phase_recovery_deferred" for row in events)
    assert not any(row.get("event_type") == "phase_process_started" for row in events)
    assert not list((experiment / "records").glob("recovered-phase-*.json"))


def test_launch_hook_persists_token_before_process_creation(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _seed_frozen_manifest(manifest)
    monkeypatch.setattr(runner.core, "MANIFEST", manifest)
    token = str(uuid.uuid4())
    command = [sys.executable, "-c", "pass"]
    saw_durable_token = False

    def fake_popen(*args, **kwargs):
        nonlocal saw_durable_token
        rows = runner.core.load_manifest(manifest)
        token_event = rows[-1]
        assert token_event["event_type"] == "phase_launch_token"
        assert token_event["invocation_id"] == token
        assert kwargs["env"]["VARE_PHASE_INVOCATION_ID"] == token
        saw_durable_token = True
        return "started"

    monkeypatch.setattr(runner, "_ORIGINAL_POPEN", fake_popen)
    result = runner._tracked_phase_popen(
        command,
        env={"HOME": str(tmp_path / f"home-{token}")},
    )
    assert result == "started"
    assert saw_durable_token is True
    assert runner.subprocess.Popen is not runner.core.subprocess.Popen


def _seed_full_frozen_identity(manifest: Path, commit: str, tree: str) -> None:
    event = {
        "sequence": 1,
        "event_type": "protocol_frozen",
        "timestamp_utc": "2026-10-26T00:00:00+00:00",
        "previous_event_sha256": "0" * 64,
        "protocol_commit": commit,
        "protocol_tree": tree,
        "protocol_sha256": runner.sha_file(runner.PROTOCOL),
        "runner_sha256": runner.sha_file(runner.SCRIPT),
        "lock_sha256": runner.sha_file(runner.LOCK),
    }
    event["event_sha256"] = runner.core.event_hash(event)
    runner.core.write_new(manifest, (json.dumps(event, sort_keys=True) + "\n").encode())


def test_execute_identity_accepts_exact_frozen_clean_checkout(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    commit, tree = "a" * 40, "b" * 40
    _seed_full_frozen_identity(manifest, commit, tree)
    monkeypatch.setattr(runner.core, "MANIFEST", manifest)
    monkeypatch.setattr(runner.core, "repo_identity", lambda: (commit, tree))
    monkeypatch.setattr(runner.core.subprocess, "check_output", lambda *a, **k: "")
    result = runner.validate_frozen_execution_identity()
    assert result == {"commit": commit, "tree": tree, "lock_sha256": runner.sha_file(runner.LOCK)}


def test_execute_identity_rejects_lock_mutation_after_freeze(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    commit, tree = "a" * 40, "b" * 40
    _seed_full_frozen_identity(manifest, commit, tree)
    altered_lock = tmp_path / "protocol-v4.lock.json"
    altered_lock.write_bytes(runner.LOCK.read_bytes() + b"\n")
    monkeypatch.setattr(runner.core, "MANIFEST", manifest)
    monkeypatch.setattr(runner, "LOCK", altered_lock)
    monkeypatch.setattr(runner.core, "repo_identity", lambda: (commit, tree))
    monkeypatch.setattr(runner.core.subprocess, "check_output", lambda *a, **k: "")
    with pytest.raises(RuntimeError, match="source lock differs from the frozen"):
        runner.validate_frozen_execution_identity()


def test_identity_mismatch_does_not_repair_pending_manifest_before_rejection(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    commit, tree = "a" * 40, "b" * 40
    _seed_full_frozen_identity(manifest, commit, tree)
    prefix = manifest.read_bytes()
    first = json.loads(prefix.splitlines()[0])
    row = {
        "sequence": 2,
        "event_type": "phase_admitted",
        "timestamp_utc": "2026-10-26T00:00:01+00:00",
        "previous_event_sha256": first["event_sha256"],
        "invocation_id": str(uuid.uuid4()),
    }
    row["event_sha256"] = runner.core.event_hash(row)
    event_line = (json.dumps(row, sort_keys=True) + "\n").encode()
    manifest.write_bytes(prefix + event_line[:19])
    pending_path = manifest.with_name(manifest.name + ".pending.json")
    pending = runner.core._pending_record(row, event_line, prefix)
    pending_path.write_text(json.dumps(pending, sort_keys=True), encoding="utf-8")

    altered_lock = tmp_path / "protocol-v4.lock.json"
    altered_lock.write_bytes(runner.LOCK.read_bytes() + b"\n")
    before_manifest = manifest.read_bytes()
    before_pending = pending_path.read_bytes()
    monkeypatch.setattr(runner.core, "MANIFEST", manifest)
    monkeypatch.setattr(runner, "LOCK", altered_lock)
    monkeypatch.setattr(runner.core, "repo_identity", lambda: (commit, tree))
    monkeypatch.setattr(runner.core.subprocess, "check_output", lambda *a, **k: "")

    with pytest.raises(RuntimeError, match="source lock differs from the frozen"):
        runner.validate_frozen_execution_identity()

    assert manifest.read_bytes() == before_manifest
    assert pending_path.read_bytes() == before_pending
    assert not list(tmp_path.glob("manifest.jsonl.quarantine-*.suffix"))

    repair_probe = tmp_path / "repair-probe.jsonl"
    repair_probe.write_bytes(before_manifest)
    repair_probe.with_name(repair_probe.name + ".pending.json").write_bytes(before_pending)
    repaired = runner._ORIGINAL_LOAD_MANIFEST(repair_probe)
    assert repaired[-1]["event_type"] == "journal_tail_repaired"
    assert not repair_probe.with_name(repair_probe.name + ".pending.json").exists()


def test_execute_identity_rejects_dirty_worktree(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    commit, tree = "a" * 40, "b" * 40
    _seed_full_frozen_identity(manifest, commit, tree)
    monkeypatch.setattr(runner.core, "MANIFEST", manifest)
    monkeypatch.setattr(runner.core, "repo_identity", lambda: (commit, tree))
    monkeypatch.setattr(runner.core.subprocess, "check_output", lambda *a, **k: " M protocol lock")
    with pytest.raises(RuntimeError, match="worktree must be clean"):
        runner.validate_frozen_execution_identity()


def test_execute_identity_rejects_mismatched_commit(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _seed_full_frozen_identity(manifest, "a" * 40, "b" * 40)
    monkeypatch.setattr(runner.core, "MANIFEST", manifest)
    monkeypatch.setattr(runner.core, "repo_identity", lambda: ("c" * 40, "b" * 40))
    monkeypatch.setattr(runner.core.subprocess, "check_output", lambda *a, **k: "")
    with pytest.raises(RuntimeError, match="checkout commit differs"):
        runner.validate_frozen_execution_identity()


def test_runtime_records_and_phase_output_use_versioned_ignored_paths(tmp_path):
    assert runner.core.MANIFEST == runner.RUNTIME_ROOT / "records/manifest.jsonl"
    assert runner.RUNTIME_RUN == runner.RUNTIME_ROOT / "run"
    assert runner.RUNTIME_ROOT.is_relative_to(runner.EXPERIMENT)
    assert runner.core.MANIFEST.is_relative_to(runner.EXPERIMENT)
    assert runner.RUNTIME_RUN.is_relative_to(runner.EXPERIMENT)
    clean_repo = tmp_path / "clean-repo"
    clean_repo.mkdir()
    (clean_repo / ".gitignore").write_bytes((runner.ROOT / ".gitignore").read_bytes())
    subprocess.run(["git", "init", "--quiet", str(clean_repo)], check=True)
    subprocess.run(["git", "-C", str(clean_repo), "add", ".gitignore"], check=True)
    subprocess.run(
        ["git", "-C", str(clean_repo), "-c", "user.name=VARE test",
         "-c", "user.email=vare-test@example.invalid", "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    for path in (
        "results/transformers-47468-development-qualification-v4/run-v4/records/manifest.jsonl",
        "results/transformers-47468-development-qualification-v4/run-v4/run/records/phase.json",
        "results/transformers-47468-development-qualification-v4/run-v4/records/recovered-phase.json",
    ):
        result = subprocess.run(
            ["git", "-C", str(clean_repo), "check-ignore", "--no-index", "-v", path],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0
        assert result.stdout.startswith(".gitignore:")
        destination = clean_repo / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"retained runtime evidence\n")
    status = subprocess.run(
        ["git", "-C", str(clean_repo), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, check=True,
    )
    assert status.stdout == ""


@pytest.mark.parametrize("component", ["results", "experiment", "runtime-root", "records", "run"])
def test_pinned_runtime_location_rejects_symlinked_components(tmp_path, monkeypatch, component):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    results = repo / "results"
    experiment = results / "qualification-v4"
    runtime_root = experiment / "run-v4"
    runtime_run = runtime_root / "run"

    if component == "results":
        results.symlink_to(outside, target_is_directory=True)
    else:
        results.mkdir()
        if component == "experiment":
            experiment.symlink_to(outside, target_is_directory=True)
        else:
            experiment.mkdir()
            if component == "runtime-root":
                runtime_root.symlink_to(outside, target_is_directory=True)
            elif component == "records":
                runtime_root.mkdir()
                (runtime_root / "records").symlink_to(outside, target_is_directory=True)
            else:
                runtime_root.mkdir(exist_ok=True)
                if component == "run":
                    runtime_run.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(runner, "ROOT", repo)
    monkeypatch.setattr(runner, "EXPERIMENT", experiment)
    monkeypatch.setattr(runner, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(runner, "RUNTIME_RUN", runtime_run)
    with pytest.raises(RuntimeError, match="contains a symlink"):
        runner.validate_pinned_runtime_location()
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("sidecar", ["manifest.jsonl", "manifest.jsonl.lock", "manifest.jsonl.pending.json"])
def test_pinned_runtime_location_rejects_symlinked_manifest_files(tmp_path, monkeypatch, sidecar):
    repo = tmp_path / "repo"
    records = repo / "results/qualification-v4/run-v4/records"
    records.mkdir(parents=True)
    outside = tmp_path / "outside-file"
    outside.write_text("must remain untouched")

    monkeypatch.setattr(runner, "ROOT", repo)
    monkeypatch.setattr(runner, "EXPERIMENT", repo / "results/qualification-v4")
    monkeypatch.setattr(runner, "RUNTIME_ROOT", repo / "results/qualification-v4/run-v4")
    monkeypatch.setattr(runner, "RUNTIME_RUN", runner.RUNTIME_ROOT / "run")
    (records / sidecar).symlink_to(outside)

    with pytest.raises(RuntimeError, match="symlink"):
        runner.validate_pinned_runtime_location()
    assert outside.read_text() == "must remain untouched"


def test_runtime_write_wrapper_rechecks_symlinked_records_before_write(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    experiment = repo / "results/qualification-v4"
    runtime_root = experiment / "run-v4"
    experiment.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime_root).mkdir()
    (runtime_root / "records").symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(runner, "ROOT", repo)
    monkeypatch.setattr(runner, "EXPERIMENT", experiment)
    monkeypatch.setattr(runner, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(runner, "RUNTIME_RUN", runtime_root / "run")

    with pytest.raises(RuntimeError, match="contains a symlink"):
        runner.core.write_new(runtime_root / "records" / "manifest.jsonl", b"blocked\n")
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("mode", ["--freeze-record", "--execute", "--recover-interrupted"])
def test_mutating_cli_modes_reject_runtime_symlink_before_core(monkeypatch, tmp_path, mode):
    repo = tmp_path / "repo"
    repo.mkdir()
    results = repo / "results"
    results.mkdir()
    experiment = results / "qualification-v4"
    experiment.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    runtime_root = experiment / "run-v4"
    runtime_root.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(runner, "ROOT", repo)
    monkeypatch.setattr(runner, "EXPERIMENT", experiment)
    monkeypatch.setattr(runner, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(runner, "RUNTIME_RUN", runtime_root / "run")
    monkeypatch.setattr(sys, "argv", [str(runner.SCRIPT), mode])
    monkeypatch.setattr(
        runner, "validate_frozen_execution_identity",
        lambda: pytest.fail("frozen identity gate must not run through a symlink"),
    )
    monkeypatch.setattr(runner, "_ORIGINAL_MAIN", lambda: pytest.fail("core main must not run"))

    with pytest.raises(RuntimeError, match="contains a symlink"):
        runner.core.main()
    assert list(outside.iterdir()) == []


def test_cli_recovery_records_use_the_pinned_runtime_root(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _seed_frozen_manifest(manifest)
    captured = {}

    def recover(manifest_path, run_dir, experiment_root):
        captured.update({
            "manifest": manifest_path,
            "run_dir": run_dir,
            "experiment_root": experiment_root,
        })
        return []

    monkeypatch.setattr(runner, "_ORIGINAL_RECOVERY", recover)
    result = runner._recover_interrupted_attempts_with_launch_tokens(
        manifest, tmp_path / "run", runner.EXPERIMENT,
    )
    assert result == []
    assert captured["experiment_root"] == runner.RUNTIME_ROOT


@pytest.mark.parametrize("mode", ["--execute", "--recover-interrupted"])
def test_cli_requires_frozen_identity_gate_before_mutating_modes(monkeypatch, mode):
    calls = []
    monkeypatch.setattr(sys, "argv", [str(runner.SCRIPT), mode])
    monkeypatch.setattr(runner, "validate_frozen_execution_identity", lambda: calls.append(mode))
    monkeypatch.setattr(runner, "_ORIGINAL_MAIN", lambda: 0)
    assert runner.core.main() == 0
    assert calls == [mode]
    assert sys.argv[-2:] == ["--output", str(runner.RUNTIME_RUN)]


def test_mutating_mode_rejects_output_outside_pinned_runtime_path(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [str(runner.SCRIPT), "--execute", "--output", str(tmp_path)])
    monkeypatch.setattr(runner, "validate_frozen_execution_identity", lambda: None)
    monkeypatch.setattr(runner, "_ORIGINAL_MAIN", lambda: pytest.fail("core main must not run"))
    with pytest.raises(RuntimeError, match="pinned ignored runtime directory"):
        runner.core.main()


@pytest.mark.skipif(platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file(),
                    reason="process recovery probe requires macOS sandbox-exec")
def test_launch_token_recovers_sandboxed_writer_before_hashing(tmp_path):
    experiment = tmp_path / "experiment"
    run_dir = tmp_path / "run"
    manifest = experiment / "records" / "manifest.jsonl"
    (experiment / "records").mkdir(parents=True)
    for name in ("logs", "resources", "records"):
        (run_dir / name).mkdir(parents=True)
    _seed_frozen_manifest(manifest)
    invocation_id = str(uuid.uuid4())
    test_source = tmp_path / "sandboxed_fake_test.py"
    test_source.write_text("import time\nwhile True:\n print('writer', flush=True)\n time.sleep(0.01)\n")
    command = ["/usr/bin/sandbox-exec", "-p", runner.PROFILE, sys.executable, str(test_source)]
    artifacts = _seed_ambiguous_launch(manifest, invocation_id, str(test_source), tmp_path,
                                       launch_token=True, command=command)
    stdout_path = run_dir / artifacts["stdout"]
    stderr_path = run_dir / artifacts["stderr"]
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        child = runner._ORIGINAL_POPEN(
            command, cwd=tmp_path,
            env={"PATH": os.environ.get("PATH", ""), "VARE_PHASE_INVOCATION_ID": invocation_id},
            stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, start_new_session=True,
        )
        time.sleep(0.05)
        recovered = runner.core.recover_interrupted_attempts(manifest, run_dir, experiment)
        stdout.flush()
        stderr.flush()
        child.wait(timeout=2)
    events = runner.core.load_manifest(manifest)
    process_event = next(row for row in events if row.get("event_type") == "phase_process_started")
    assert process_event["recovered_from_launch_token"] is True
    assert process_event["pid"] == child.pid
    assert len(recovered) == 1
    assert recovered[0]["record"]["status"] == "INTERRUPTED_AFTER_PROCESS_START_RECONSTRUCTED"
    before = (stdout_path.stat().st_size, runner.sha_file(stdout_path))
    time.sleep(0.1)
    assert (stdout_path.stat().st_size, runner.sha_file(stdout_path)) == before
    assert child.poll() is not None
