#!/usr/bin/env python3
"""Prospective v4 runner with network, process-creation, and launch recovery.

The v3 source remains byte-for-byte unchanged and is pinned by SHA-256. V4
wraps its process-launch and recovery hooks under a new source identity, and
changes the macOS sandbox profile before any preflight or phase runs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import Any
import uuid

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "results/transformers-47468-development-qualification-v4"
PROTOCOL = EXPERIMENT / "protocol/protocol-v4.json"
LOCK = EXPERIMENT / "protocol/protocol-v4.lock.json"
PROMPT = EXPERIMENT / "protocol/candidate-prompt.md"
SCRIPT = Path(__file__).resolve()
CORE_SCRIPT = ROOT / "scripts/run_transformers_47468_qualification_v3.py"
PROFILE = "(version 1) (allow default) (deny network*) (deny process-fork)"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_locked_source_identity(lock_data: dict[str, Any]) -> None:
    """Validate both the v4 envelope and the exact immutable v3 core."""
    require(lock_data.get("protocol_id") == "transformers-47468-generic-scalar-float-development-qualification-v4",
            "source lock protocol identity mismatch")
    require(lock_data.get("runner_path") == "scripts/run_transformers_47468_qualification_v4.py",
            "source lock runner path mismatch")
    require(sha_file(PROTOCOL) == lock_data["protocol_sha256"], "protocol SHA mismatch")
    require(sha_file(SCRIPT) == lock_data["runner_sha256"], "v4 runner SHA mismatch")
    require(lock_data.get("core_runner_path") == "scripts/run_transformers_47468_qualification_v3.py",
            "pinned core runner path mismatch")
    require(sha_file(CORE_SCRIPT) == lock_data["core_runner_sha256"], "immutable v3 core runner SHA mismatch")
    tests_path = ROOT / lock_data["tests_path"]
    require(sha_file(tests_path) == lock_data["tests_sha256"], "test suite SHA mismatch")
    require(sha_file(EXPERIMENT / "README.md") == lock_data["readme_sha256"],
            "qualification README SHA mismatch")


def validate_core_before_import(lock_data: dict[str, Any]) -> None:
    """Reject a changed core before Python executes its module body."""
    require(sha_file(SCRIPT) == lock_data["runner_sha256"], "v4 runner SHA mismatch before core import")
    require(sha_file(PROTOCOL) == lock_data["protocol_sha256"], "protocol SHA mismatch before core import")
    require(lock_data.get("core_runner_path") == "scripts/run_transformers_47468_qualification_v3.py",
            "pinned core runner path mismatch before import")
    require(sha_file(CORE_SCRIPT) == lock_data["core_runner_sha256"],
            "immutable v3 core runner SHA mismatch before import")


def validate_execution_boundary(python: Path) -> dict[str, Any]:
    """Require the actual sandbox to deny networking, fork, and subprocess spawn."""
    probe = r"""
import errno, json, os, socket, subprocess, sys
checks = {}
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    sock.connect(("1.1.1.1", 443))
    checks["network_denied"] = False
except OSError as exc:
    checks["network_denied"] = exc.errno in (errno.EPERM, errno.EACCES)
try:
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    checks["fork_denied"] = False
except OSError as exc:
    checks["fork_denied"] = exc.errno in (errno.EPERM, errno.EACCES)
try:
    subprocess.run([sys.executable, "-c", "pass"], check=False, timeout=2,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    checks["spawn_denied"] = False
except OSError as exc:
    checks["spawn_denied"] = exc.errno in (errno.EPERM, errno.EACCES)
print(json.dumps(checks, sort_keys=True))
"""
    argv = ["/usr/bin/sandbox-exec", "-p", PROFILE, str(python), "-c", probe]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=8)
    require(result.returncode == 0, "sandbox boundary probe did not complete cleanly")
    try:
        checks = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("sandbox boundary probe returned malformed evidence") from exc
    require(checks.get("network_denied") is True, "sandbox network-deny preflight failed")
    require(checks.get("fork_denied") is True, "sandbox process-fork deny preflight failed")
    require(checks.get("spawn_denied") is True, "sandbox subprocess-spawn deny preflight failed")
    checks.update({
        "profile": PROFILE,
        "return_code": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "probe_sha256": hashlib.sha256(probe.encode()).hexdigest(),
    })
    return checks


_bootstrap_lock = json.loads(LOCK.read_text(encoding="utf-8"))
validate_core_before_import(_bootstrap_lock)
_core_spec = importlib.util.spec_from_file_location("vare_transformers_47468_v3_core", CORE_SCRIPT)
require(_core_spec is not None and _core_spec.loader is not None, "cannot load pinned v3 core runner")
core = importlib.util.module_from_spec(_core_spec)
_core_spec.loader.exec_module(core)

# Bind the historical, unchanged transaction/recovery implementation to this
# prospective source identity before exposing its CLI.
core.EXPERIMENT = EXPERIMENT
core.PROTOCOL = PROTOCOL
core.LOCK = LOCK
core.PROMPT = PROMPT
core.SCRIPT = SCRIPT
core.MANIFEST = EXPERIMENT / "records/manifest.jsonl"
core.PROFILE = PROFILE
core.validate_locked_source_identity = validate_locked_source_identity
core.validate_network_deny = validate_execution_boundary

_ORIGINAL_APPEND_EVENT = core.append_event
_ORIGINAL_LOAD_MANIFEST = core.load_manifest


def _append_event_v4(event_type, fields, manifest_path=None):
    target = core.MANIFEST if manifest_path is None else manifest_path
    return _ORIGINAL_APPEND_EVENT(event_type, fields, manifest_path=target)


def _load_manifest_v4(manifest_path=None):
    target = core.MANIFEST if manifest_path is None else manifest_path
    return _ORIGINAL_LOAD_MANIFEST(manifest_path=target)


core.append_event = _append_event_v4
core.load_manifest = _load_manifest_v4


_core_subprocess = types.ModuleType("vare_v4_subprocess")
_core_subprocess.__dict__.update(vars(core.subprocess))
core.subprocess = _core_subprocess
_ORIGINAL_POPEN = core.subprocess.Popen


def _tracked_phase_popen(*args, **kwargs):
    """Persist a unique launch token before creating each grader process."""
    env = kwargs.get("env")
    if isinstance(env, dict):
        home_name = Path(str(env.get("HOME", ""))).name
        if home_name.startswith("home-"):
            token = home_name.removeprefix("home-")
            try:
                uuid.UUID(token)
            except ValueError:
                token = ""
            if token:
                updated_env = dict(env)
                updated_env["VARE_PHASE_INVOCATION_ID"] = token
                kwargs["env"] = updated_env
                command = list(args[0] if args else kwargs["args"])
                command_sha256 = core.sha_bytes(json.dumps(command, separators=(",", ":")).encode())
                core.append_event("phase_launch_token", {
                    "invocation_id": token,
                    "marker_name": "VARE_PHASE_INVOCATION_ID",
                    "marker_value": token,
                    "command_sha256": command_sha256,
                    "recorded_before_process_creation": True,
                })
    return _ORIGINAL_POPEN(*args, **kwargs)


core.subprocess.Popen = _tracked_phase_popen


def _find_processes_for_launch_token(token: str, expected_test: str, expected_cwd: str):
    """Return exact same-user processes carrying the token, or incomplete scan state."""
    current_user = core.psutil.Process(os.getpid()).username()
    matches = []
    scan_complete = True
    started = time.monotonic()
    for process in core.psutil.process_iter():
        if time.monotonic() - started > 10:
            scan_complete = False
            break
        try:
            if process.pid == os.getpid() or process.username() != current_user:
                continue
            environment = process.environ()
        except core.psutil.NoSuchProcess:
            continue
        except (core.psutil.AccessDenied, OSError):
            # An unreadable same-user process could carry the unique marker.
            scan_complete = False
            continue
        if environment.get("VARE_PHASE_INVOCATION_ID") != token:
            continue
        try:
            command = process.cmdline()
            cwd = process.cwd()
        except (core.psutil.NoSuchProcess, core.psutil.AccessDenied, OSError):
            scan_complete = False
            continue
        require(expected_test in command, "launch-token process command differs from the admitted test")
        require(cwd == expected_cwd, "launch-token process working directory differs from the admitted directory")
        matches.append(process)
    return matches, scan_complete


_ORIGINAL_RECOVERY = core.recover_interrupted_attempts


def _recover_interrupted_attempts_with_launch_tokens(manifest_path, run_dir, experiment_root):
    """Recover a launch-window PID from its durable marker before core recovery."""
    rows = core.load_manifest(manifest_path)
    admissions = {row["invocation_id"]: row for row in rows if row.get("event_type") == "phase_admitted"}
    starts = {row["invocation_id"]: row for row in rows if row.get("event_type") == "phase_started"}
    processes = {row["invocation_id"]: row for row in rows if row.get("event_type") == "phase_process_started"}
    tokens = {row["invocation_id"]: row for row in rows if row.get("event_type") == "phase_launch_token"}
    scans = {row["invocation_id"]: row for row in rows if row.get("event_type") == "phase_launch_scan_completed"}
    for invocation_id, started_event in starts.items():
        if invocation_id in processes or invocation_id not in tokens:
            continue
        token_event = tokens[invocation_id]
        admission = admissions.get(invocation_id)
        require(admission is not None, "launch token has no matching phase admission")
        command = started_event["command_argv"]
        command_sha256 = core.sha_bytes(json.dumps(command, separators=(",", ":")).encode())
        require(command_sha256 == admission["command_sha256"] == token_event["command_sha256"],
                "launch token command differs from phase admission")
        matches, scan_complete = _find_processes_for_launch_token(
            token_event["marker_value"], command[-1], started_event["cwd"],
        )
        require(scan_complete, "launch-token process scan incomplete; refusing artifact finalization")
        require(len(matches) <= 1, "multiple processes carry one phase launch token; refusing recovery")
        if matches:
            process = matches[0]
            try:
                process_event = {
                    "invocation_id": invocation_id,
                    "phase": started_event["phase"],
                    "pid": process.pid,
                    "pid_create_time": process.create_time(),
                    "process_group_id": os.getpgid(process.pid),
                    "recovered_from_launch_token": True,
                    "launch_token_event_sha256": token_event["event_sha256"],
                }
            except (core.psutil.NoSuchProcess, ProcessLookupError):
                if invocation_id not in scans:
                    core.append_event("phase_launch_scan_completed", {
                        "invocation_id": invocation_id,
                        "process_count": 0,
                        "scan_complete": True,
                        "reason": "token process exited during identity capture; no phase outcome inferred",
                    }, manifest_path=manifest_path)
                continue
            core.append_event("phase_process_started", process_event, manifest_path=manifest_path)
        elif invocation_id not in scans:
            core.append_event("phase_launch_scan_completed", {
                "invocation_id": invocation_id,
                "process_count": 0,
                "scan_complete": True,
                "reason": "no live same-user process carried the durable launch token",
            }, manifest_path=manifest_path)
    return _ORIGINAL_RECOVERY(manifest_path, run_dir, experiment_root)


core.recover_interrupted_attempts = _recover_interrupted_attempts_with_launch_tokens


def validate_frozen_execution_identity() -> dict[str, str]:
    """Reject changed or dirty source before execute/recover can touch evidence."""
    rows = core.load_manifest()
    require(rows and rows[0].get("event_type") == "protocol_frozen",
            "frozen protocol identity is absent")
    frozen = rows[0]
    lock_data = json.loads(LOCK.read_text(encoding="utf-8"))
    validate_locked_source_identity(lock_data)
    require(sha_file(LOCK) == frozen.get("lock_sha256"),
            "source lock differs from the frozen protocol identity")
    require(sha_file(PROTOCOL) == frozen.get("protocol_sha256"),
            "protocol differs from the frozen protocol identity")
    require(sha_file(SCRIPT) == frozen.get("runner_sha256"),
            "runner differs from the frozen protocol identity")
    commit, tree = core.repo_identity()
    require(commit == frozen.get("protocol_commit"), "checkout commit differs from frozen identity")
    require(tree == frozen.get("protocol_tree"), "checkout tree differs from frozen identity")
    status = core.subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True,
    ).strip()
    require(not status, "worktree must be clean before execute or recovery")
    return {"commit": commit, "tree": tree, "lock_sha256": sha_file(LOCK)}


_ORIGINAL_MAIN = core.main


def _main_v4() -> int:
    if "--execute" in sys.argv or "--recover-interrupted" in sys.argv:
        validate_frozen_execution_identity()
    return _ORIGINAL_MAIN()


core.main = _main_v4


if __name__ == "__main__":
    raise SystemExit(core.main())
