#!/usr/bin/env python3
"""Versioned v3 one-shot no-network qualification runner for Transformers #47468."""
from __future__ import annotations

import argparse
import base64
import datetime as dt
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
import uuid
from typing import Any

import fcntl

import psutil

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "results/transformers-47468-development-qualification-v3"
PROTOCOL = EXPERIMENT / "protocol/protocol-v3.json"
LOCK = EXPERIMENT / "protocol/protocol-v3.lock.json"
PROMPT = EXPERIMENT / "protocol/candidate-prompt.md"
SCRIPT = Path(__file__).resolve()
MANIFEST = EXPERIMENT / "records/manifest.jsonl"
PROFILE = "(version 1) (allow default) (deny network*)"
TARGET = "src/transformers/utils/generic.py"
OLD_MUTANT_EXPR = (
    b"return x.to(torch.float32) if torch.jit.is_tracing() "
    b"and isinstance(x, torch.Tensor) else float(x)"
)
NEW_MUTANT_EXPR = (
    b"return x.to(torch.float32) if torch.jit.is_tracing() "
    b"and isinstance(x, torch.Tensor) else int(x)"
)
SAMPLE_INTERVAL = 0.5
MAX_SECONDS = 180
MIN_AVAILABLE = 8 * 1024**3
MIN_FREE_SWAP = 256 * 1024**2
MAX_SWAP_GROWTH = 128 * 1024**2
MIN_DISK = 20 * 1024**3
MAX_TREE_RSS = 8 * 1024**3


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_new(path: Path, data: bytes) -> None:
    """Atomically create and fsync a file without replacing any existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        os.unlink(temp)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def write_json_new(path: Path, value: Any) -> None:
    write_new(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def event_hash(event: dict[str, Any]) -> str:
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return sha_bytes(encoded)


def _manifest_pending_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(manifest_path.name + ".pending.json")


@contextmanager
def _manifest_lock(manifest_path: Path):
    lock_path = manifest_path.with_name(manifest_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _parse_complete_manifest(data: bytes) -> list[dict[str, Any]]:
    if not data:
        return []
    require(data.endswith(b"\n"), "manifest has a torn final line without a matching pending-event record")
    rows = []
    previous = "0" * 64
    for sequence, raw_line in enumerate(data.splitlines(keepends=True), start=1):
        require(raw_line.endswith(b"\n"), "manifest contains a non-final partial line")
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"manifest line {sequence} is invalid JSON; preserving journal for audit") from exc
        require(row.get("sequence") == sequence, f"manifest sequence is not contiguous at line {sequence}")
        require(row.get("previous_event_sha256") == previous, f"manifest hash chain is broken at line {sequence}")
        require(row.get("event_sha256") == event_hash(row), f"manifest event hash mismatch at line {sequence}")
        previous = row["event_sha256"]
        rows.append(row)
    return rows


def _verified_prefix_and_suffix(data: bytes) -> tuple[list[dict[str, Any]], bytes, bytes]:
    """Return complete verified rows, exact prefix bytes, and unterminated suffix bytes."""
    if not data:
        return [], b"", b""
    if data.endswith(b"\n"):
        return _parse_complete_manifest(data), data, b""
    split = data.rfind(b"\n") + 1
    prefix, suffix = data[:split], data[split:]
    return _parse_complete_manifest(prefix), prefix, suffix


def _pending_record(row: dict[str, Any], event_line: bytes, prior_prefix: bytes) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "sequence": row["sequence"],
        "previous_event_sha256": row["previous_event_sha256"],
        "prior_prefix_sha256": sha_bytes(prior_prefix),
        "event_sha256": row["event_sha256"],
        "event_line_sha256": sha_bytes(event_line),
        "event_line_base64": base64.b64encode(event_line).decode("ascii"),
    }
    body["pending_record_sha256"] = sha_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    return body


def _verify_pending_record(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        pending = json.loads(path.read_text(encoding="utf-8"))
        signature = pending.pop("pending_record_sha256")
        require(signature == sha_bytes(json.dumps(pending, sort_keys=True, separators=(",", ":")).encode()),
                "pending-event record hash mismatch")
        pending["pending_record_sha256"] = signature
        event_line = base64.b64decode(pending["event_line_base64"], validate=True)
        require(event_line.endswith(b"\n"), "pending event line lacks final newline")
        require(sha_bytes(event_line) == pending["event_line_sha256"], "pending event line hash mismatch")
        row = json.loads(event_line)
        require(row.get("event_sha256") == pending["event_sha256"] == event_hash(row),
                "pending event identity/hash mismatch")
        require(row.get("sequence") == pending["sequence"], "pending event sequence mismatch")
        require(row.get("previous_event_sha256") == pending["previous_event_sha256"],
                "pending event predecessor mismatch")
        return pending, event_line
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("pending-event record is invalid; refusing journal repair") from exc


def _write_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.replace")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _append_bytes_fsync(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            require(written > 0, "manifest append returned a zero-byte write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_pending(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _append_row_locked(manifest_path: Path, rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    current = manifest_path.read_bytes() if manifest_path.exists() else b""
    require(not current or current.endswith(b"\n"), "cannot append while manifest has an unresolved tail")
    event_line = (json.dumps(row, sort_keys=True) + "\n").encode()
    pending_path = _manifest_pending_path(manifest_path)
    require(not pending_path.exists(), "pending-event record exists; reconcile it before a new append")
    write_json_new(pending_path, _pending_record(row, event_line, current))
    _append_bytes_fsync(manifest_path, event_line)
    _remove_pending(pending_path)


def _committed_pending_event(
    data: bytes, pending: dict[str, Any], event_line: bytes,
) -> list[dict[str, Any]] | None:
    """Recognize a pending intent already committed, including atomic tail repair."""
    if not data.endswith(b"\n"):
        return None
    rows = _parse_complete_manifest(data)
    sequence = pending["sequence"]
    if len(rows) < sequence:
        return None
    lines = data.splitlines(keepends=True)
    prior_prefix = b"".join(lines[:sequence - 1])
    actual_line = lines[sequence - 1]
    require(actual_line == event_line, "manifest event at pending sequence differs from durable intent")
    require(sha_bytes(prior_prefix) == pending["prior_prefix_sha256"],
            "pending-event prior prefix mismatch; refusing to alter verified journal")
    if len(rows) > sequence:
        recovery_row = rows[sequence]
        require(
            recovery_row.get("event_type") == "journal_tail_repaired"
            and recovery_row.get("pending_event_sha256") == pending["event_sha256"],
            "pending event has unexpected successors; refusing to reconcile journal",
        )
    return rows


def _repair_pending_tail_locked(manifest_path: Path, data: bytes, pending_path: Path) -> list[dict[str, Any]]:
    pending, event_line = _verify_pending_record(pending_path)
    committed = _committed_pending_event(data, pending, event_line)
    if committed is not None:
        _remove_pending(pending_path)
        return committed
    rows, prefix, suffix = _verified_prefix_and_suffix(data)
    previous = rows[-1]["event_sha256"] if rows else "0" * 64
    require(sha_bytes(prefix) == pending["prior_prefix_sha256"],
            "pending-event prior prefix mismatch; refusing to alter verified journal")
    require(pending["sequence"] == len(rows) + 1 and pending["previous_event_sha256"] == previous,
            "pending-event sequence/predecessor mismatch; refusing journal repair")

    # A complete event plus a pending file means append succeeded before cleanup.
    if not suffix and data.endswith(b"\n"):
        if rows and rows[-1].get("event_sha256") == pending["event_sha256"]:
            require(data.endswith(event_line), "committed pending event bytes differ from durable intent")
            _remove_pending(pending_path)
            return rows
        # The intent was durable but no bytes were appended; recover the event explicitly below.

    require(event_line.startswith(suffix),
            "damaged manifest suffix does not match the durable pending-event prefix")
    suffix_sha = sha_bytes(suffix)
    quarantine = manifest_path.with_name(
        f"{manifest_path.name}.quarantine-{pending['sequence']}-{suffix_sha}.suffix"
    )
    if quarantine.exists():
        require(quarantine.read_bytes() == suffix, "quarantined suffix path contains different bytes")
    else:
        write_new(quarantine, suffix)

    pending_event = json.loads(event_line)
    repair_row = {
        "sequence": len(rows) + 2,
        "event_type": "journal_tail_repaired",
        "timestamp_utc": utc_now(),
        "previous_event_sha256": pending_event["event_sha256"],
        "repair_type": "completed_durable_pending_event_after_interrupted_append",
        "pending_record_sha256": pending["pending_record_sha256"],
        "pending_event_sha256": pending["event_sha256"],
        "damaged_suffix_path": quarantine.name,
        "damaged_suffix_sha256": suffix_sha,
        "damaged_suffix_bytes": len(suffix),
        "verified_prior_event_count": len(rows),
        "verified_prior_prefix_sha256": sha_bytes(prefix),
    }
    repair_row["event_sha256"] = event_hash(repair_row)
    repair_line = (json.dumps(repair_row, sort_keys=True) + "\n").encode()
    _write_replace(manifest_path, prefix + event_line + repair_line)
    _remove_pending(pending_path)
    return _parse_complete_manifest(prefix + event_line + repair_line)


def _load_manifest_locked(manifest_path: Path) -> list[dict[str, Any]]:
    pending_path = _manifest_pending_path(manifest_path)
    if not manifest_path.exists() and not pending_path.exists():
        return []
    data = manifest_path.read_bytes() if manifest_path.exists() else b""
    if pending_path.exists():
        return _repair_pending_tail_locked(manifest_path, data, pending_path)
    return _parse_complete_manifest(data)


def load_manifest(manifest_path: Path = MANIFEST) -> list[dict[str, Any]]:
    with _manifest_lock(manifest_path):
        return _load_manifest_locked(manifest_path)


def validate_locked_source_identity(lock_data: dict[str, Any]) -> None:
    require(sha_file(PROTOCOL) == lock_data["protocol_sha256"], "protocol SHA mismatch")
    require(sha_file(SCRIPT) == lock_data["runner_sha256"], "runner SHA mismatch")
    tests_path = ROOT / lock_data["tests_path"]
    require(sha_file(tests_path) == lock_data["tests_sha256"], "test suite SHA mismatch")
    require(sha_file(EXPERIMENT / "README.md") == lock_data["readme_sha256"], "qualification README SHA mismatch")


def validate_recovery_identity(
    manifest_path: Path, *, protocol_commit: str, protocol_tree: str,
    protocol_sha256: str, runner_sha256: str, lock_sha256: str,
) -> None:
    data = manifest_path.read_bytes() if manifest_path.exists() else b""
    rows, _, _ = _verified_prefix_and_suffix(data)
    if rows:
        frozen = rows[0]
    else:
        pending_path = _manifest_pending_path(manifest_path)
        require(pending_path.is_file(), "frozen identity is absent from manifest and pending journal")
        _, event_line = _verify_pending_record(pending_path)
        frozen = json.loads(event_line)
    require(frozen.get("event_type") == "protocol_frozen", "manifest does not start with protocol_frozen")
    require(frozen.get("protocol_commit") == protocol_commit, "recovery checkout commit differs from frozen identity")
    require(frozen.get("protocol_tree") == protocol_tree, "recovery checkout tree differs from frozen identity")
    require(frozen.get("protocol_sha256") == protocol_sha256, "recovery protocol differs from frozen identity")
    require(frozen.get("runner_sha256") == runner_sha256, "recovery runner differs from frozen identity")
    require(frozen.get("lock_sha256") == lock_sha256, "recovery source lock differs from frozen identity")


def append_event(
    event_type: str, fields: dict[str, Any], manifest_path: Path = MANIFEST,
) -> dict[str, Any]:
    with _manifest_lock(manifest_path):
        rows = _load_manifest_locked(manifest_path)
        previous = rows[-1]["event_sha256"] if rows else "0" * 64
        row = {
            "sequence": len(rows) + 1,
            "event_type": event_type,
            "timestamp_utc": utc_now(),
            "previous_event_sha256": previous,
            **fields,
        }
        row["event_sha256"] = event_hash(row)
        _append_row_locked(manifest_path, rows, row)
        return row


def _run_identity_command(python: Path, argv: list[str], run=subprocess.run) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_NO_INDEX": "1",
        "PIP_CONFIG_FILE": "/dev/null",
    }
    return run([str(python), *argv], capture_output=True, text=True, env=env, timeout=30)


def validate_runtime_identity(
    python: Path,
    overlay: Path,
    expected: dict[str, Any],
    host_lock_path: Path,
    overlay_lock_path: Path,
    run=subprocess.run,
) -> dict[str, Any]:
    """Fail closed unless the live interpreter and both package inventories match exactly."""
    probe = (
        "import json,sys; "
        "print(json.dumps({'executable':sys.executable,'sys_version':sys.version,"
        "'version_info':list(sys.version_info[:3]),'implementation':sys.implementation.name,"
        "'cache_tag':sys.implementation.cache_tag},sort_keys=True))"
    )
    identity_result = _run_identity_command(python, ["-c", probe], run=run)
    require(identity_result.returncode == 0, "runtime Python identity probe failed")
    try:
        identity = json.loads(identity_result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("runtime Python identity probe returned invalid JSON") from exc
    identity["executable"] = str(Path(identity["executable"]).resolve())
    expected_identity = dict(expected)
    expected_identity["executable"] = str(Path(expected_identity["executable"]).resolve())
    require(identity == expected_identity, f"runtime Python identity drift: {identity!r}")

    inventory_commands = {
        "host": (["-m", "pip", "freeze", "--all"], host_lock_path),
        "overlay": (["-m", "pip", "freeze", "--all", "--path", str(overlay)], overlay_lock_path),
    }
    observations: dict[str, Any] = {"python_identity": identity, "inventories": {}}
    for label, (argv, lock_path) in inventory_commands.items():
        result = _run_identity_command(python, argv, run=run)
        require(result.returncode == 0, f"{label} dependency inventory command failed")
        expected_bytes = lock_path.read_bytes()
        actual_bytes = result.stdout.encode()
        observations["inventories"][label] = {
            "expected_sha256": sha_bytes(expected_bytes),
            "actual_sha256": sha_bytes(actual_bytes),
            "expected_lines": len(expected_bytes.splitlines()),
            "actual_lines": len(actual_bytes.splitlines()),
        }
        require(actual_bytes == expected_bytes, f"{label} dependency inventory differs from frozen identity")
    return observations


def prior_attempt_exists(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("event_type") in {
        "phase_admitted", "phase_started", "phase_process_started", "phase_record",
        "phase_interrupted_recovered", "phase_record_recovered",
    } for row in rows)


def require_no_prior_attempt(rows: list[dict[str, Any]]) -> None:
    require(not prior_attempt_exists(rows), "a phase attempt is already admitted; this protocol is one-shot and cannot rerun")


def _artifact_observation(run_dir: Path, relpath: str) -> dict[str, Any]:
    candidate = (run_dir / relpath).resolve()
    require(candidate.is_relative_to(run_dir.resolve()), "manifest artifact path escapes run directory")
    if not candidate.is_file():
        return {"path": relpath, "exists": False, "bytes": None, "sha256": None}
    fd = os.open(candidate, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"path": relpath, "exists": True, "bytes": candidate.stat().st_size, "sha256": sha_file(candidate)}


def _group_has_live_processes(process_group_id: int) -> bool:
    for process in psutil.process_iter(attrs=["pid", "status"]):
        try:
            if os.getpgid(process.pid) != process_group_id:
                continue
            if process.info.get("status") != psutil.STATUS_ZOMBIE:
                return True
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
        except (PermissionError, psutil.AccessDenied) as exc:
            raise RuntimeError("cannot establish process-group quiescence") from exc
    return False


def quiesce_recorded_process(
    process_event: dict[str, Any], timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Verify PID/create-time, then terminate and wait for the recorded isolated group."""
    pid = int(process_event["pid"])
    expected_create_time = float(process_event["pid_create_time"])
    expected_group = int(process_event["process_group_id"])
    require(expected_group == pid, "recorded process group is not the isolated leader PID")
    process = None
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        process = None
    if process is not None:
        actual_create_time = process.create_time()
        require(abs(actual_create_time - expected_create_time) <= 0.01,
                "recorded PID create-time mismatch; refusing to signal possible PID reuse")
        try:
            actual_group = os.getpgid(pid)
        except ProcessLookupError:
            actual_group = None
        require(actual_group in (None, expected_group), "recorded process group changed; refusing to signal")
        try:
            if process.status() == psutil.STATUS_ZOMBIE and not _group_has_live_processes(expected_group):
                return {
                    "status": "leader_zombie_group_quiescent",
                    "pid": pid,
                    "pid_create_time": expected_create_time,
                    "process_group_id": expected_group,
                    "signals_sent": [],
                }
        except psutil.NoSuchProcess:
            pass
    elif not _group_has_live_processes(expected_group):
        return {
            "status": "leader_absent_group_quiescent",
            "pid": pid,
            "pid_create_time": expected_create_time,
            "process_group_id": expected_group,
            "signals_sent": [],
        }

    signals_sent = []
    try:
        os.killpg(expected_group, signal.SIGTERM)
        signals_sent.append("SIGTERM")
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and _group_has_live_processes(expected_group):
        time.sleep(0.02)
    if _group_has_live_processes(expected_group):
        os.killpg(expected_group, signal.SIGKILL)
        signals_sent.append("SIGKILL")
        kill_deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < kill_deadline and _group_has_live_processes(expected_group):
            time.sleep(0.02)
    require(not _group_has_live_processes(expected_group),
            "recorded process group remains active after bounded termination")
    return {
        "status": "terminated_after_" + (signals_sent[-1] if signals_sent else "exit"),
        "pid": pid,
        "pid_create_time": expected_create_time,
        "process_group_id": expected_group,
        "signals_sent": signals_sent,
    }


def wait_and_quiesce_phase_process(
    process: subprocess.Popen, process_event: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Reap the phase leader and prove its recorded process group cannot write."""
    return_code = process.wait()
    quiescence = quiesce_recorded_process(process_event)
    return return_code, quiescence


def terminate_launched_process_group(
    process: subprocess.Popen, timeout_seconds: float = 2.0,
) -> None:
    """Stop a just-launched isolated group if its identity event cannot be persisted."""
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=timeout_seconds)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and _group_has_live_processes(process_group_id):
        time.sleep(0.02)
    if _group_has_live_processes(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline and _group_has_live_processes(process_group_id):
            time.sleep(0.02)
    require(not _group_has_live_processes(process_group_id),
            "unrecorded launched process group remains active after bounded termination")


def record_phase_process_start(
    process: subprocess.Popen, invocation_id: str, phase: str,
) -> dict[str, Any]:
    try:
        process_event = {
            "invocation_id": invocation_id,
            "phase": phase,
            "pid": process.pid,
            "pid_create_time": psutil.Process(process.pid).create_time(),
            "process_group_id": os.getpgid(process.pid),
        }
        append_event("phase_process_started", process_event)
    except BaseException:
        # The process group is known from Popen(start_new_session=True), even
        # if collecting or durably publishing its richer identity failed.
        terminate_launched_process_group(process)
        raise
    return process_event


def recover_interrupted_attempts(
    manifest_path: Path, run_dir: Path, experiment_root: Path,
) -> list[dict[str, Any]]:
    """Reconstruct unterminated attempts from append-only events and partial files; never execute work."""
    rows = load_manifest(manifest_path)
    admissions = {row["invocation_id"]: row for row in rows if row.get("event_type") == "phase_admitted"}
    starts = {row["invocation_id"]: row for row in rows if row.get("event_type") == "phase_started"}
    processes = {row["invocation_id"]: row for row in rows if row.get("event_type") == "phase_process_started"}
    terminal_ids = {
        row.get("invocation_id") for row in rows
        if row.get("event_type") in {"phase_record", "phase_interrupted_recovered", "phase_record_recovered"}
    }
    deferred_by_id = {
        row["invocation_id"]: row for row in rows if row.get("event_type") == "phase_recovery_deferred"
    }
    recovered = []
    for invocation_id, admission in admissions.items():
        if invocation_id in terminal_ids:
            continue
        start = starts.get(invocation_id)
        process = processes.get(invocation_id)
        if start and not process:
            deferred = deferred_by_id.get(invocation_id)
            if deferred is None:
                deferred = append_event("phase_recovery_deferred", {
                    "invocation_id": invocation_id,
                    "phase": admission["phase"],
                    "reason": "phase_started exists but process PID/create-time record is absent; writer liveness cannot be established",
                    "artifact_hashes_finalized": False,
                }, manifest_path=manifest_path)
            recovered.append({"deferred": True, "event": deferred})
            continue

        quiescence = None
        if process:
            try:
                quiescence = quiesce_recorded_process(process)
            except (RuntimeError, OSError) as exc:
                deferred = deferred_by_id.get(invocation_id)
                if deferred is None or deferred.get("reason") != str(exc):
                    deferred = append_event("phase_recovery_deferred", {
                        "invocation_id": invocation_id,
                        "phase": admission["phase"],
                        "reason": str(exc),
                        "artifact_hashes_finalized": False,
                    }, manifest_path=manifest_path)
                recovered.append({"deferred": True, "event": deferred})
                continue

        artifacts = admission.get("artifacts", {})
        observations = {name: _artifact_observation(run_dir, path) for name, path in artifacts.items()}
        existing_record = observations.get("record", {})
        if existing_record.get("exists"):
            raw_path = run_dir / existing_record["path"]
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            event_type = "phase_record_recovered"
            record = {
                "schema_version": 2,
                "status": "RECOVERED_TERMINAL_RECORD",
                "invocation_id": invocation_id,
                "phase": admission["phase"],
                "original_record_path": existing_record["path"],
                "original_record_sha256": existing_record["sha256"],
                "original_record_status": raw.get("status"),
                "original_record_expected_outcome_matched": raw.get("expected_outcome_matched"),
                "artifact_observations": observations,
                "process_quiescence": quiescence,
                "reconstructed_at_utc": utc_now(),
            }
        else:
            event_type = "phase_interrupted_recovered"
            record = {
                "schema_version": 2,
                "status": (
                    "INTERRUPTED_AFTER_PROCESS_START_RECONSTRUCTED" if process
                    else "INTERRUPTED_LAUNCH_STATE_UNCERTAIN_RECONSTRUCTED" if start
                    else "INTERRUPTED_BEFORE_START_RECONSTRUCTED"
                ),
                "invocation_id": invocation_id,
                "phase": admission["phase"],
                "phase_order_index": admission["phase_order_index"],
                "source_identity": admission.get("source_identity"),
                "command_sha256": admission.get("command_sha256"),
                "started_event": start,
                "process_started_event": process,
                "process_quiescence": quiescence,
                "artifact_observations": observations,
                "reconstructed_at_utc": utc_now(),
                "interpretation": "No grade or test outcome is inferred from incomplete logs. Process-group quiescence was established before artifact fsync and hashing; this invocation is terminal for the one-shot protocol and cannot be rerun.",
            }
        record_path = experiment_root / "records" / f"recovered-phase-{invocation_id}.json"
        if record_path.exists():
            retained = json.loads(record_path.read_text(encoding="utf-8"))
            require(
                retained.get("invocation_id") == invocation_id
                and retained.get("phase") == admission["phase"]
                and retained.get("artifact_observations") == observations,
                "existing recovery record does not match current interrupted attempt evidence",
            )
            record = retained
            event_type = (
                "phase_record_recovered" if record.get("status") == "RECOVERED_TERMINAL_RECORD"
                else "phase_interrupted_recovered"
            )
        else:
            write_json_new(record_path, record)
        row = append_event(event_type, {
            "invocation_id": invocation_id,
            "phase": admission["phase"],
            "recovery_record_path": str(record_path.relative_to(experiment_root)),
            "recovery_record_sha256": sha_file(record_path),
            "artifact_observations": observations,
        }, manifest_path=manifest_path)
        recovered.append({"record": record, "event": row})
    return recovered


def tree_hash(root: Path) -> tuple[int, int, str]:
    base = root.parent.parent
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        rel = path.relative_to(base).as_posix().encode()
        raw = path.read_bytes()
        total_bytes += len(raw)
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        digest.update(hashlib.sha256(raw).digest())
    return len(files), total_bytes, digest.hexdigest()


def make_mutant(fix_bytes: bytes) -> bytes:
    require(fix_bytes.count(OLD_MUTANT_EXPR) == 1, "fixed target does not match the frozen mutant transform")
    mutant = fix_bytes.replace(OLD_MUTANT_EXPR, NEW_MUTANT_EXPR, 1)
    require(sha_bytes(mutant) == PROTOCOL_DATA["task"]["mutant"]["sha256"], "mutant hash differs from protocol")
    return mutant


def resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def validate_network_deny(python: Path) -> dict[str, Any]:
    argv = [
        "/usr/bin/sandbox-exec", "-p", PROFILE, str(python), "-c",
        "import socket; s=socket.socket(); s.settimeout(1); s.connect(('1.1.1.1',443))",
    ]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=5)
    output = result.stdout + result.stderr
    denied = result.returncode != 0 and ("Operation not permitted" in output or "PermissionError" in output)
    require(denied, "sandbox network-deny preflight failed")
    return {"return_code": result.returncode, "denied": True, "output_sha256": sha_bytes(output.encode())}


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    require(PROTOCOL.is_file() and LOCK.is_file(), "protocol or lock is missing")
    protocol_raw = PROTOCOL.read_bytes()
    lock_data = json.loads(LOCK.read_text(encoding="utf-8"))
    validate_locked_source_identity(lock_data)
    protocol = json.loads(protocol_raw)
    require(sha_file(PROMPT) == protocol["task"]["candidate_prompt_sha256"], "candidate prompt SHA mismatch")

    archive = resolve(args.source_archive)
    grader_test = resolve(args.grader_test)
    fix_target = resolve(args.fix_target)
    python = resolve(args.python)
    overlay = resolve(args.overlay_site_packages)
    output_root = resolve(args.output)
    host_lock = EXPERIMENT / "locks/host-pip-freeze.txt"
    overlay_lock = EXPERIMENT / "locks/overlay-pip-freeze.txt"
    exposure = EXPERIMENT / "protocol/exposure-scan.json"
    preflight_record = EXPERIMENT / "records/runtime-inventory-observation-v3.json"
    require(archive.is_file() and sha_file(archive) == protocol["task"]["candidate_source_archive_sha256"], "baseline source archive missing or changed")
    require(grader_test.is_file() and sha_file(grader_test) == protocol["task"]["test_sha256"], "pinned grader test missing or changed")
    require(fix_target.is_file() and sha_file(fix_target) == protocol["task"]["fix_target_sha256"], "exact fix target missing or changed")
    require(sha_file(host_lock) == protocol["execution"]["dependency_freeze_sha256"]["locks/host-pip-freeze.txt"], "host Python freeze changed")
    require(sha_file(overlay_lock) == protocol["execution"]["dependency_freeze_sha256"]["locks/overlay-pip-freeze.txt"], "overlay Python freeze changed")
    require(sha_file(exposure) == protocol["exposure"]["local_vare_scan_sha256"], "exposure scan changed")
    require(sha_file(preflight_record) == lock_data["preflight_record_sha256"], "pre-outcome preflight record changed")
    require(python.is_file() and overlay.is_dir(), "pinned Python executable or package overlay missing")
    require(not grader_test.is_relative_to(output_root / "workspaces"),
            "grader test must remain outside candidate source workspaces")
    expected_python = Path(protocol["execution"]["python_identity"]["executable"]).expanduser().resolve()
    require(python == expected_python, "Python executable differs from frozen protocol")
    expected_overlay = Path(protocol["execution"]["overlay_site_packages"]).expanduser().resolve()
    require(overlay == expected_overlay, "overlay site-packages path differs from frozen protocol")
    runtime_identity = validate_runtime_identity(
        python, overlay, protocol["execution"]["python_identity"], host_lock, overlay_lock,
    )
    with tarfile.open(archive, mode="r:") as tar:
        names = tar.getnames()
        require(names and all(name in {"src", "src/transformers"} or name.startswith("src/transformers/") for name in names), "source archive contains paths outside src/transformers")
        require(not any("/tests/" in f"/{name}/" or ".git" in name.split("/") for name in names), "source archive contains tests or Git history")
        require(not any(member.issym() or member.islnk() for member in tar.getmembers()), "source archive contains symlinks")
        scratch = Path(tempfile.mkdtemp(prefix="vare-transformers-47468-preflight-"))
        try:
            tar.extractall(scratch, filter="data")
            source_root = scratch / "src" / "transformers"
            count, size, digest = tree_hash(source_root)
            expected_task = protocol["task"]
            require((count, size, digest) == (
                expected_task["candidate_source_file_count"],
                expected_task["candidate_source_bytes"],
                expected_task["candidate_source_tree_sha256"],
            ), "source tree inventory differs from the frozen protocol")
            base_target = source_root.parent / "transformers/utils/generic.py"
            require(sha_file(base_target) == expected_task["base_target_sha256"], "baseline target differs from its upstream base hash")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    fix_bytes = fix_target.read_bytes()
    mutant_bytes = make_mutant(fix_bytes)
    network_check = validate_network_deny(python)
    return {
        "protocol_sha256": lock_data["protocol_sha256"],
        "runner_sha256": lock_data["runner_sha256"],
        "source_archive_sha256": sha_file(archive),
        "source_tree_sha256": protocol["task"]["candidate_source_tree_sha256"],
        "grader_test_sha256": sha_file(grader_test),
        "fix_target_sha256": sha_file(fix_target),
        "mutant_target_sha256": sha_bytes(mutant_bytes),
        "network_deny_probe": network_check,
        "python": str(python),
        "overlay_site_packages": str(overlay),
        "runtime_identity": runtime_identity,
    }


def resource_sample(pid: int | None = None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage(str(ROOT))
    tree_rss = 0
    if pid is not None:
        try:
            process = psutil.Process(pid)
            tree_rss = sum(item.memory_info().rss for item in [process, *process.children(recursive=True)] if item.is_running())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return {
        "timestamp_utc": utc_now(),
        "available_memory_bytes": memory.available,
        "total_memory_bytes": memory.total,
        "free_swap_bytes": swap.free,
        "swap_used_bytes": swap.used,
        "disk_free_bytes": disk.free,
        "process_tree_rss_bytes": tree_rss,
    }


def test_matches(record: dict[str, Any], expected_pass: bool) -> bool:
    if record["status"] != "EXITED":
        return False
    output = Path(record["stdout_path"]).read_text(errors="replace") + Path(record["stderr_path"]).read_text(errors="replace")
    if expected_pass:
        return record["return_code"] == 0 and re.search(r"\b1 passed\b", output) is not None
    return (
        record["return_code"] == 1
        and "GenericTester::test_torch_float" in output
        and re.search(r"\b1 failed\b", output) is not None
        and "AssertionError" in output
        and "ERROR collecting" not in output
    )


def run_phase(
    phase: str, expected_pass: bool, target_bytes: bytes, args: argparse.Namespace,
    protocol: dict[str, Any], run_dir: Path, validation: dict[str, Any],
) -> dict[str, Any]:
    phase_id = str(uuid.uuid4())
    phase_dir = run_dir / "workspaces" / f"{phase}-{phase_id}"
    phase_dir.mkdir(parents=True, exist_ok=False)
    with tarfile.open(resolve(args.source_archive), mode="r:") as tar:
        tar.extractall(phase_dir, filter="data")
    source_root = phase_dir / "src" / "transformers"
    target = phase_dir / TARGET
    source_tree_base = tree_hash(source_root)
    require(source_tree_base == (
        protocol["task"]["candidate_source_file_count"],
        protocol["task"]["candidate_source_bytes"],
        protocol["task"]["candidate_source_tree_sha256"],
    ), f"{phase}: reconstructed source inventory mismatch")
    target.write_bytes(target_bytes)
    target_hash = sha_file(target)
    source_tree_variant_before = tree_hash(source_root)
    expected_target = {
        "baseline": protocol["task"]["base_target_sha256"],
        "mutant": protocol["task"]["mutant"]["sha256"],
        "fix": protocol["task"]["fix_target_sha256"],
    }[phase]
    require(target_hash == expected_target, f"{phase}: target source hash mismatch")

    logs = run_dir / "logs"
    resources = run_dir / "resources"
    records = run_dir / "records"
    for folder in (logs, resources, records):
        folder.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{phase}-{phase_id}.stdout.log"
    stderr_path = logs / f"{phase}-{phase_id}.stderr.log"
    samples_path = resources / f"{phase}-{phase_id}.jsonl"
    record_path = records / f"{phase}-{phase_id}.json"
    grader_test = resolve(args.grader_test)
    overlay = resolve(args.overlay_site_packages)
    python = resolve(args.python)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str((run_dir / "tmp" / f"home-{phase_id}").resolve()),
        "TMPDIR": str((run_dir / "tmp" / f"tmp-{phase_id}").resolve()),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PIP_NO_INDEX": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((str(source_root.parent.resolve()), str(overlay))),
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "",
    }
    Path(env["HOME"]).mkdir(parents=True, exist_ok=False)
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=False)
    command = [
        "/usr/bin/sandbox-exec", "-p", PROFILE, str(python), "-m", "pytest", "-q",
        "-p", "no:cacheprovider",
        f"{grader_test}::GenericTester::test_torch_float",
    ]
    artifacts = {
        "stdout": str(stdout_path.relative_to(run_dir)),
        "stderr": str(stderr_path.relative_to(run_dir)),
        "resources": str(samples_path.relative_to(run_dir)),
        "record": str(record_path.relative_to(run_dir)),
    }
    source_identity = {
        "base_commit": protocol["task"]["base_commit"],
        "target_path": TARGET,
        "target_sha256": target_hash,
        "source_tree_sha256": protocol["task"]["candidate_source_tree_sha256"],
    }
    pre = resource_sample()
    baseline_swap = pre["swap_used_bytes"]
    min_available = pre["available_memory_bytes"]
    min_free_swap = pre["free_swap_bytes"]
    max_swap = baseline_swap
    min_disk_free = pre["disk_free_bytes"]
    max_rss = 0
    abort_reason = None
    started = time.monotonic()
    started_utc = utc_now()
    return_code = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr, samples_path.open("xb") as samples:
        for stream in (stdout, stderr, samples):
            stream.flush()
            os.fsync(stream.fileno())
        append_event("phase_admitted", {
            "invocation_id": phase_id,
            "phase": phase,
            "phase_order_index": {"baseline": 0, "mutant": 1, "fix": 2}[phase],
            "expected_pass": expected_pass,
            "source_identity": source_identity,
            "runtime_identity": validation["runtime_identity"],
            "command_sha256": sha_bytes(json.dumps(command, separators=(",", ":")).encode()),
            "artifacts": artifacts,
        })
        append_event("phase_started", {
            "invocation_id": phase_id,
            "phase": phase,
            "started_at_utc": started_utc,
            "command_argv": command,
            "cwd": str(grader_test.parent),
            "environment": env,
            "environment_sha256": sha_bytes(json.dumps(env, sort_keys=True, separators=(",", ":")).encode()),
            "artifacts": artifacts,
        })

        def sample_and_write(pid: int | None) -> dict[str, Any]:
            nonlocal min_available, min_free_swap, max_swap, min_disk_free, max_rss
            row = resource_sample(pid)
            min_available = min(min_available, row["available_memory_bytes"])
            min_free_swap = min(min_free_swap, row["free_swap_bytes"])
            max_swap = max(max_swap, row["swap_used_bytes"])
            min_disk_free = min(min_disk_free, row["disk_free_bytes"])
            max_rss = max(max_rss, row["process_tree_rss_bytes"])
            samples.write((json.dumps(row, sort_keys=True) + "\n").encode())
            samples.flush()
            os.fsync(samples.fileno())
            for stream in (stdout, stderr):
                stream.flush()
                os.fsync(stream.fileno())
            return row

        if pre["available_memory_bytes"] < MIN_AVAILABLE:
            abort_reason = "preflight_available_memory_below_floor"
        elif pre["free_swap_bytes"] < MIN_FREE_SWAP:
            abort_reason = "preflight_free_swap_below_floor"
        elif pre["disk_free_bytes"] < MIN_DISK:
            abort_reason = "preflight_disk_below_floor"
        proc = None
        if abort_reason is None:
            proc = subprocess.Popen(command, cwd=grader_test.parent, env=env,
                                    stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
            process_event = record_phase_process_start(proc, phase_id, phase)
            while proc.poll() is None:
                row = sample_and_write(proc.pid)
                growth = row["swap_used_bytes"] - baseline_swap
                elapsed = time.monotonic() - started
                if row["available_memory_bytes"] < MIN_AVAILABLE:
                    abort_reason = "available_memory_below_floor"
                elif row["free_swap_bytes"] < MIN_FREE_SWAP:
                    abort_reason = "free_swap_below_floor"
                elif growth > MAX_SWAP_GROWTH:
                    abort_reason = "swap_growth_above_limit"
                elif row["disk_free_bytes"] < MIN_DISK:
                    abort_reason = "disk_below_floor"
                elif row["process_tree_rss_bytes"] > MAX_TREE_RSS:
                    abort_reason = "process_tree_rss_above_limit"
                elif elapsed > MAX_SECONDS:
                    abort_reason = "phase_timeout"
                if abort_reason:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    break
                time.sleep(SAMPLE_INTERVAL)
            return_code, process_quiescence = wait_and_quiesce_phase_process(
                proc, process_event,
            )
            if process_quiescence["signals_sent"] and abort_reason is None:
                abort_reason = "orphaned_process_group_after_phase"
            final = sample_and_write(None)
            if abort_reason is None:
                if final["available_memory_bytes"] < MIN_AVAILABLE:
                    abort_reason = "completion_available_memory_below_floor"
                elif final["free_swap_bytes"] < MIN_FREE_SWAP:
                    abort_reason = "completion_free_swap_below_floor"
                elif final["swap_used_bytes"] - baseline_swap > MAX_SWAP_GROWTH:
                    abort_reason = "completion_swap_growth_above_limit"
                elif final["disk_free_bytes"] < MIN_DISK:
                    abort_reason = "completion_disk_below_floor"
        else:
            process_quiescence = None
            sample_and_write(None)
    elapsed = round(time.monotonic() - started, 3)
    status = "ABORTED" if abort_reason else "EXITED"
    source_tree_after = tree_hash(source_root)
    source_mutated_by_test = source_tree_variant_before != source_tree_after
    record = {
        "schema_version": 1,
        "invocation_id": phase_id,
        "phase": phase,
        "expected_pass": expected_pass,
        "started_at_utc": started_utc,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "command_argv": command,
        "cwd": str(grader_test.parent),
        "environment": env,
        "runtime_identity": validation["runtime_identity"],
        "protocol_sha256": validation["protocol_sha256"],
        "runner_sha256": validation["runner_sha256"],
        "source_identity": {
            "base_commit": protocol["task"]["base_commit"],
            "fix_commit_for_control": protocol["task"]["fix_commit"] if phase == "fix" else None,
            "target_path": TARGET,
            "target_sha256": target_hash,
            "phase": phase,
        },
        "grader_identity": {
            "test_sha256": sha_file(grader_test),
            "selector": protocol["task"]["test_selector"],
            "test_outside_source_tree": True,
        },
        "return_code": return_code,
        "abort_reason": abort_reason,
        "status": status,
        "process_quiescence": process_quiescence,
        "resources": {
            "preflight": pre,
            "minimum_available_memory_bytes": min_available,
            "minimum_free_swap_bytes": min_free_swap,
            "maximum_swap_growth_bytes": max(0, max_swap - baseline_swap),
            "minimum_disk_free_bytes": min_disk_free,
            "maximum_process_tree_rss_bytes": max_rss,
            "sample_count": sum(1 for _ in samples_path.open()),
            "sample_interval_seconds": SAMPLE_INTERVAL,
        },
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha_file(stderr_path),
        "resource_samples_path": str(samples_path),
        "resource_samples_sha256": sha_file(samples_path),
        "source_tree_base": source_tree_base,
        "source_tree_variant_before_test": source_tree_variant_before,
        "source_tree_after": source_tree_after,
        "source_mutated_by_test": source_mutated_by_test,
        "raw_record_path": str(record_path),
    }
    expected_outcome_matched = test_matches(record, expected_pass) and not source_mutated_by_test
    record["expected_outcome_matched"] = expected_outcome_matched
    write_json_new(record_path, record)
    record_sha = sha_file(record_path)
    append_event("phase_record", {
        "phase": phase,
        "invocation_id": phase_id,
        "record_path": str(record_path.relative_to(EXPERIMENT)),
        "record_sha256": record_sha,
        "stdout_sha256": record["stdout_sha256"],
        "stderr_sha256": record["stderr_sha256"],
        "resource_samples_sha256": record["resource_samples_sha256"],
        "status": status,
        "return_code": return_code,
        "abort_reason": abort_reason,
        "expected_outcome_matched": expected_outcome_matched,
        "source_mutated_by_test": source_mutated_by_test,
    })
    return record


def repo_identity() -> tuple[str, str]:
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"], text=True).strip()
    return commit, tree


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-record", action="store_true", help="append protocol identity only; runs no test")
    mode.add_argument("--execute", action="store_true", help="run the one frozen baseline/mutant/fix sequence")
    mode.add_argument("--recover-interrupted", action="store_true", help="reconstruct incomplete phase evidence only; never execute tests")
    parser.add_argument("--source-archive", default="work/transformers-src-base.tar")
    parser.add_argument("--grader-test", default="work/transformers-grader-v1/tests/utils/test_generic.py")
    parser.add_argument("--fix-target", default="work/transformers-47468-controls-v1/fix-generic.py")
    parser.add_argument("--python", default="/opt/homebrew/Caskroom/miniforge/base/bin/python")
    parser.add_argument("--overlay-site-packages", default=(
        "/Users/user/Documents/Codex/2026-10-24/vare-flagship-research-verified-coding-agent/"
        "work/task-certification-20261025/work/mlx-core-development-agent-feasibility-v1/"
        "venv/lib/python3.12/site-packages"
    ))
    parser.add_argument("--output", default=str(EXPERIMENT / "run-v3"))
    args = parser.parse_args()

    if args.recover_interrupted:
        run_dir = resolve(args.output)
        require(MANIFEST.is_file(), "protocol manifest is missing; there are no attempts to reconstruct")
        lock_data = json.loads(LOCK.read_text(encoding="utf-8"))
        validate_locked_source_identity(lock_data)
        commit, tree = repo_identity()
        validate_recovery_identity(
            MANIFEST, protocol_commit=commit, protocol_tree=tree,
            protocol_sha256=lock_data["protocol_sha256"],
            runner_sha256=lock_data["runner_sha256"],
            lock_sha256=sha_file(LOCK),
        )
        recovered = recover_interrupted_attempts(MANIFEST, run_dir, EXPERIMENT)
        print(json.dumps({
            "status": "RECOVERY_COMPLETE_NO_TEST_EXECUTED",
            "reconstructed_attempts": len(recovered),
            "records": [row["event"].get("recovery_record_path") for row in recovered if not row.get("deferred")],
            "deferred_attempts": [row["event"].get("reason") for row in recovered if row.get("deferred")],
            "rerun_permitted": False,
        }, indent=2))
        return 0

    global PROTOCOL_DATA
    PROTOCOL_DATA = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    validation = validate_inputs(args)
    if not args.freeze_record and not args.execute:
        print(json.dumps({"status": "PREFLIGHT_VALID_NO_TEST_EXECUTED", **validation}, indent=2))
        return 0
    if args.freeze_record:
        require(not MANIFEST.exists(), "protocol manifest already exists; refusing overwrite")
        commit, tree = repo_identity()
        require(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip() == "",
                "worktree must be clean before freezing protocol identity")
        event = {
            "sequence": 1,
            "event_type": "protocol_frozen",
            "timestamp_utc": utc_now(),
            "previous_event_sha256": "0" * 64,
            "protocol_commit": commit,
            "protocol_tree": tree,
            "protocol_sha256": validation["protocol_sha256"],
            "lock_sha256": sha_file(LOCK),
            "runner_sha256": validation["runner_sha256"],
            "source_archive_sha256": validation["source_archive_sha256"],
            "grader_test_sha256": validation["grader_test_sha256"],
            "fix_target_sha256": validation["fix_target_sha256"],
            "mutant_target_sha256": validation["mutant_target_sha256"],
            "runtime_identity": validation["runtime_identity"],
            "network_deny_probe": validation["network_deny_probe"],
            "claim_limit": "Development-only source/grader qualification; no model inference or held-out evidence.",
        }
        event["event_sha256"] = event_hash(event)
        write_new(MANIFEST, (json.dumps(event, sort_keys=True) + "\n").encode())
        print(json.dumps({"status": "PROTOCOL_IDENTITY_FROZEN_NO_TEST_EXECUTED", "manifest": str(MANIFEST), **validation}, indent=2))
        return 0

    frozen_events = load_manifest()
    require(frozen_events and frozen_events[0]["event_type"] == "protocol_frozen", "frozen protocol identity missing")
    require_no_prior_attempt(frozen_events)
    commit, tree = repo_identity()
    frozen = frozen_events[0]
    require(commit == frozen["protocol_commit"] and tree == frozen["protocol_tree"], "code commit differs from frozen pre-outcome identity")
    require(not Path(args.output).exists(), "run output exists; one-shot runner refuses to repeat")
    run_dir = resolve(args.output)
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "records", "resources", "workspaces", "tmp"):
        (run_dir / name).mkdir()
    protocol = PROTOCOL_DATA
    fix_bytes = resolve(args.fix_target).read_bytes()
    mutant_bytes = make_mutant(fix_bytes)
    phases = [
        ("baseline", False, None),
        ("mutant", False, mutant_bytes),
        ("fix", True, fix_bytes),
    ]
    outcomes = []
    for phase, expected_pass, target_bytes in phases:
        if phase == "baseline":
            # Reconstruct the immutable base target directly from its source archive.
            with tarfile.open(resolve(args.source_archive), mode="r:") as tar:
                member = tar.extractfile(TARGET)
                require(member is not None, "baseline target is missing from source archive")
                target_bytes = member.read()
        result = run_phase(phase, expected_pass, target_bytes, args, protocol, run_dir, validation)
        outcomes.append(result)
        if not result["expected_outcome_matched"]:
            break
    expected_phases = ["baseline", "mutant", "fix"]
    qualified = (
        [row["phase"] for row in outcomes] == expected_phases
        and all(row["expected_outcome_matched"] for row in outcomes)
    )
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_commit": frozen["protocol_commit"],
        "protocol_sha256": validation["protocol_sha256"],
        "runner_sha256": validation["runner_sha256"],
        "status": "SOURCE_GRADER_QUALIFIED" if qualified else "UNQUALIFIED",
        "task_id": protocol["task"]["task_id"],
        "task_role": "DEVELOPMENT_ONLY",
        "phases": [{key: row[key] for key in ("phase", "expected_pass", "expected_outcome_matched", "return_code", "abort_reason", "raw_record_path")} for row in outcomes],
        "claim_limit": protocol["evidence_policy"]["claim_limit"],
    }
    summary_path = run_dir / "summary.json"
    write_json_new(summary_path, summary)
    append_event("run_summary", {"run_summary_path": str(summary_path.relative_to(EXPERIMENT)), "summary_sha256": sha_file(summary_path), "status": summary["status"]})
    print(json.dumps(summary, indent=2))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
