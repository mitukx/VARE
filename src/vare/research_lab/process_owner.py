"""Bounded process-group execution for one candidate command.

The process-group guard is the existing ``vare._grader_guard`` owner. When
the owning pipe closes, including after this process is killed, the guard
signals its own group. A candidate that leaves the group is outside this
containment. This module is not a secure sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Callable

_GUARD = Path(__file__).resolve().parents[1] / "_grader_guard.py"
_WRAPPER = Path(__file__).resolve().with_name("_bounded_child.py")
_ENV_KEEP = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR")


class ProcessOwnerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OwnedCommand:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    wall_s: float
    failure_class: str
    peak_rss_bytes: int
    pgid: int | None


def process_identity(pid: int) -> tuple[int, int] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    rparen = stat.rfind(")")
    fields = stat[rparen + 2 :].split()
    if len(fields) < 20:
        return None
    return pid, int(fields[19])


def group_rss_bytes(pgid: int) -> int:
    total = 0
    proc = Path("/proc")
    if not proc.is_dir():
        return 0
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            rparen = stat.rfind(")")
            fields = stat[rparen + 2 :].split()
            if len(fields) < 3 or int(fields[2]) != pgid:
                continue
            status = (entry / "status").read_text(encoding="utf-8")
        except (OSError, ValueError):
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                total += int(line.split()[1]) * 1024
                break
    return total


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return


def _classify(returncode: int | None, *, reason: str | None, stderr: bytes, cpu_limited: bool) -> str:
    text = stderr.decode("utf-8", "replace")
    if reason is not None:
        return reason
    if returncode == 126 and "research_lab_network_isolation_unavailable" in text:
        return "network_isolation_unavailable"
    if returncode == 127 and "research_lab_exec_failed" in text:
        return "launch_error"
    if cpu_limited and returncode in {-9, -24}:
        return "cpu_limit"
    if returncode == 0:
        return "completed"
    if returncode is None or returncode < 0:
        return "signal_killed"
    return "nonzero_exit"


def run_bounded_command(
    argv: list[str],
    *,
    cwd: Path,
    limits: dict,
    extra_env: dict[str, str] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_start: Callable[[int, int], None] | None = None,
) -> OwnedCommand:
    """Run ``argv`` once. ``limits`` is trusted data from the parent, not the candidate."""
    if os.name != "posix" or not Path("/proc/self").exists():
        raise ProcessOwnerError("process-group execution requires Linux /proc")
    if not argv or not _GUARD.is_file() or not _WRAPPER.is_file():
        raise ProcessOwnerError("execution guard is unavailable")
    env = {key: os.environ[key] for key in _ENV_KEEP if key in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["HOME"] = str(cwd)
    if extra_env:
        env.update(extra_env)
    env["VARE_RESEARCH_LAB_LIMITS"] = json.dumps(limits, sort_keys=True, separators=(",", ":"))
    status_read, status_write = os.pipe()
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    stdout_len = 0
    stderr_len = 0
    overflow = threading.Event()
    max_output = int(limits["max_output_bytes"])

    def drain(stream, sink: list[bytes], which: str) -> None:
        nonlocal stdout_len, stderr_len
        while True:
            block = stream.read(16384)
            if not block:
                return
            if which == "stdout":
                stdout_len += len(block)
                if stdout_len <= max_output:
                    sink.append(block)
            else:
                stderr_len += len(block)
                if stderr_len <= max_output:
                    sink.append(block)
            if stdout_len + stderr_len > max_output:
                overflow.set()

    proc = subprocess.Popen(
        [sys.executable, "-B", str(_GUARD), str(status_write), sys.executable, "-B", str(_WRAPPER), "--", *argv],
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(status_write,),
        start_new_session=True,
        close_fds=True,
    )
    os.close(status_write)
    identity = process_identity(proc.pid)
    if on_start is not None and identity is not None:
        on_start(*identity)
    assert proc.stdout is not None and proc.stderr is not None and proc.stdin is not None
    readers = [
        threading.Thread(target=drain, args=(proc.stdout, stdout_chunks, "stdout"), daemon=True),
        threading.Thread(target=drain, args=(proc.stderr, stderr_chunks, "stderr"), daemon=True),
    ]
    for reader in readers:
        reader.start()
    status_holder: dict[str, bytes] = {}

    def read_status() -> None:
        chunks: list[bytes] = []
        remaining = 4
        try:
            while remaining:
                block = os.read(status_read, remaining)
                if not block:
                    break
                chunks.append(block)
                remaining -= len(block)
        except OSError:
            pass
        status_holder["raw"] = b"".join(chunks)

    status_thread = threading.Thread(target=read_status, daemon=True)
    status_thread.start()
    started = time.monotonic()
    deadline = started + float(limits["wall_time_s"])
    memory_mb = limits.get("memory_mb")
    reason: str | None = None
    peak = 0
    cpu_seen = False
    try:
        while True:
            elapsed = time.monotonic() - started
            if limits.get("cpu_time_s") and elapsed >= float(limits["cpu_time_s"]) * 0.8:
                cpu_seen = True
            if should_cancel is not None and should_cancel():
                reason = "cancelled"
                _kill_group(proc.pid)
                break
            if overflow.is_set():
                reason = "output_limit"
                _kill_group(proc.pid)
                break
            if isinstance(memory_mb, int):
                rss = group_rss_bytes(proc.pid)
                peak = max(peak, rss)
                if rss > memory_mb * 1024 * 1024:
                    reason = "memory_limit"
                    _kill_group(proc.pid)
                    break
            # The guard stays alive until the owner closes stdin. Completion is
            # the status read plus stdout/stderr EOF, not the guard's exit.
            if not status_thread.is_alive() and not any(reader.is_alive() for reader in readers):
                if reason is None and should_cancel is not None and should_cancel():
                    reason = "cancelled"
                break
            if time.monotonic() >= deadline:
                reason = "wall_timeout"
                _kill_group(proc.pid)
                break
            time.sleep(0.02)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _kill_group(proc.pid)
            proc.wait(timeout=2)
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        for reader in readers:
            reader.join(timeout=2)
        status_thread.join(timeout=2)
        os.close(status_read)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        if proc.poll() is None:
            _kill_group(proc.pid)
            proc.wait(timeout=2)
    stdout = b"".join(stdout_chunks)[:max_output]
    stderr = b"".join(stderr_chunks)[:max_output]
    returncode = proc.returncode
    if len(status_holder.get("raw") or b"") == 4 and reason is None:
        import struct
        child_code = struct.unpack("!i", status_holder["raw"])[0]
        # The guard's own return code is not the candidate's code.
        returncode = child_code
    failure = _classify(returncode, reason=reason, stderr=stderr, cpu_limited=bool(limits.get("cpu_time_s")) and cpu_seen)
    return OwnedCommand(returncode, stdout, stderr, time.monotonic() - started, failure, peak, proc.pid)
