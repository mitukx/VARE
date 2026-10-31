#!/usr/bin/env python3
"""Run one local coding-agent attempt under wall-clock and process-tree RSS limits."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process_tree_rss(root_pid: int) -> int:
    """Return resident bytes for the root process and its live descendants."""
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    rows: dict[int, tuple[int, int]] = {}
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        pid, parent_pid, rss_kib = map(int, fields)
        rows[pid] = (parent_pid, rss_kib * 1024)
        children.setdefault(parent_pid, []).append(pid)

    pending = [root_pid]
    seen: set[int] = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if pid in rows:
            total += rows[pid][1]
        pending.extend(children.get(pid, []))
    return total


def wait_for_child(proc: subprocess.Popen[bytes], *, block: bool = False):
    """Reap the command with wait4 so its OS-reported RSS high-water is retained."""
    options = 0 if block else os.WNOHANG
    while True:
        try:
            pid, status, usage = os.wait4(proc.pid, options)
            break
        except InterruptedError:
            continue
    if pid == 0:
        return None
    proc.returncode = os.waitstatus_to_exitcode(status)
    return usage


def wait4_max_rss_bytes(usage) -> int:
    """Convert wait4 ru_maxrss to bytes (Darwin reports bytes; Linux KiB)."""
    if sys.platform == "darwin":
        return int(usage.ru_maxrss)
    return int(usage.ru_maxrss) * 1024


def kill_process_group(pid: int, sig: int = signal.SIGKILL) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def stop_process_group(proc: subprocess.Popen[bytes]):
    """Stop every remaining process in the attempt group."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    usage = None
    deadline = time.monotonic() + 5
    while proc.returncode is None and time.monotonic() < deadline:
        usage = wait_for_child(proc)
        if usage is not None:
            break
        time.sleep(0.05)
    kill_process_group(proc.pid)
    if proc.returncode is None:
        usage = wait_for_child(proc, block=True)
    return usage


def run_limited(
    command: list[str], *, wall_seconds: int, rss_limit_bytes: int,
    log_path: Path, record_path: Path | None = None, sample_seconds: float = 0.5,
) -> dict:
    """Execute exactly one command, recording limits and combined output."""
    if not command:
        raise ValueError("command must not be empty")
    if wall_seconds < 1 or rss_limit_bytes < 1 or sample_seconds <= 0:
        raise ValueError("resource limits and sampling interval must be positive")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "status": "LAUNCHING",
                "started_utc": started_utc,
                "command": command,
                "wall_seconds_limit": wall_seconds,
                "process_tree_rss_limit_bytes": rss_limit_bytes,
                "rss_measurement_scope": (
                    "sampled process-tree total plus OS direct-child high-water; "
                    "transient aggregate descendant peaks are not excluded"
                ),
            }, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    peak_rss = 0
    limit_reason = None
    monitor_error = None
    child_usage = None
    group_stopped = False
    try:
        with log_path.open("xb") as log:
            proc = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            if record_path is not None:
                record_path.write_text(json.dumps({
                    "status": "RUNNING",
                    "started_utc": started_utc,
                    "command": command,
                    "wall_seconds_limit": wall_seconds,
                    "process_tree_rss_limit_bytes": rss_limit_bytes,
                    "rss_measurement_scope": (
                        "sampled process-tree total plus OS direct-child high-water; "
                        "transient aggregate descendant peaks are not excluded"
                    ),
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            while True:
                child_usage = wait_for_child(proc)
                if child_usage is not None:
                    break
                try:
                    rss = process_tree_rss(proc.pid)
                except Exception as exc:  # fail closed if the monitor stops working
                    monitor_error = f"{type(exc).__name__}: {exc}"
                    limit_reason = "process_tree_monitor_failed"
                    child_usage = stop_process_group(proc)
                    group_stopped = True
                    break
                peak_rss = max(peak_rss, rss)
                elapsed = time.monotonic() - started
                if rss > rss_limit_bytes:
                    limit_reason = "process_tree_rss_limit_exceeded"
                elif elapsed > wall_seconds:
                    limit_reason = "wall_clock_limit_exceeded"
                if limit_reason is not None:
                    child_usage = stop_process_group(proc)
                    group_stopped = True
                    break
                time.sleep(min(sample_seconds, max(0.01, wall_seconds - elapsed)))
            # A candidate can exit successfully after leaving a child in the
            # session. The attempt boundary applies to the whole process group.
            if not group_stopped:
                cleanup_usage = stop_process_group(proc)
                if child_usage is None:
                    child_usage = cleanup_usage
    except OSError as exc:
        if "proc" in locals():
            stop_process_group(proc)
        result = {
            "status": "LAUNCH_FAILED",
            "started_utc": started_utc,
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds_limit": wall_seconds,
            "process_tree_rss_limit_bytes": rss_limit_bytes,
            "rss_measurement_scope": (
                "sampled process-tree total plus OS direct-child high-water; "
                "transient aggregate descendant peaks are not excluded"
            ),
            "log_path": str(log_path),
        }
        if record_path is not None:
            record_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    os_wait4_child_max_rss = (
        wait4_max_rss_bytes(child_usage) if child_usage is not None else None
    )
    rss_limit_exceeded = (
        peak_rss > rss_limit_bytes
        or (
            os_wait4_child_max_rss is not None
            and os_wait4_child_max_rss > rss_limit_bytes
        )
    )
    if limit_reason is None and rss_limit_exceeded:
        limit_reason = "process_tree_rss_limit_exceeded"
    result = {
        "status": "COMPLETED" if limit_reason is None else "ABORTED_BY_LIMIT",
        "started_utc": started_utc,
        "command": command,
        "exit_code": proc.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "sampled_process_tree_rss_max_bytes": peak_rss,
        "os_wait4_child_max_rss_bytes": os_wait4_child_max_rss,
        "rss_limit_exceeded": rss_limit_exceeded,
        "rss_measurement_scope": (
            "sampled process-tree total plus OS direct-child high-water; "
            "transient aggregate descendant peaks are not excluded"
        ),
        "wall_seconds_limit": wall_seconds,
        "process_tree_rss_limit_bytes": rss_limit_bytes,
        "limit_reason": limit_reason,
        "monitor_error": monitor_error,
        "log_path": str(log_path),
        "log_bytes": log_path.stat().st_size,
    }
    if record_path is not None:
        record_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--max-process-tree-rss-bytes", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = run_limited(
        command,
        wall_seconds=args.wall_seconds,
        rss_limit_bytes=args.max_process_tree_rss_bytes,
        log_path=args.log,
        record_path=args.record,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "COMPLETED" and result.get("exit_code") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
