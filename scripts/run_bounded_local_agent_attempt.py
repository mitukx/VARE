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


def stop_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Stop every process in the attempt group after a resource-limit breach."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit on SIGTERM while a descendant ignores it. Keep the
    # process-group ID and kill any survivors even after reaping the leader.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if proc.poll() is None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


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
            }, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    peak_rss = 0
    limit_reason = None
    monitor_error = None
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
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            while proc.poll() is None:
                try:
                    rss = process_tree_rss(proc.pid)
                except Exception as exc:  # fail closed if the monitor stops working
                    monitor_error = f"{type(exc).__name__}: {exc}"
                    limit_reason = "process_tree_monitor_failed"
                    stop_process_group(proc)
                    break
                peak_rss = max(peak_rss, rss)
                elapsed = time.monotonic() - started
                if rss > rss_limit_bytes:
                    limit_reason = "process_tree_rss_limit_exceeded"
                elif elapsed > wall_seconds:
                    limit_reason = "wall_clock_limit_exceeded"
                if limit_reason is not None:
                    stop_process_group(proc)
                    break
                time.sleep(min(sample_seconds, max(0.01, wall_seconds - elapsed)))
            returncode = proc.wait()
    except OSError as exc:
        if "proc" in locals() and proc.poll() is None:
            stop_process_group(proc)
        result = {
            "status": "LAUNCH_FAILED",
            "started_utc": started_utc,
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds_limit": wall_seconds,
            "process_tree_rss_limit_bytes": rss_limit_bytes,
            "log_path": str(log_path),
        }
        if record_path is not None:
            record_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    result = {
        "status": "COMPLETED" if limit_reason is None else "ABORTED_BY_LIMIT",
        "started_utc": started_utc,
        "command": command,
        "exit_code": returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "max_observed_process_tree_rss_bytes": peak_rss,
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
    return 0 if result["exit_code"] == 0 and result["limit_reason"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
