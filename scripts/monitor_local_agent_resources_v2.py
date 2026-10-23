#!/usr/bin/env python3
"""Bound one local agent process by wall time, RSS, swap, and free memory."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monitor_local_agent_resources_v1 as host_monitor  # noqa: E402


def process_rss_mib(pid: int) -> float:
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                            text=True, check=False, timeout=5)
    raw = result.stdout.strip()
    if result.returncode or not raw or not raw.isdigit():
        raise RuntimeError("could not read agent RSS from ps")
    return int(raw) / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--swap-abort-mib", type=float, required=True)
    parser.add_argument("--memory-free-abort-percent", type=int, required=True)
    parser.add_argument("--rss-abort-mib", type=float, required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("provide an agent command after --")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "DOCKER_CONTEXT": "colima"})
    started = time.monotonic()
    observations: list[dict[str, Any]] = []
    reason: str | None = None
    failure: str | None = None
    max_rss = 0.0
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        while process.poll() is None:
            try:
                host = host_monitor.observe()
                rss = process_rss_mib(process.pid)
            except Exception as exc:
                reason = "resource_telemetry_unavailable"
                failure = f"{type(exc).__name__}: {exc}"
                break
            elapsed = round(time.monotonic() - started, 3)
            max_rss = max(max_rss, rss)
            observations.append({"elapsed_seconds": elapsed, **host, "child_rss_mib": round(rss, 3)})
            if host["swap_used_mib"] >= args.swap_abort_mib:
                reason = "swap_abort_threshold"
                break
            if host["memory_free_percent"] < args.memory_free_abort_percent:
                reason = "memory_pressure_abort_threshold"
                break
            if rss >= args.rss_abort_mib:
                reason = "child_rss_abort_threshold"
                break
            if elapsed >= args.max_seconds:
                reason = "wall_time_abort_threshold"
                break
            time.sleep(args.poll_interval_seconds)
        if reason:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    result = {"schema_version": 1, "command": command, "returncode": process.returncode,
              "status": "ABORTED_RESOURCE" if reason else "COMPLETED" if process.returncode == 0 else "FAILED",
              "abort_reason": reason, "telemetry_failure": failure,
              "max_wall_seconds": args.max_seconds, "swap_abort_mib": args.swap_abort_mib,
              "memory_free_abort_percent": args.memory_free_abort_percent,
              "rss_abort_mib": args.rss_abort_mib, "max_child_rss_mib": round(max_rss, 3),
              "observations": observations, "elapsed_seconds": round(time.monotonic() - started, 3)}
    (output / "resource-monitor.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "returncode": result["returncode"],
                      "abort_reason": reason, "max_child_rss_mib": result["max_child_rss_mib"],
                      "elapsed_seconds": result["elapsed_seconds"]}))
    return 0 if result["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
