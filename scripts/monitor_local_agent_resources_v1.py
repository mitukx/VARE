#!/usr/bin/env python3
"""Run one agent attempt with fail-closed macOS memory/swap monitoring."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def observe() -> dict[str, Any]:
    swap = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, check=True, timeout=5).stdout
    pressure = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=True, timeout=5).stdout
    swap_match = re.search(r"used = ([\d.]+)M", swap)
    free_match = re.search(r"System-wide memory free percentage: (\d+)%", pressure)
    if not swap_match or not free_match:
        raise RuntimeError("could not parse required resource telemetry")
    return {"swap_used_mib": float(swap_match.group(1)), "memory_free_percent": int(free_match.group(1))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--swap-abort-mib", type=float, default=2900.0)
    parser.add_argument("--memory-free-abort-percent", type=int, default=15)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("provide an agent command after --")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "DOCKER_CONTEXT": "colima"})
    started = time.monotonic()
    observations = []
    reason = None
    failure = None
    with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        while process.poll() is None:
            try:
                sample = observe()
            except Exception as exc:
                reason = "resource_telemetry_unavailable"
                failure = f"{type(exc).__name__}: {exc}"
                break
            observations.append({"elapsed_seconds": round(time.monotonic() - started, 3), **sample})
            if sample["swap_used_mib"] >= args.swap_abort_mib:
                reason = "swap_abort_threshold"
                break
            if sample["memory_free_percent"] < args.memory_free_abort_percent:
                reason = "memory_pressure_abort_threshold"
                break
            if time.monotonic() - started >= args.max_seconds:
                reason = "wall_time_abort_threshold"
                break
            time.sleep(2)
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
              "observations": observations, "elapsed_seconds": round(time.monotonic() - started, 3)}
    (out / "resource-monitor.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "returncode": result["returncode"],
                      "abort_reason": reason, "elapsed_seconds": result["elapsed_seconds"]}))
    return 0 if result["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
