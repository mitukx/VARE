#!/usr/bin/env python3
"""Run one bounded local coding-agent attempt under host resource guards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import subprocess
import sys
import threading
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def swap_usage_bytes() -> tuple[int, int, str]:
    raw = subprocess.run(
        ["/usr/sbin/sysctl", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    used_match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", raw)
    total_match = re.search(r"total\s*=\s*([0-9.]+)([KMG])", raw)
    if used_match is None or total_match is None:
        raise RuntimeError("cannot parse macOS swap usage")
    units = {"K": 1024, "M": 1024**2, "G": 1024**3}
    used = int(float(used_match.group(1)) * units[used_match.group(2)])
    total = int(float(total_match.group(1)) * units[total_match.group(2)])
    return used, max(total - used, 0), raw


def available_memory_bytes() -> tuple[int, str]:
    raw = subprocess.run(
        ["/usr/bin/vm_stat"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout
    page_match = re.search(r"page size of (\d+) bytes", raw)
    if page_match is None:
        raise RuntimeError("cannot parse vm_stat page size")
    page_size = int(page_match.group(1))

    def pages(label: str) -> int:
        match = re.search(rf"^{re.escape(label)}:\s*(\d+)\.", raw, re.MULTILINE)
        return int(match.group(1)) if match is not None else 0

    available = sum(
        pages(label)
        for label in ("Pages free", "Pages inactive", "Pages speculative")
    ) * page_size
    return available, raw


def process_tree_rss_bytes(root_pid: int) -> int:
    raw = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,rss="],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout
    table: dict[int, tuple[int, int]] = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, rss_kib = map(int, parts)
        table[pid] = (ppid, rss_kib)
    descendants = {root_pid}
    while True:
        found = {pid for pid, (ppid, _rss) in table.items() if ppid in descendants}
        new = found - descendants
        if not new:
            break
        descendants.update(new)
    return sum(table[pid][1] * 1024 for pid in descendants if pid in table)


def terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def copy_stream(source, destination: Path) -> None:
    with destination.open("xb") as output:
        while True:
            block = source.read(65536)
            if not block:
                break
            output.write(block)
            output.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--agent-record", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    record_path = args.record
    if record_path.exists():
        print("refusing to overwrite guard receipt", file=sys.stderr)
        return 2
    if args.stdout_log.exists() or args.stderr_log.exists():
        print("refusing to overwrite process logs", file=sys.stderr)
        return 2
    protocol_path = args.protocol.resolve(strict=True)
    preflight_path = args.preflight.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(protocol_path)
    preflight_hash = sha256_file(preflight_path)
    gates = protocol["resource_gates"]
    model = protocol["model_trial"]
    if preflight.get("status") != "PASS_PRELOAD_GATE":
        raise SystemExit("resource guard requires a passing preload record")
    if preflight.get("protocol_sha256") != protocol_hash:
        raise SystemExit("preflight protocol hash mismatch")
    if not args.command:
        raise SystemExit("missing bounded child command")
    runner_path = args.runner.resolve(strict=True)
    if sha256_file(runner_path) != model["runner_sha256"]:
        raise SystemExit("runner SHA-256 differs from the frozen protocol")
    if str(runner_path) not in args.command:
        raise SystemExit("child command does not invoke the pinned runner")
    if args.agent_record.exists():
        raise SystemExit("refusing to overwrite an existing agent record")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("this resource guard is frozen for macOS arm64 only")

    baseline_swap, baseline_free_swap, baseline_swap_text = swap_usage_bytes()
    initial_memory, initial_vm_stat = available_memory_bytes()
    initial_disk = shutil.disk_usage(protocol_path.parent).free
    static_gate = {
        "minimum_available_memory_bytes": initial_memory >= gates["minimum_available_memory_bytes"],
        "minimum_free_swap_bytes": baseline_free_swap >= gates["minimum_free_swap_bytes"],
        "minimum_disk_free_bytes": initial_disk >= gates["minimum_disk_free_bytes"],
    }
    if not all(static_gate.values()):
        raise SystemExit("host resource gate failed immediately before model load")

    args.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    args.stderr_log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["DOCKER_CONTEXT"] = "colima"
    process = subprocess.Popen(
        args.command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    readers = [
        threading.Thread(target=copy_stream, args=(process.stdout, args.stdout_log), daemon=True),
        threading.Thread(target=copy_stream, args=(process.stderr, args.stderr_log), daemon=True),
    ]
    for reader in readers:
        reader.start()

    started = time.monotonic()
    max_rss = 0
    max_swap_growth = 0
    minimum_available = initial_memory
    minimum_disk_free = initial_disk
    stop_reason = None
    samples = 0
    last_swap_text = baseline_swap_text
    while process.poll() is None:
        elapsed = time.monotonic() - started
        rss = process_tree_rss_bytes(process.pid)
        swap_now, free_swap_now, swap_text = swap_usage_bytes()
        available_now, _vmstat = available_memory_bytes()
        disk_now = shutil.disk_usage(protocol_path.parent).free
        samples += 1
        max_rss = max(max_rss, rss)
        max_swap_growth = max(max_swap_growth, swap_now - baseline_swap)
        minimum_available = min(minimum_available, available_now)
        minimum_disk_free = min(minimum_disk_free, disk_now)
        last_swap_text = swap_text

        if rss > gates["max_process_rss_bytes"]:
            stop_reason = "process_rss_cap_exceeded"
        elif max_swap_growth > gates["max_swap_growth_bytes"]:
            stop_reason = "host_swap_growth_cap_exceeded"
        elif free_swap_now < gates["minimum_free_swap_bytes"]:
            stop_reason = "minimum_free_swap_gate_failed"
        elif available_now < gates["minimum_available_memory_bytes"]:
            stop_reason = "host_available_memory_gate_failed"
        elif disk_now < gates["minimum_disk_free_bytes"]:
            stop_reason = "minimum_disk_free_gate_failed"
        elif elapsed > model["wall_seconds"]:
            stop_reason = "wall_time_cap_exceeded"
        if stop_reason is not None:
            terminate_process_group(process)
            break
        time.sleep(args.sample_interval_seconds)

    return_code = process.wait()
    for reader in readers:
        reader.join(timeout=5)
    elapsed = time.monotonic() - started
    final_swap, final_free_swap, final_swap_text = swap_usage_bytes()
    final_memory, final_vm_stat = available_memory_bytes()
    final_disk = shutil.disk_usage(protocol_path.parent).free
    if max_rss == 0:
        max_rss = process_tree_rss_bytes(process.pid)
    max_swap_growth = max(max_swap_growth, final_swap - baseline_swap)

    agent_result = None
    if args.agent_record.is_file():
        try:
            agent_result = json.loads(args.agent_record.read_text(encoding="utf-8"))
        except Exception as exc:
            stop_reason = stop_reason or f"agent_record_unreadable:{type(exc).__name__}"
    if stop_reason is None and return_code != 0:
        stop_reason = "runner_nonzero_exit"
    if stop_reason is None and agent_result is None:
        stop_reason = "runner_record_missing"

    mlx_peak = (
        agent_result.get("runtime", {}).get("mlx_peak_memory_bytes")
        if isinstance(agent_result, dict)
        else None
    )
    if stop_reason is None and (not isinstance(mlx_peak, int) or mlx_peak > gates["max_mlx_peak_memory_bytes"]):
        stop_reason = "mlx_peak_memory_cap_exceeded_or_unmeasured"

    record = {
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "preflight_sha256": preflight_hash,
        "command": args.command,
        "child_return_code": return_code,
        "status": "COMPLETED_WITHIN_RESOURCE_GATES" if stop_reason is None else "STOPPED_OR_FAILED",
        "stop_reason": stop_reason,
        "elapsed_seconds": round(elapsed, 3),
        "sample_count": samples,
        "baseline_swap_used_bytes": baseline_swap,
        "baseline_free_swap_bytes": baseline_free_swap,
        "baseline_swap_usage_text": baseline_swap_text,
        "maximum_swap_growth_bytes": max_swap_growth,
        "last_swap_usage_text": last_swap_text,
        "final_swap_usage_text": final_swap_text,
        "final_free_swap_bytes": final_free_swap,
        "maximum_process_tree_rss_bytes": max_rss,
        "minimum_available_memory_bytes": minimum_available,
        "final_available_memory_bytes": final_memory,
        "final_vm_stat": final_vm_stat,
        "minimum_disk_free_bytes": minimum_disk_free,
        "final_disk_free_bytes": final_disk,
        "static_resource_gate": static_gate,
        "mlx_peak_memory_bytes": mlx_peak,
        "mlx_peak_limit_bytes": gates["max_mlx_peak_memory_bytes"],
        "agent_record_path": str(args.agent_record),
        "agent_record_sha256": sha256_file(args.agent_record) if args.agent_record.is_file() else None,
        "stdout_log": str(args.stdout_log),
        "stdout_log_sha256": sha256_file(args.stdout_log) if args.stdout_log.is_file() else None,
        "stderr_log": str(args.stderr_log),
        "stderr_log_sha256": sha256_file(args.stderr_log) if args.stderr_log.is_file() else None,
        "historical_protocols_modified": False,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: record[key]
        for key in (
            "status",
            "stop_reason",
            "elapsed_seconds",
            "maximum_process_tree_rss_bytes",
            "maximum_swap_growth_bytes",
            "mlx_peak_memory_bytes",
        )
    }, sort_keys=True))
    return 0 if stop_reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
