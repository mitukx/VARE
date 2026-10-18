#!/usr/bin/env python3
"""Repeat the locked SQLite worker regression against pinned source trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=checkout, text=True).strip()


def require_clean_checkout(checkout: Path) -> None:
    status = git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"source checkout is dirty: {checkout}: {status}")


def run_one(checkout: Path, output: Path, timeout_s: float, index: int) -> dict:
    require_clean_checkout(checkout)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_trajectory_workers.py::test_independent_stages_bounded_backlog_and_slow_step_heartbeats",
    ]
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = str(checkout / "src")
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=checkout,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        raw, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        raw, _ = process.communicate()
    require_clean_checkout(checkout)
    elapsed_s = time.monotonic() - started
    log_path = output / f"run-{index}.log"
    log_path.write_bytes(raw)
    return {
        "index": index,
        "status": "timeout" if timed_out else process.returncode,
        "elapsed_s": elapsed_s,
        "log": log_path.name,
        "log_sha256": sha256(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()

    checkout = args.checkout.resolve()
    output = args.output.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    arm = protocol["arms"][args.arm]
    source_path = checkout / protocol["source_file"]
    test_path = checkout / protocol["test_file"]
    require_clean_checkout(checkout)

    actual = {
        "head": git(checkout, "rev-parse", "HEAD"),
        "source_sha256": sha256(source_path),
        "test_sha256": sha256(test_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "working_tree_clean": True,
    }
    expected = {
        "head": arm["head"],
        "source_sha256": arm["source_sha256"],
        "test_sha256": protocol["test_sha256"],
        "runner_sha256": protocol["runner_sha256"],
        "working_tree_clean": True,
    }
    if actual != expected:
        raise SystemExit(f"source identity mismatch: expected={expected}, actual={actual}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    runs = [
        run_one(checkout, output, protocol["timeout_s"], i)
        for i in range(1, protocol["repetitions"] + 1)
    ]
    expected_status = arm["expected_status"]
    summary = {
        "protocol_id": protocol["protocol_id"],
        "arm": args.arm,
        "source_identity": actual,
        "runs": runs,
        "all_expected": all(run["status"] == expected_status for run in runs),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
