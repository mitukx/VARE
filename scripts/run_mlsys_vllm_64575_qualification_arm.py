#!/usr/bin/env python3
"""Run one isolated VARE vLLM 64575 qualification arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

MAX_SECONDS = 600
MAX_RSS_BYTES = 4 * 1024**3
ARM_CHOICES = ("baseline", "wrong-mutant", "upstream-fixed", "agent-candidate")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_tree_rss(root_pid: int) -> int:
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    rows = {}
    children = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        pid, ppid, rss_kib = map(int, fields)
        rows[pid] = rss_kib * 1024
        children.setdefault(ppid, []).append(pid)
    pending = [root_pid]
    seen = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rows.get(pid, 0)
        pending.extend(children.get(pid, []))
    return total


def build_pytest_command(
    profile: Path,
    python: str,
    test_file: Path,
    basetemp: Path,
    log_file: Path,
) -> list[str]:
    return [
        "/usr/bin/sandbox-exec",
        "-f",
        str(profile.resolve()),
        python,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-o",
        f"log_file={log_file.resolve()}",
        "--basetemp",
        str(basetemp.resolve()),
        str(test_file.resolve()),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARM_CHOICES)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "pytest.log"
    result_path = out / "attempt.json"
    tmp = out / "tmp"
    home = out / "home"
    basetemp = out / "pytest-tmp"
    for path in (tmp, home, basetemp):
        path.mkdir(parents=True, exist_ok=True)
    command = build_pytest_command(
        args.profile,
        args.python,
        args.test_file,
        basetemp,
        out / "pytest-internal.log",
    )
    env = os.environ.copy()
    env.update({
        "VARE_SOURCE_FILE": str(args.source_file.resolve()),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "PIP_NO_INDEX": "1",
    })
    started = time.monotonic()
    peak_rss = 0
    timed_out = False
    rss_exceeded = False
    with log_path.open("wb") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        while proc.poll() is None:
            rss = process_tree_rss(proc.pid)
            peak_rss = max(peak_rss, rss)
            if time.monotonic() - started > MAX_SECONDS:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                break
            if rss > MAX_RSS_BYTES:
                rss_exceeded = True
                os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    record = {
        "arm": args.arm,
        "command": command,
        "exit_code": proc.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_process_tree_rss_bytes": peak_rss,
        "timed_out": timed_out,
        "rss_limit_exceeded": rss_exceeded,
        "test_file_sha256": sha256_file(args.test_file.resolve()),
        "source_file_sha256": sha256_file(args.source_file.resolve()),
        "sandbox_profile_sha256": sha256_file(args.profile.resolve()),
        "pytest_log_sha256": sha256_file(log_path),
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
    return 0 if not timed_out and not rss_exceeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
