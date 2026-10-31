#!/usr/bin/env python3
"""Run one isolated VARE vLLM 64575 qualification arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
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
    """Terminate every process left in this attempt's dedicated session."""
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


def create_attempt_output_dir(output_dir: Path) -> Path:
    """Create a fresh attempt directory without overwriting prior evidence."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


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
    out = create_attempt_output_dir(args.output_dir)
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
    monitor_error = None
    child_usage = None
    with log_path.open("wb") as log:
        try:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        except OSError as exc:
            record = {
                "arm": args.arm,
                "command": command,
                "status": "LAUNCH_FAILED",
                "launch_error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "sampled_process_tree_rss_max_bytes": 0,
                "os_wait4_child_max_rss_bytes": None,
                "timed_out": False,
                "rss_limit_exceeded": False,
                "rss_limit_scope": "no child started; no memory measurement",
                "monitor_error": None,
                "test_file_sha256": sha256_file(args.test_file.resolve()),
                "source_file_sha256": sha256_file(args.source_file.resolve()),
                "sandbox_profile_sha256": sha256_file(args.profile.resolve()),
                "pytest_log_sha256": sha256_file(log_path),
            }
            result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))
            return 2
        try:
            while True:
                child_usage = wait_for_child(proc)
                if child_usage is not None:
                    break
                try:
                    rss = process_tree_rss(proc.pid)
                except Exception as exc:
                    monitor_error = f"{type(exc).__name__}: {exc}"
                    child_usage = stop_process_group(proc)
                    break
                peak_rss = max(peak_rss, rss)
                if time.monotonic() - started > MAX_SECONDS:
                    timed_out = True
                    child_usage = stop_process_group(proc)
                    break
                if rss > MAX_RSS_BYTES:
                    child_usage = stop_process_group(proc)
                    break
                time.sleep(0.5)
        except BaseException:
            if proc.returncode is None:
                stop_process_group(proc)
            else:
                kill_process_group(proc.pid)
            raise
        cleanup_usage = stop_process_group(proc)
        if child_usage is None:
            child_usage = cleanup_usage
    os_wait4_child_max_rss = (
        wait4_max_rss_bytes(child_usage) if child_usage is not None else None
    )
    rss_exceeded = (
        peak_rss > MAX_RSS_BYTES
        or (os_wait4_child_max_rss is not None and os_wait4_child_max_rss > MAX_RSS_BYTES)
    )
    record = {
        "arm": args.arm,
        "command": command,
        "exit_code": proc.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "sampled_process_tree_rss_max_bytes": peak_rss,
        "os_wait4_child_max_rss_bytes": os_wait4_child_max_rss,
        "timed_out": timed_out,
        "rss_limit_exceeded": rss_exceeded,
        "rss_limit_scope": (
            "sampled process-tree total or OS direct-child high-water; "
            "transient aggregate peaks across unreported descendants are not excluded"
        ),
        "monitor_error": monitor_error,
        "test_file_sha256": sha256_file(args.test_file.resolve()),
        "source_file_sha256": sha256_file(args.source_file.resolve()),
        "sandbox_profile_sha256": sha256_file(args.profile.resolve()),
        "pytest_log_sha256": sha256_file(log_path),
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
    return 0 if not timed_out and not rss_exceeded and monitor_error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
