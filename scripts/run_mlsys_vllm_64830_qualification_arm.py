#!/usr/bin/env python3
"""Run exactly one isolated source-qualification arm for vLLM #64830."""
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
ARM_CHOICES = ("baseline", "wrong-mutant", "upstream-fixed")
TEST_IDS = (
    "test_v1_model_runner_rejects_v2_only_config[overrides0-HiSparse]",
    "test_v1_model_runner_rejects_v2_only_config[overrides1-watermarking]",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_tree_rss(root_pid: int) -> int:
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    rows: dict[int, int] = {}
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        pid, ppid, rss_kib = map(int, fields)
        rows[pid] = rss_kib * 1024
        children.setdefault(ppid, []).append(pid)
    pending = [root_pid]
    seen: set[int] = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rows.get(pid, 0)
        pending.extend(children.get(pid, []))
    return total


def wait_for_child(process: subprocess.Popen, *, block: bool = False):
    """Reap a child with wait4 so its OS-reported high-water mark is retained."""
    options = 0 if block else os.WNOHANG
    while True:
        try:
            pid, status, usage = os.wait4(process.pid, options)
            break
        except InterruptedError:
            continue
    if pid == 0:
        return None
    process.returncode = os.waitstatus_to_exitcode(status)
    return usage


def wait4_max_rss_bytes(usage) -> int:
    """Convert wait4's ru_maxrss to bytes (Darwin reports bytes; Linux KiB)."""
    if sys.platform == "darwin":
        return int(usage.ru_maxrss)
    return int(usage.ru_maxrss) * 1024


def kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def build_pytest_command(
    profile: Path,
    python: Path,
    test_file: Path,
    pytest_config: Path,
    basetemp: Path,
    log_file: Path,
    test_ids: list[str],
) -> list[str]:
    command = [
        "/usr/bin/sandbox-exec",
        "-f",
        str(profile.resolve()),
        str(python.resolve()),
        "-m",
        "pytest",
        "-c",
        str(pytest_config.resolve()),
        "-q",
        "-p",
        "no:cacheprovider",
        "-o",
        f"log_file={log_file.resolve()}",
        "--basetemp",
        str(basetemp.resolve()),
    ]
    command.extend(
        f"{test_file.resolve()}::{test_id}" for test_id in test_ids
    )
    return command


def make_environment(output_dir: Path, method_file: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(output_dir / "home"),
        "TMPDIR": str(output_dir / "tmp"),
        "PIP_NO_INDEX": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "VARE_METHOD_FILE": str(method_file.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARM_CHOICES)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--method-file", type=Path, required=True)
    parser.add_argument("--method-sha256", required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--test-sha256", required=True)
    parser.add_argument("--pytest-config", type=Path, required=True)
    parser.add_argument("--pytest-config-sha256", required=True)
    parser.add_argument("--test-id", action="append", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--python-sha256", required=True)
    args = parser.parse_args()

    paths = {
        "profile": args.profile.resolve(),
        "method_file": args.method_file.resolve(),
        "test_file": args.test_file.resolve(),
        "pytest_config": args.pytest_config.resolve(),
        "python": args.python.resolve(),
    }
    expected = {
        "profile": args.profile_sha256,
        "method_file": args.method_sha256,
        "test_file": args.test_sha256,
        "pytest_config": args.pytest_config_sha256,
        "python": args.python_sha256,
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != expected:
        raise SystemExit(f"frozen input hash mismatch: expected={expected} actual={actual}")
    if tuple(args.test_id) != TEST_IDS:
        raise SystemExit(f"unexpected frozen test IDs: {args.test_id}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "invocation.started.json"
    result_path = output_dir / "attempt.json"
    if marker.exists() or result_path.exists():
        raise SystemExit(f"refusing to consume an arm twice: {output_dir}")
    tmp = output_dir / "tmp"
    home = output_dir / "home"
    basetemp = output_dir / "pytest-tmp"
    for path in (tmp, home, basetemp):
        path.mkdir(parents=True, exist_ok=True)

    command = build_pytest_command(
        paths["profile"],
        paths["python"],
        paths["test_file"],
        paths["pytest_config"],
        basetemp,
        output_dir / "pytest-internal.log",
        args.test_id,
    )
    env = make_environment(output_dir, paths["method_file"])
    runtime = subprocess.check_output(
        [
            str(paths["python"]),
            "-c",
            "import sys, pytest; print(sys.version); print('pytest=' + pytest.__version__)",
        ],
        text=True,
        env=env,
    ).strip()
    invocation = {
        "arm": args.arm,
        "source_revision": args.source_revision,
        "selected_test_ids": args.test_id,
        "command": command,
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "environment": {
            key: env[key]
            for key in (
                "HOME",
                "TMPDIR",
                "PIP_NO_INDEX",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONNOUSERSITE",
                "PYTHONHASHSEED",
                "VARE_METHOD_FILE",
            )
        },
        "input_sha256": actual,
        "runtime": runtime,
        "max_wall_seconds": MAX_SECONDS,
        "max_sampled_process_tree_rss_bytes": MAX_RSS_BYTES,
        "max_os_wait4_child_rss_bytes": MAX_RSS_BYTES,
        "rss_sample_interval_seconds": 0.5,
        "os_wait4_rss_scope": (
            "OS-reported high-water mark for the direct waited child; "
            "not a complete concurrent process-tree high-water mark"
        ),
    }
    marker.write_text(json.dumps(invocation, indent=2, sort_keys=True) + "\n")

    log_path = output_dir / "pytest.log"
    started = time.monotonic()
    sampled_process_tree_rss_max = 0
    child_usage = None
    timed_out = False
    rss_exceeded = False
    process: subprocess.Popen | None = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=output_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            while True:
                child_usage = wait_for_child(process)
                if child_usage is not None:
                    break
                rss = process_tree_rss(process.pid)
                sampled_process_tree_rss_max = max(
                    sampled_process_tree_rss_max, rss
                )
                if time.monotonic() - started > MAX_SECONDS:
                    timed_out = True
                    kill_process_group(process.pid)
                    break
                if rss > MAX_RSS_BYTES:
                    kill_process_group(process.pid)
                    break
                time.sleep(0.5)
            if process.returncode is None:
                child_usage = wait_for_child(process, block=True)
    except BaseException:
        if process is not None and process.returncode is None:
            kill_process_group(process.pid)
            wait_for_child(process, block=True)
        raise

    # The command leader can exit while same-session descendants remain alive.
    kill_process_group(process.pid)

    os_wait4_child_max_rss = (
        wait4_max_rss_bytes(child_usage) if child_usage is not None else None
    )
    rss_exceeded = (
        sampled_process_tree_rss_max > MAX_RSS_BYTES
        or (
            os_wait4_child_max_rss is not None
            and os_wait4_child_max_rss > MAX_RSS_BYTES
        )
    )

    record = {
        **invocation,
        "exit_code": process.returncode if process is not None else None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "sampled_process_tree_rss_max_bytes": sampled_process_tree_rss_max,
        "os_wait4_child_max_rss_bytes": os_wait4_child_max_rss,
        "rss_limit_exceeded": rss_exceeded,
        "rss_limit_scope": (
            "sampled process-tree total or OS direct-child high-water; "
            "transient aggregate peaks across unreported descendants are not excluded"
        ),
        "timed_out": timed_out,
        "pytest_log_sha256": sha256_file(log_path),
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
    return 2 if timed_out or rss_exceeded else (process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
