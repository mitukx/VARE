#!/usr/bin/env python3
"""Run exactly one isolated source-qualification arm for vLLM #64830."""
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


def build_pytest_command(
    profile: Path,
    python: Path,
    test_file: Path,
    pytest_config: Path,
    basetemp: Path,
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
        "max_process_tree_rss_bytes": MAX_RSS_BYTES,
    }
    marker.write_text(json.dumps(invocation, indent=2, sort_keys=True) + "\n")

    log_path = output_dir / "pytest.log"
    started = time.monotonic()
    peak_rss = 0
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
            while process.poll() is None:
                rss = process_tree_rss(process.pid)
                peak_rss = max(peak_rss, rss)
                if time.monotonic() - started > MAX_SECONDS:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                if rss > MAX_RSS_BYTES:
                    rss_exceeded = True
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.5)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise

    record = {
        **invocation,
        "exit_code": process.returncode if process is not None else None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_process_tree_rss_bytes": peak_rss,
        "timed_out": timed_out,
        "rss_limit_exceeded": rss_exceeded,
        "pytest_log_sha256": sha256_file(log_path),
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
    return 2 if timed_out or rss_exceeded else (process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
