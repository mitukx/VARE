import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_bounded_local_agent_attempt import process_tree_rss, run_limited


def test_process_tree_rss_sums_root_and_nested_descendants() -> None:
    ps_output = "10 1 100\n11 10 50\n12 1 200\n13 11 25\n"
    with patch(
        "scripts.run_bounded_local_agent_attempt.subprocess.check_output",
        return_value=ps_output,
    ):
        assert process_tree_rss(10) == (100 + 50 + 25) * 1024


def test_run_limited_records_command_and_output(tmp_path) -> None:
    record = run_limited(
        ["/usr/bin/true"],
        wall_seconds=2,
        rss_limit_bytes=1024**3,
        log_path=tmp_path / "attempt.log",
    )

    assert record["exit_code"] == 0
    assert record["limit_reason"] is None
    assert record["command"] == ["/usr/bin/true"]
    assert record["log_bytes"] == 0


def test_run_limited_stops_process_when_rss_budget_is_exceeded(tmp_path) -> None:
    record = run_limited(
        ["/bin/sleep", "5"],
        wall_seconds=3,
        rss_limit_bytes=1,
        sample_seconds=0.05,
        log_path=tmp_path / "attempt.log",
    )

    assert record["limit_reason"] == "process_tree_rss_limit_exceeded"
    assert record["sampled_process_tree_rss_max_bytes"] > 1


def test_run_limited_stops_process_at_wall_budget(tmp_path) -> None:
    record = run_limited(
        ["/bin/sleep", "5"],
        wall_seconds=1,
        rss_limit_bytes=1024**3,
        sample_seconds=0.05,
        log_path=tmp_path / "attempt.log",
    )

    assert record["limit_reason"] == "wall_clock_limit_exceeded"
    assert record["elapsed_seconds"] < 3


def test_run_limited_catches_short_lived_child_rss_high_water(tmp_path) -> None:
    allocation_mib = 64
    child_code = (
        "import time; time.sleep(0.15); "
        f"payload = bytearray({allocation_mib} * 1024 * 1024); "
        "payload[::4096] = b'x' * (len(payload) // 4096); "
        "time.sleep(0.05)"
    )
    with patch(
        "scripts.run_bounded_local_agent_attempt.process_tree_rss",
        return_value=1,
    ):
        record = run_limited(
            [sys.executable, "-c", child_code],
            wall_seconds=2,
            rss_limit_bytes=32 * 1024**2,
            sample_seconds=0.5,
            log_path=tmp_path / "short-lived.log",
        )

    assert record["sampled_process_tree_rss_max_bytes"] == 1
    assert record["os_wait4_child_max_rss_bytes"] >= allocation_mib * 1024**2
    assert record["rss_limit_exceeded"] is True
    assert record["limit_reason"] == "process_tree_rss_limit_exceeded"


def _pid_is_running(pid: int) -> bool:
    try:
        state = subprocess.check_output(
            ["ps", "-o", "stat=", "-p", str(pid)], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return False
    return bool(state) and not state.startswith("Z")


def test_run_limited_kills_descendant_that_ignores_sigterm(tmp_path) -> None:
    ready_file = tmp_path / "child-ready"
    pid_file = tmp_path / "child.pid"
    child_code = (
        "import pathlib,signal,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(60)"
    )
    parent_code = """\
import pathlib
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]])
pathlib.Path(sys.argv[3]).write_text(str(child.pid))
deadline = time.monotonic() + 5
ready = pathlib.Path(sys.argv[2])
while not ready.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
time.sleep(60)
"""
    command = [
        sys.executable,
        "-c",
        parent_code,
        child_code,
        str(ready_file),
        str(pid_file),
    ]

    def report_over_limit_after_child_is_ready(_root_pid):
        deadline = time.monotonic() + 5
        while not ready_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready_file.exists()
        return 2

    with patch(
        "scripts.run_bounded_local_agent_attempt.process_tree_rss",
        side_effect=report_over_limit_after_child_is_ready,
    ):
        record = run_limited(
            command,
            wall_seconds=10,
            rss_limit_bytes=1,
            sample_seconds=0.01,
            log_path=tmp_path / "attempt.log",
        )

    child_pid = int(pid_file.read_text(encoding="utf-8"))
    try:
        assert record["limit_reason"] == "process_tree_rss_limit_exceeded"
        deadline = time.monotonic() + 3
        while _pid_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _pid_is_running(child_pid)
    finally:
        if _pid_is_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)


@pytest.mark.parametrize(
    ("wall_seconds", "rss_limit_bytes", "sample_seconds"),
    [(0, 1, 0.5), (1, 0, 0.5), (1, 1, 0)],
)
def test_run_limited_rejects_invalid_budget(
    tmp_path, wall_seconds, rss_limit_bytes, sample_seconds,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        run_limited(
            ["/usr/bin/true"],
            wall_seconds=wall_seconds,
            rss_limit_bytes=rss_limit_bytes,
            sample_seconds=sample_seconds,
            log_path=tmp_path / "attempt.log",
        )


def test_run_limited_kills_descendant_after_leader_exits(tmp_path) -> None:
    ready_file = tmp_path / "child-ready"
    pid_file = tmp_path / "child.pid"
    child_code = (
        "import pathlib,signal,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(60)"
    )
    parent_code = """\
import pathlib
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]])
pathlib.Path(sys.argv[3]).write_text(str(child.pid))
ready = pathlib.Path(sys.argv[2])
deadline = time.monotonic() + 5
while not ready.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
raise SystemExit(0 if ready.exists() else 1)
"""
    record = run_limited(
        [sys.executable, "-c", parent_code, child_code, str(ready_file), str(pid_file)],
        wall_seconds=5,
        rss_limit_bytes=1024**3,
        sample_seconds=0.02,
        log_path=tmp_path / "attempt.log",
    )
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    try:
        assert record["status"] == "COMPLETED"
        assert record["exit_code"] == 0
        deadline = time.monotonic() + 3
        while _pid_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _pid_is_running(child_pid)
    finally:
        if _pid_is_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_cli_returns_launch_failure_receipt_without_traceback(tmp_path) -> None:
    record_path = tmp_path / "record.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "run_bounded_local_agent_attempt.py"),
            "--wall-seconds", "2",
            "--max-process-tree-rss-bytes", str(1024**3),
            "--log", str(tmp_path / "attempt.log"),
            "--record", str(record_path),
            "--", str(tmp_path / "missing-command"),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert json.loads(completed.stdout)["status"] == "LAUNCH_FAILED"
    assert json.loads(record_path.read_text(encoding="utf-8"))["status"] == "LAUNCH_FAILED"
