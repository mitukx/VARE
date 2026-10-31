import subprocess
import sys
import time
from pathlib import Path

import scripts.run_mlsys_vllm_64830_qualification_arm as qualification_arm
from scripts.run_mlsys_vllm_64830_qualification_arm import build_pytest_command


def test_pytest_internal_log_is_written_inside_the_arm_output_directory(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "baseline.sb"
    python = tmp_path / "python"
    test_file = tmp_path / "test_case.py"
    pytest_config = tmp_path / "pytest.ini"
    output_dir = tmp_path / "baseline-attempt-002"
    basetemp = output_dir / "pytest-tmp"
    log_file = output_dir / "pytest-internal.log"
    test_ids = ["test_case[one]", "test_case[two]"]

    command = build_pytest_command(
        profile,
        python,
        test_file,
        pytest_config,
        basetemp,
        log_file,
        test_ids,
    )

    assert command[command.index("-o") + 1] == f"log_file={log_file}"
    assert command[command.index("--basetemp") + 1] == str(basetemp)
    assert command[-2:] == [
        f"{test_file}::test_case[one]",
        f"{test_file}::test_case[two]",
    ]
    assert "/dev/null" not in command


def test_wait4_records_a_short_lived_child_high_water_mark() -> None:
    allocation_mib = 64
    child_code = (
        f"payload = bytearray({allocation_mib} * 1024 * 1024); "
        "payload[::4096] = b'x' * (len(payload) // 4096); "
        "import time; time.sleep(0.05)"
    )
    process = subprocess.Popen([sys.executable, "-c", child_code])

    usage = qualification_arm.wait_for_child(process, block=True)

    assert process.returncode == 0
    assert usage is not None
    assert qualification_arm.wait4_max_rss_bytes(usage) >= allocation_mib * 1024**2


def test_wait4_rss_units_follow_platform_convention(monkeypatch) -> None:
    class Usage:
        ru_maxrss = 128

    monkeypatch.setattr(qualification_arm.sys, "platform", "darwin")
    assert qualification_arm.wait4_max_rss_bytes(Usage()) == 128

    monkeypatch.setattr(qualification_arm.sys, "platform", "linux")
    assert qualification_arm.wait4_max_rss_bytes(Usage()) == 128 * 1024


def test_process_group_cleanup_stops_child_left_by_exited_leader(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import os, time; "
        f"open({str(child_pid_path)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    leader_code = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_code], start_new_session=True
    )
    try:
        usage = qualification_arm.wait_for_child(leader, block=True)
        assert leader.returncode == 0
        assert usage is not None
        deadline = time.monotonic() + 2
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        child_pid = int(child_pid_path.read_text())

        qualification_arm.kill_process_group(leader.pid)

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                state = subprocess.check_output(
                    ["ps", "-o", "stat=", "-p", str(child_pid)], text=True
                ).strip()
            except subprocess.CalledProcessError:
                state = ""
            if not state or state.startswith("Z"):
                break
            time.sleep(0.02)
        assert not state or state.startswith("Z")
    finally:
        qualification_arm.kill_process_group(leader.pid)
