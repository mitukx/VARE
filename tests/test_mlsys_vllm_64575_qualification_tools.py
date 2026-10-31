import ast
import os
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from scripts.prepare_mlsys_vllm_64575_qualification_overlay import (
    build_overlay,
    extract_source_segment,
)
from scripts.run_mlsys_vllm_64575_qualification_arm import (
    ARM_CHOICES,
    build_pytest_command,
    create_attempt_output_dir,
    process_tree_rss,
    stop_process_group,
)


def test_overlay_extracts_only_frozen_helper_and_regression_class() -> None:
    source = '''
def _make_request(messages, **kwargs):
    return Request(messages=messages, **kwargs)

class UnrelatedTests:
    def test_other(self):
        assert False

class TestToolDefinitions:
    def test_custom_tool(self):
        assert True
'''

    overlay, hashes = build_overlay(source)

    assert "def _make_request(messages, **kwargs):" in overlay
    assert "class TestToolDefinitions:" in overlay
    assert "class UnrelatedTests:" not in overlay
    assert len(hashes["overlay_sha256"]) == 64


def test_overlay_fails_closed_when_expected_test_class_is_missing() -> None:
    with pytest.raises(ValueError, match="TestToolDefinitions"):
        build_overlay("def _make_request(messages, **kwargs):\n    return None\n")


def test_extract_source_segment_requires_one_top_level_definition() -> None:
    source = "def target():\n    return 1\n\ndef target():\n    return 2\n"

    with pytest.raises(ValueError, match="exactly one"):
        extract_source_segment(source, "target", ast.FunctionDef)


def test_process_tree_rss_sums_only_root_and_descendants() -> None:
    ps_output = "10 1 100\n11 10 50\n12 1 200\n13 11 25\n"
    with patch(
        "scripts.run_mlsys_vllm_64575_qualification_arm.subprocess.check_output",
        return_value=ps_output,
    ):
        assert process_tree_rss(10) == (100 + 50 + 25) * 1024


def test_pytest_log_file_stays_inside_the_arm_writable_directory(tmp_path) -> None:
    profile = tmp_path / "arm.sb"
    test_file = tmp_path / "test_case.py"
    basetemp = tmp_path / "pytest-tmp"
    log_file = tmp_path / "pytest-internal.log"

    command = build_pytest_command(profile, "python", test_file, basetemp, log_file)

    assert command[command.index("-o") + 1] == f"log_file={log_file}"
    assert command[-2] == str(basetemp)


def test_candidate_arm_is_recorded_separately_from_qualification_arms() -> None:
    assert "agent-candidate" in ARM_CHOICES
    assert ARM_CHOICES[:3] == ("baseline", "wrong-mutant", "upstream-fixed")


def test_attempt_output_dir_refuses_to_overwrite_existing_evidence(tmp_path) -> None:
    output = tmp_path / "attempt-001"
    output.mkdir()
    receipt = output / "attempt.json"
    receipt.write_text('{"status":"kept"}', encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_attempt_output_dir(output)

    assert receipt.read_text(encoding="utf-8") == '{"status":"kept"}'


def test_stop_process_group_kills_descendant_after_leader_exits(tmp_path) -> None:
    pid_file = tmp_path / "child.pid"
    ready_file = tmp_path / "child-ready"
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
    proc = subprocess.Popen(
        [sys.executable, "-c", parent_code, child_code, str(ready_file), str(pid_file)],
        start_new_session=True,
    )
    return_code = proc.wait(timeout=5)
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    try:
        assert return_code == 0
        stop_process_group(proc)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child_pid)],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if not status or status.startswith("Z"):
                break
            time.sleep(0.02)
        assert not status or status.startswith("Z")
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
