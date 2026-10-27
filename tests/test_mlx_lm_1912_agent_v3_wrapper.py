from __future__ import annotations

import subprocess
import sys

from scripts.run_mlx_lm_1912_qwen25_agent_v3 import terminate_processes_from_traceback


def test_monitor_exception_stops_escaped_child_process_group() -> None:
    child: subprocess.Popen[bytes] | None = None

    def raise_after_start() -> None:
        nonlocal child
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        child = process
        raise KeyError("frozen resource key absent")

    try:
        raise_after_start()
    except KeyError as exc:
        stopped = terminate_processes_from_traceback(exc.__traceback__)

    assert child is not None
    assert stopped == [{"pid": child.pid, "returncode": -15}]
    assert child.poll() == -15
