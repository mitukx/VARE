"""Regression checks for externally owned coding-agent processes."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from vare.environments import CommandWorkspaceAgent


def test_command_agent_timeout_kills_child_and_descendants(tmp_path: Path):
    script = tmp_path / "long.py"
    marker = tmp_path / "unexpected"
    script.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        "'import time, pathlib; time.sleep(1); pathlib.Path(%r).write_text(\"escaped\")' % sys.argv[1]])\n"
        "time.sleep(20)\n",
        encoding="utf-8",
    )
    agent = CommandWorkspaceAgent((sys.executable, str(script), str(marker)), timeout_s=0.15)
    with pytest.raises(subprocess.TimeoutExpired):
        asyncio.run(agent.run(prompt="x", workspace=tmp_path, step=0))
    time.sleep(1.2)
    assert not marker.exists(), "timed-out agent descendant continued executing"


def test_command_agent_cancellation_kills_child_and_descendants(tmp_path: Path):
    script = tmp_path / "long.py"
    marker = tmp_path / "unexpected"
    signal = tmp_path / "started"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text('started')\n"
        "subprocess.Popen([sys.executable, '-c', "
        "'import time, pathlib; time.sleep(1); pathlib.Path(%r).write_text(\"escaped\")' % sys.argv[2]])\n"
        "time.sleep(20)\n",
        encoding="utf-8",
    )
    agent = CommandWorkspaceAgent(
        (sys.executable, str(script), str(signal), str(marker)), timeout_s=15,
    )

    async def scenario():
        task = asyncio.create_task(agent.run(prompt="x", workspace=tmp_path, step=0))
        async with asyncio.timeout(5):
            while not signal.exists():
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    time.sleep(1.2)
    assert not marker.exists(), "cancelled agent descendant continued executing"


def test_command_agent_output_limit_is_fail_closed(tmp_path: Path):
    script = tmp_path / "loud.py"
    script.write_text("import sys\nsys.stdout.write('x' * 300000)\n", encoding="utf-8")
    agent = CommandWorkspaceAgent((sys.executable, str(script)), timeout_s=5)
    with pytest.raises(RuntimeError, match="output budget"):
        asyncio.run(agent.run(prompt="x", workspace=tmp_path, step=0))


def test_command_agent_returns_actual_process_status(tmp_path: Path):
    script = tmp_path / "status.py"
    script.write_text("import sys\nprint('hello')\nprint('oops', file=sys.stderr)\nsys.exit(13)\n", encoding="utf-8")
    agent = CommandWorkspaceAgent((sys.executable, str(script)), timeout_s=5)
    result = json.loads(asyncio.run(agent.run(prompt="x", workspace=tmp_path, step=0)))
    assert result["returncode"] == 13
    assert "hello" in result["stdout"] and "oops" in result["stderr"]
