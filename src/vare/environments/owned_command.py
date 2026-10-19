"""Own a POSIX command-agent process group through cancellation and owner loss.

The existing grader guard owns its session and kills ordinary descendants if
the controller closes stdin *or is SIGKILLed*. This is not hostile-code
containment: an agent may escape its group or access same-UID host resources.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys


async def run_owned_command(
    argv: list[str], *, cwd: Path, env: dict[str, str], timeout_s: float,
    output_limit_bytes: int = 262_144,
) -> tuple[int, str, str]:
    """Run one externally configured tool agent; never abandon a live child.

    Retain at most 100 KB per stream and reject excessive *total* output.
    The guard's separate four-byte status pipe avoids conflating its exit with
    the agent's return code. A closed owner pipe terminates the entire ordinary
    process group, including on uncatchable controller death.
    """
    if os.name != "posix":
        raise RuntimeError("owned command agents require POSIX process groups")
    if not argv or timeout_s <= 0 or output_limit_bytes <= 0:
        raise ValueError("invalid bounded agent execution arguments")

    status_read, status_write = os.pipe()
    guard = Path(__file__).resolve().parents[1] / "_grader_guard.py"
    proc: asyncio.subprocess.Process | None = None
    pumps: list[asyncio.Task[None]] = []
    status_task: asyncio.Task[bytes] | None = None
    excess_task: asyncio.Task[bool] | None = None
    outputs = [bytearray(), bytearray()]
    total = 0
    overflow = asyncio.Event()

    async def pump(stream: asyncio.StreamReader, index: int) -> None:
        nonlocal total
        while block := await stream.read(16_384):
            total += len(block)
            outputs[index].extend(block)
            if len(outputs[index]) > 100_000:
                del outputs[index][:-100_000]
            if total > output_limit_bytes:
                overflow.set()

    try:
        # A cancel during process creation must still acquire its handle, so
        # it can close the guardian's owner pipe rather than orphaning work.
        launch = asyncio.create_task(asyncio.create_subprocess_exec(
            sys.executable, "-B", str(guard), str(status_write), *argv,
            cwd=cwd, env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            pass_fds=(status_write,), start_new_session=True,
        ))
        cancelled = False
        while True:
            try:
                proc = await asyncio.shield(launch)
                break
            except asyncio.CancelledError:
                cancelled = True
                if launch.cancelled():
                    raise
        os.close(status_write)
        status_write = -1
        if cancelled:
            raise asyncio.CancelledError()
        assert proc.stdout is not None and proc.stderr is not None
        pumps = [asyncio.create_task(pump(proc.stdout, 0)),
                 asyncio.create_task(pump(proc.stderr, 1))]
        status_task = asyncio.create_task(asyncio.to_thread(os.read, status_read, 4))
        excess_task = asyncio.create_task(overflow.wait())
        done, _ = await asyncio.wait(
            (status_task, excess_task), timeout=timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if excess_task in done:
            raise RuntimeError("agent exceeded bounded output budget")
        if status_task not in done:
            raise subprocess.TimeoutExpired(argv, timeout_s)
        raw = status_task.result()
        if len(raw) != 4:
            raise RuntimeError("agent ownership guard exited without status")
        returncode = struct.unpack("!i", raw)[0]
    finally:
        if status_write >= 0:
            os.close(status_write)
        if proc is not None:
            assert proc.stdin is not None
            proc.stdin.close()  # EOF kills ordinary descendants, including after cancellation.

            async def finish() -> None:
                pending: list[Awaitable[object]] = [proc.wait(), *pumps]
                if status_task is not None:
                    pending.append(status_task)
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True), timeout=3,
                    )
                except asyncio.TimeoutError:
                    # Guard failure fallback; do not abandon a running session.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    if proc.returncode is None:
                        proc.kill()
                    for task in pumps:
                        task.cancel()
                    if status_task is not None:
                        status_task.cancel()
                    await asyncio.gather(proc.wait(), *pumps, return_exceptions=True)

            closing = asyncio.create_task(finish())
            interrupted = False
            while True:
                try:
                    await asyncio.shield(closing)
                    break
                except asyncio.CancelledError:
                    interrupted = True
            if excess_task is not None:
                excess_task.cancel()
                await asyncio.gather(excess_task, return_exceptions=True)
            os.close(status_read)
            if interrupted:
                raise asyncio.CancelledError()
        else:
            os.close(status_read)

    if overflow.is_set():
        raise RuntimeError("agent exceeded bounded output budget")
    return (
        returncode,
        outputs[0].decode("utf-8", errors="replace"),
        outputs[1].decode("utf-8", errors="replace"),
    )
