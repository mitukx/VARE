"""Guard lifecycle faults beyond the frozen owner-loss comparison."""
import asyncio
import os
import signal
import subprocess
import sys
import time

import pytest

from vare import runner


@pytest.mark.parametrize('phase', ['launch', 'cleanup'])
def test_cancellation_drains_owned_launch_and_cleanup(tmp_path, monkeypatch, phase):
    grader = tmp_path / 'grade.py'
    grader.write_text('import time; time.sleep(30)\n')
    job = runner.Job('cancel', tmp_path, tmp_path, 5)

    async def scenario():
        original = asyncio.create_subprocess_exec
        entered = asyncio.Event()
        release = asyncio.Event()
        children = []

        async def launch(*args, **kwargs):
            child = await original(*args, **kwargs)
            children.append(child)
            if phase == 'launch':
                entered.set()
                await release.wait()
            else:
                original_wait = child.wait

                async def wait():
                    entered.set()
                    await release.wait()
                    return await original_wait()

                child.wait = wait
            return child

        monkeypatch.setattr(asyncio, 'create_subprocess_exec', launch)
        task = asyncio.create_task(runner.execute(grader, job))
        try:
            if phase == 'cleanup':
                while not children:
                    await asyncio.sleep(.01)
                task.cancel()
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            await asyncio.sleep(.02)
            assert not task.done(), 'cancellation abandoned an owned operation'
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            assert children[0].returncode == -signal.SIGKILL
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_broken_guard_status_fails_closed(tmp_path, monkeypatch):
    # An early guard failure must never convert even a valid-looking stdout
    # into accepted evaluator output. This fault precedes any grader launch.
    original = asyncio.create_subprocess_exec

    async def fail(*args, **kwargs):
        return await original(args[0], '-c', 'import os; os._exit(71)', **kwargs)

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fail)
    forced, code, stdout, stderr = asyncio.run(runner.execute(
        tmp_path / 'missing.py', runner.Job('broken', tmp_path, tmp_path, 1)))
    assert forced == 'grader_error' and code == 71 and stdout == stderr == b''


def test_launch_failure_closes_status_descriptors(tmp_path, monkeypatch):
    opened = []
    original = os.pipe

    def pipe():
        pair = original()
        opened.extend(pair)
        return pair

    async def fail(*args, **kwargs):
        raise OSError('injected spawn failure')

    monkeypatch.setattr(os, 'pipe', pipe)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fail)
    with pytest.raises(OSError, match='injected spawn failure'):
        asyncio.run(runner.execute(tmp_path / 'missing.py', runner.Job('broken', tmp_path, tmp_path)))
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_self_cancelled_launch_does_not_spin(tmp_path, monkeypatch):
    async def fail(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fail)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.execute(tmp_path / 'missing.py', runner.Job('cancel', tmp_path, tmp_path)))


def test_broken_status_pipe_cannot_abandon_descendants(tmp_path):
    grader = tmp_path / 'grade.py'
    ready, release = tmp_path / 'pid', tmp_path / 'release'
    grader.write_text("import pathlib,subprocess,sys,time\n"
                     "p=pathlib.Path(sys.argv[1]); c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); p.write_text(str(c.pid))\n"
                     "while not pathlib.Path(sys.argv[2]).exists(): time.sleep(.01)\n")
    status_read, status_write = os.pipe()
    guard = subprocess.Popen([sys.executable, '-S', str(runner.Path(runner.__file__).with_name('_grader_guard.py')),
                              str(status_write), sys.executable, str(grader), str(ready), str(release)],
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True, pass_fds=(status_write,))
    os.close(status_write)
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        os.close(status_read)
        status_read = None
        release.touch()
        guard.wait(timeout=3)
        assert guard.returncode == -signal.SIGKILL
        pid = int(ready.read_text())
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                                   capture_output=True, text=True).stdout.strip()
            if not state or state.startswith('Z'):
                break
            time.sleep(.01)
        assert not state or state.startswith('Z')
    finally:
        if status_read is not None:
            os.close(status_read)
        guard.stdin.close()
        try:
            os.killpg(guard.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        guard.wait(timeout=3)
