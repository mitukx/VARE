"""Fresh process-loss fixtures; no model/task-capability claims."""
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from benchmarks.scheduler.fixtures import make_fixture
from vare.durable import Store, audit_durable, worker
from vare.runner import digest, encode


def running(pid):
    observed = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                              capture_output=True, text=True)
    assert observed.returncode in (0, 1) and not observed.stderr, observed.stderr
    state = observed.stdout.strip()
    return bool(state and not state.startswith('Z'))


def until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.02)
    return False


def make_blocked_job(parent):
    root, job = make_fixture(parent, 'valid', timeout=15)
    grader = job.task_root / 'grade.py'
    prefix = '''import json, os, pathlib, subprocess, sys, time
workspace = pathlib.Path(sys.argv[sys.argv.index('--workspace') + 1])
ready = workspace / 'ready.json'
if not ready.exists():
    child_code = "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]);\\nwhile True: p.write_text(str(time.monotonic())); time.sleep(.02)"
    child = subprocess.Popen([sys.executable, '-c', child_code, str(workspace / 'heartbeat')])
    while not (workspace / 'heartbeat').exists(): time.sleep(.01)
    temporary = workspace / 'ready.tmp'
    temporary.write_text(json.dumps({'grader': os.getpid(), 'child': child.pid, 'group': os.getpgrp(), 'parent': os.getppid()}))
    os.replace(temporary, ready)
    while True: time.sleep(.02)
'''
    grader.write_text(prefix + grader.read_text())
    lock_path = job.task_root / 'protocol.lock.json'
    lock = json.loads(lock_path.read_text())
    lock['locked_hashes']['evaluator_sha256'] = digest(grader.read_bytes())
    lock_path.write_bytes(encode(lock))
    return root, job


@pytest.mark.parametrize('fault', ['SIGKILL', 'SIGTERM', 'exit'])
def test_owner_loss_stops_grader_and_descendant_then_resumes(tmp_path, fault):
    root, job = make_blocked_job(tmp_path)
    store = Store.create(tmp_path / 'state.db', [job], root)
    # Match the imported implementation, even when pytest compares a pinned tree.
    import vare.runner
    source = Path(vare.runner.__file__).resolve().parents[1]
    script = '''import asyncio,os,signal,sys
from vare.durable import Store,worker
signal.signal(signal.SIGUSR1, lambda *_: os._exit(73))
asyncio.run(worker(Store(sys.argv[1]), 'lost-owner', 1))
'''
    owner = subprocess.Popen([sys.executable, '-c', script, str(store.path)],
                             cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(source)},
                             start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    receipt = {}
    evidence = {'fault': fault}
    try:
        assert until(lambda: (job.workspace / 'ready.json').exists()), 'grader never became ready'
        receipt = json.loads((job.workspace / 'ready.json').read_text())
        assert running(receipt['grader']) and running(receipt['child'])
        assert store.jobs()[0]['state'] == 'running'
        injected = time.monotonic()
        owner.send_signal(signal.SIGUSR1 if fault == 'exit' else getattr(signal, fault))
        owner.wait(timeout=5)
        stopped = until(lambda: not running(receipt['grader']) and not running(receipt['child']), 2)
        evidence.update(receipt=receipt, owner_exit=owner.returncode,
                        stopped=stopped,
                        stop_seconds=time.monotonic() - injected)
        assert stopped, 'owner died but its grader or descendant kept running'
        heartbeat = (job.workspace / 'heartbeat').read_text()
        time.sleep(.1)
        assert (job.workspace / 'heartbeat').read_text() == heartbeat
        started = time.monotonic()
        asyncio.run(worker(store, 'replacement', 2))
        evidence['resume_seconds'] = time.monotonic() - started
        exported = tmp_path / 'export'
        store.export(exported)
        replay = audit_durable(exported)
        state = json.loads((exported / 'state.json').read_text())
        assert replay['outcomes'] == {'candidate_passed': 1}
        assert replay['attempts'] == 2 and state['jobs'][0]['token'] == 2
        assert [event['kind'] for event in state['events']] == [
            'enrolled', 'claimed', 'expired', 'claimed', 'committed']
        assert json.loads((exported / 'attempts/valid/1/attempt.json').read_text())['state'] == 'expired'
        evidence['replay'] = replay
        destination = os.environ.get('VARE_OWNER_LOSS_EVIDENCE_DIR')
        if destination:
            import shutil
            shutil.copytree(exported, Path(destination) / fault / 'export')
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if receipt:
            try:
                os.killpg(receipt['group'], signal.SIGKILL)
            except ProcessLookupError:
                pass
        destination = os.environ.get('VARE_OWNER_LOSS_EVIDENCE_DIR')
        if destination:
            folder = Path(destination) / fault
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'observation.json').write_bytes(encode(evidence))
