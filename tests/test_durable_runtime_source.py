"""Source-pinned durable resume; real graders, no model-learning claims."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from benchmarks.scheduler.fixtures import make_fixture
from vare import durable
from vare.runner import evaluate


ASSETS = ['__init__.py', '__main__.py', 'runner.py', 'durable.py', '_grader_guard.py']


def state(store):
    with store.connection() as db:
        return ([dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY id')],
                [dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY job_id,token')],
                [dict(r) for r in db.execute('SELECT * FROM events ORDER BY sequence')])


def drift(store, name):
    with store.connection(True) as db:
        db.execute('UPDATE assets SET content=content || ? WHERE name=?',
                   (b'\n# different enrolled revision\n', 'vare/' + name))


@pytest.mark.parametrize('phase', ['claim', 'complete'])
@pytest.mark.parametrize('name', ASSETS)
def test_declared_executor_drift_cannot_change_state(tmp_path, name, phase):
    root, job = make_fixture(tmp_path, 'valid')
    store = durable.Store.create(tmp_path / 'state.db', [job], root)
    if phase == 'complete':
        claim = store.claim('worker', 10)
        outcome = asyncio.run(evaluate(job, root, time.monotonic()))
        assert outcome[0]['status'] == 'candidate_passed'
    drift(store, name)
    before = state(store)
    with pytest.raises(ValueError, match='runtime'):
        if phase == 'claim':
            store.claim('changed', 10)
        else:
            store.complete(claim, outcome)
    assert state(store) == before


@pytest.mark.parametrize('fault', ['missing', 'extra'])
def test_executor_inventory_mismatch_refuses_claim(tmp_path, fault):
    root, job = make_fixture(tmp_path, 'valid')
    store = durable.Store.create(tmp_path / 'state.db', [job], root)
    with store.connection(True) as db:
        if fault == 'missing':
            db.execute("DELETE FROM assets WHERE name='vare/_grader_guard.py'")
        else:
            db.execute('INSERT INTO assets VALUES(?,?)', ('vare/unexpected.py', b'unreviewed'))
    before = state(store)
    with pytest.raises(ValueError, match='runtime'):
        store.claim('changed')
    assert state(store) == before


def test_worker_rejects_drift_before_refresh_or_grading(tmp_path, monkeypatch):
    root, job = make_fixture(tmp_path, 'valid')
    store = durable.Store.create(tmp_path / 'state.db', [job], root)
    drift(store, 'runner.py')
    calls = []
    original = store.refresh

    def refresh(*args):
        calls.append('refresh')
        return original(*args)

    monkeypatch.setattr(store, 'refresh', refresh)
    before = state(store)
    with pytest.raises(ValueError, match='runtime'):
        asyncio.run(durable.worker(store, 'changed'))
    assert calls == [] and state(store) == before


@pytest.mark.parametrize('changed', [True, False], ids=['changed-source', 'matching-control'])
def test_real_crash_resume_pins_exported_executor(tmp_path, changed):
    root, job = make_fixture(tmp_path, 'valid')
    store = durable.Store.create(tmp_path / 'state.db', [job], root)
    source = Path(durable.__file__).resolve().parent
    for label in ['a', 'b']:
        shutil.copytree(source, tmp_path / label / 'vare', ignore=shutil.ignore_patterns('__pycache__'))
    if changed:
        with (tmp_path / 'b/vare/runner.py').open('a') as stream:
            stream.write("\nSOURCE_PIN_DIAGNOSTIC = 'runtime-b'\n")
    script = '''import asyncio,hashlib,json,os,pathlib,sys
from vare import runner
from vare.durable import Store,worker,audit_durable
store=Store(sys.argv[1]); mode=sys.argv[2]
if mode=='crash':
    claim=store.claim('lost-owner',.2)
    print(json.dumps({'token':claim['token']}),flush=True)
    os._exit(42)
result={'used_runner_sha256':hashlib.sha256(pathlib.Path(runner.__file__).read_bytes()).hexdigest(),
        'loaded_marker':getattr(runner,'SOURCE_PIN_DIAGNOSTIC',None)}
try:
    asyncio.run(worker(store,'replacement',2))
except ValueError as exc:
    result.update(status='refused',error_type=type(exc).__name__,message=str(exc))
else:
    result['status']='completed'
output=pathlib.Path(sys.argv[3]);store.export(output)
result['audit']=audit_durable(output)
result['exported_runner_sha256']=hashlib.sha256((output/'snapshot/vare/runner.py').read_bytes()).hexdigest()
result['token']=store.jobs()[0]['token']
print(json.dumps(result),flush=True)
sys.exit(17 if result['status']=='refused' else 0)
'''
    records = []
    evidence = os.environ.get('VARE_RUNTIME_SOURCE_EVIDENCE_DIR')
    label = 'changed-source' if changed else 'matching-control'
    try:
        for runtime, mode, output in [('a', 'crash', 'unused'), ('b', 'resume', 'after-b')]:
            process = subprocess.run([sys.executable, '-c', script, str(store.path), mode, str(tmp_path / output)],
                cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(tmp_path / runtime),
                                  'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True, timeout=15)
            records.append({'runtime': runtime, 'mode': mode, 'exit_code': process.returncode,
                            'stdout': process.stdout, 'stderr': process.stderr})
        assert records[0]['exit_code'] == 42
        resumed = json.loads(records[1]['stdout'])
        assert records[1]['exit_code'] == (17 if changed else 0), resumed
        if changed:
            assert resumed['status'] == 'refused' and resumed['token'] == 1
            assert resumed['loaded_marker'] == 'runtime-b'
            assert resumed['audit']['outcomes'] == {'running': 1}
            assert resumed['used_runner_sha256'] != resumed['exported_runner_sha256']
            restored = subprocess.run([sys.executable, '-c', script, str(store.path), 'resume', str(tmp_path / 'after-a')],
                cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(tmp_path / 'a'),
                                  'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True, timeout=15)
            records.append({'runtime': 'a', 'mode': 'resume', 'exit_code': restored.returncode,
                            'stdout': restored.stdout, 'stderr': restored.stderr})
            assert restored.returncode == 0, restored.stderr
            resumed = json.loads(restored.stdout)
        assert resumed['audit']['outcomes'] == {'candidate_passed': 1}
        assert resumed['audit']['attempts'] == 2 and resumed['token'] == 2
        assert resumed['used_runner_sha256'] == resumed['exported_runner_sha256']
    finally:
        if evidence:
            destination = Path(evidence) / label
            destination.mkdir(parents=True, exist_ok=True)
            (destination / 'processes.json').write_text(json.dumps(records, sort_keys=True, indent=2) + '\n')
            for output in ['after-b', 'after-a']:
                if (tmp_path / output).exists():
                    shutil.copytree(tmp_path / output, destination / output)
