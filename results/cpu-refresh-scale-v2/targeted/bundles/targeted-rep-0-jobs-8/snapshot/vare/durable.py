"""Transactional evaluation state on one host; execution is at least once."""
from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time

from .runner import (ROOT, STATUSES, Job, atomic, classify, digest, encode, evaluate,
                     file_hash, fingerprint, protocol, snapshot_key, strict_json,
                     validate_jobs, write_manifest)


class FenceError(ValueError):
    """An expired or superseded attempt cannot publish a current decision."""


def inputs(job, root):
    task, _, hashes = protocol(job, root)
    return {'protocol': hashes, 'input': fingerprint(job, task), 'task_id': task['task_id']}


def job_from(spec):
    return Job(spec['id'], Path(spec['task_root']), Path(spec['workspace']),
               spec['timeout_seconds'], spec['output_limit_bytes'])


class Store:
    def __init__(self, path):
        original = Path(path)
        if original.is_symlink():
            raise ValueError("linked state database")
        self.path = original.resolve()
        if not self.path.is_file():
            raise ValueError('state database must exist and not be linked')

    @contextmanager
    def connection(self, write=False):
        db = sqlite3.connect(str(self.path), timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA synchronous=FULL')
        try:
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield db
            db.execute('COMMIT')
        except BaseException:
            if db.in_transaction:
                db.execute('ROLLBACK')
            raise
        finally:
            db.close()

    @staticmethod
    def event(db, kind, job_id, token, details=None):
        last = db.execute('SELECT sequence,event_hash FROM events ORDER BY sequence DESC LIMIT 1').fetchone()
        event = {'sequence': last['sequence'] + 1 if last else 0, 'kind': kind,
                 'job_id': job_id, 'token': token, 'details': details or {},
                 'previous_hash': last['event_hash'] if last else '0' * 64}
        event_hash = digest(encode(event))
        db.execute('INSERT INTO events VALUES(?,?,?)', (event['sequence'], encode(event).decode(), event_hash))

    @classmethod
    def create(cls, path, jobs, repository_root=ROOT, max_attempts=3):
        path, root = Path(path).resolve(), repository_root.resolve()
        if path == root or root in path.parents:
            raise ValueError('live state must stay outside the trusted repository')
        validate_jobs(jobs, root, path, 1)
        if not 1 <= max_attempts <= 10:
            raise ValueError('max attempts must be 1..10')
        enrollment = [(j, inputs(j, root)) for j in jobs]
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        db = sqlite3.connect(str(path), isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=FULL')
            db.executescript('''
                CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE assets(name TEXT PRIMARY KEY,content BLOB NOT NULL);
                CREATE TABLE jobs(id TEXT PRIMARY KEY,spec TEXT NOT NULL,expected TEXT NOT NULL,
                    snapshot TEXT NOT NULL,state TEXT NOT NULL,token INTEGER NOT NULL DEFAULT 0,
                    owner TEXT,deadline REAL,max_attempts INTEGER NOT NULL,effective TEXT);
                CREATE TABLE attempts(job_id TEXT NOT NULL,token INTEGER NOT NULL,owner TEXT NOT NULL,
                    state TEXT NOT NULL,record TEXT,data TEXT,stdout BLOB,stderr BLOB,
                    PRIMARY KEY(job_id,token),FOREIGN KEY(job_id) REFERENCES jobs(id));
                CREATE TABLE events(sequence INTEGER PRIMARY KEY,event TEXT NOT NULL,event_hash TEXT NOT NULL);
            ''')
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO meta VALUES(?,?)', ('schema_version', '1'))
            db.execute('INSERT INTO meta VALUES(?,?)', ('repository_root', str(root)))
            for name in ['__init__.py', '__main__.py', 'runner.py', 'durable.py']:
                db.execute('INSERT INTO assets VALUES(?,?)', ('vare/' + name, Path(__file__).with_name(name).read_bytes()))
            for job, expected in enrollment:
                spec = asdict(job)
                spec['task_root'], spec['workspace'] = str(job.task_root.resolve()), str(job.workspace.resolve())
                key = snapshot_key(job)
                task, evaluator, _ = protocol(job, root)
                assets = {'task.json': job.task_root / 'task.json', 'TASK.md': job.task_root / 'TASK.md',
                          'protocol.lock.json': job.task_root / 'protocol.lock.json', 'grade.py': evaluator}
                for name, source in assets.items():
                    file_hash(source)
                    db.execute('INSERT OR IGNORE INTO assets VALUES(?,?)', (key + '/' + name, source.read_bytes()))
                db.execute('INSERT INTO jobs(id,spec,expected,snapshot,state,max_attempts) VALUES(?,?,?,?,?,?)',
                           (job.id, encode(spec).decode(), encode(expected).decode(), key, 'pending', max_attempts))
                cls.event(db, 'enrolled', job.id, 0)
            db.execute('COMMIT')
        finally:
            db.close()
        return cls(path)

    def root(self):
        with self.connection() as db:
            return Path(db.execute("SELECT value FROM meta WHERE key='repository_root'").fetchone()[0])

    def jobs(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY id')]

    def refresh(self, job_id=None):
        """Recompute selected inputs at use time; preserve history and invalidate stale enrollment."""
        root = self.root()
        if job_id is None:
            rows = self.jobs()
        else:
            with self.connection() as db:
                found = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            rows = [dict(found)] if found else []
        for row in rows:
            if row['state'] == 'stale':
                continue
            try:
                current = inputs(job_from(strict_json(row['spec'])), root)
            except (ValueError, KeyError, OSError, TypeError, sqlite3.Error, subprocess.SubprocessError):
                current = None
            if current != strict_json(row['expected']):
                with self.connection(True) as db:
                    actual = db.execute('SELECT * FROM jobs WHERE id=?', (row['id'],)).fetchone()
                    if actual['state'] != 'stale':
                        db.execute("UPDATE jobs SET state='stale',effective='input_stale',owner=NULL,deadline=NULL WHERE id=?", (row['id'],))
                        self.event(db, 'invalidated', row['id'], actual['token'])

    def claim(self, owner, lease_seconds=5, *, now=None):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', owner):
            raise ValueError('unsafe worker ID')
        if not math.isfinite(lease_seconds) or not .2 <= lease_seconds <= 3600:
            raise ValueError('lease must be .2..3600 seconds')
        with self.connection(True) as db:
            now = time.time() if now is None else now
            expired = db.execute("SELECT * FROM jobs WHERE state='running' AND deadline<=? ORDER BY id", (now,)).fetchall()
            for row in expired:
                db.execute("UPDATE attempts SET state='expired' WHERE job_id=? AND token=? AND state='claimed'", (row['id'], row['token']))
                db.execute("UPDATE jobs SET state='pending',owner=NULL,deadline=NULL WHERE id=?", (row['id'],))
                self.event(db, 'expired', row['id'], row['token'])
                if row['token'] >= row['max_attempts']:
                    db.execute("UPDATE jobs SET state='complete',effective='retry_exhausted' WHERE id=?", (row['id'],))
                    self.event(db, 'exhausted', row['id'], row['token'])
            row = db.execute("SELECT * FROM jobs WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
            if not row:
                return None
            token = row['token'] + 1
            db.execute("UPDATE jobs SET state='running',token=?,owner=?,deadline=? WHERE id=?",
                       (token, owner, now + lease_seconds, row['id']))
            db.execute('INSERT INTO attempts(job_id,token,owner,state) VALUES(?,?,?,?)', (row['id'], token, owner, 'claimed'))
            self.event(db, 'claimed', row['id'], token, {'owner': owner, 'deadline': now + lease_seconds})
            return {'job': job_from(strict_json(row['spec'])), 'token': token, 'owner': owner,
                    'expected': strict_json(row['expected'])}

    def renew(self, claim, lease_seconds=5, *, now=None):
        if not math.isfinite(lease_seconds) or not .2 <= lease_seconds <= 3600:
            raise ValueError('invalid lease')
        with self.connection(True) as db:
            now = time.time() if now is None else now
            changed = db.execute("UPDATE jobs SET deadline=? WHERE id=? AND state='running' AND token=? AND owner=? AND deadline>?",
                (now + lease_seconds, claim['job'].id, claim['token'], claim['owner'], now)).rowcount
            return bool(changed)

    def complete(self, claim, outcome, *, now=None):
        record, stdout, stderr, data = outcome
        record = dict(record)
        job = claim['job']
        if record.get('job_id') != job.id or record.get('status') not in STATUSES | {'input_stale'}:
            raise ValueError('invalid terminal record')
        if len(stdout) + len(stderr) > job.output_limit_bytes:
            raise ValueError('record exceeds capture budget')
        # Bind to persisted enrollment, not caller-mutable claim metadata.
        with self.connection() as db:
            enrolled = db.execute('SELECT * FROM jobs WHERE id=?', (job.id,)).fetchone()
            if not enrolled or job != job_from(strict_json(enrolled['spec'])):
                raise FenceError('claim differs from stored job specification')
            expected = strict_json(enrolled['expected'])
            if claim['expected'] != expected:
                raise FenceError('claim differs from stored enrollment')
            task = strict_json(db.execute('SELECT content FROM assets WHERE name=?',
                               (enrolled['snapshot'] + '/task.json',)).fetchone()[0])
        if record['status'] in {'candidate_passed', 'candidate_rejected'}:
            parsed = strict_json(stdout)
            if data != parsed or record.get('input') != expected['input'] or record.get('protocol') != expected['protocol']:
                raise ValueError('terminal record disagrees with enrollment')
            if classify(parsed, record['exit_code'], task, expected['protocol'], expected['input']) != record['status']:
                raise ValueError('invalid accepted-score contract')
            if record.get('task_id') != expected['task_id']:
                raise ValueError('record task ID mismatch')
        record['stdout_sha256'], record['stderr_sha256'] = digest(stdout), digest(stderr)
        payload = (encode(record).decode(), encode(data).decode() if data is not None else None, stdout, stderr)
        # Historical payload identity stays immutable; freshness changes effective state.
        self.refresh(job.id)
        with self.connection(True) as db:
            now = time.time() if now is None else now
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job.id,)).fetchone()
            attempt = db.execute('SELECT * FROM attempts WHERE job_id=? AND token=?', (job.id, claim['token'])).fetchone()
            if not attempt or attempt['owner'] != claim['owner']:
                raise FenceError('attempt identity mismatch')
            if attempt['state'] in {'committed', 'fenced'}:
                original = (attempt['record'], attempt['data'], attempt['stdout'], attempt['stderr'])
                if original != payload:
                    raise FenceError('conflicting duplicate completion')
                return {'committed': attempt['state'] == 'committed', 'duplicate': True}
            live = row['state'] == 'running' and row['token'] == claim['token'] and row['owner'] == claim['owner'] and row['deadline'] > now
            db.execute('UPDATE attempts SET state=?,record=?,data=?,stdout=?,stderr=? WHERE job_id=? AND token=?',
                       ('committed' if live else 'fenced', *payload, job.id, claim['token']))
            if live:
                db.execute("UPDATE jobs SET state='complete',effective=?,owner=NULL,deadline=NULL WHERE id=?", (record['status'], job.id))
            self.event(db, 'committed' if live else 'fenced', job.id, claim['token'], {'status': record['status'], 'record_sha256': digest(encode(record))})
            return {'committed': live, 'duplicate': False}

    def unfinished(self):
        with self.connection() as db:
            return db.execute("SELECT count(*) FROM jobs WHERE state IN ('pending','running')").fetchone()[0]

    def export(self, output):
        """A consistent read transaction; exports never contain live workspace paths."""
        self.refresh()
        output = Path(output).resolve()
        root = self.root()
        specs = [job_from(strict_json(r['spec'])) for r in self.jobs()]
        validate_jobs(specs, root, output, 1)
        if output == self.path or output in self.path.parents:
            raise ValueError('export overlaps live database')
        output.mkdir(parents=True, exist_ok=False)
        with self.connection() as db:
            state = {'schema_version': 1, 'jobs': [], 'events': []}
            for row in db.execute('SELECT * FROM jobs ORDER BY id'):
                spec = strict_json(row['spec'])
                state['jobs'].append({'id': row['id'], 'state': row['state'], 'token': row['token'],
                    'effective': row['effective'], 'snapshot': row['snapshot'], 'expected': strict_json(row['expected']),
                    'timeout_seconds': spec['timeout_seconds'], 'output_limit_bytes': spec['output_limit_bytes'],
                    'max_attempts': row['max_attempts']})
            for row in db.execute('SELECT * FROM events ORDER BY sequence'):
                state['events'].append({**strict_json(row['event']), 'event_sha256': row['event_hash']})
            for row in db.execute('SELECT * FROM assets ORDER BY name'):
                atomic(output / 'snapshot' / row['name'], row['content'])
            for row in db.execute('SELECT * FROM attempts ORDER BY job_id,token'):
                directory = output / 'attempts' / row['job_id'] / str(row['token'])
                atomic(directory / 'attempt.json', encode({'job_id': row['job_id'], 'token': row['token'], 'owner': row['owner'], 'state': row['state']}))
                if row['record'] is not None:
                    atomic(directory / 'record.json', row['record'].encode())
                    atomic(directory / 'stdout.bin', row['stdout'])
                    atomic(directory / 'stderr.bin', row['stderr'])
                    if row['data'] is not None:
                        atomic(directory / 'grader.json', row['data'].encode())
        atomic(output / 'state.json', encode(state))
        write_manifest(output)
        return {'jobs': len(state['jobs']), 'outcomes': dict(Counter(r['effective'] or r['state'] for r in state['jobs']))}


async def worker(store, owner, lease_seconds=5):
    await asyncio.to_thread(store.refresh)
    root = store.root()
    while True:
        claim = await asyncio.to_thread(store.claim, owner, lease_seconds)
        if claim is None:
            if not await asyncio.to_thread(store.unfinished):
                return
            await asyncio.sleep(min(.1, lease_seconds / 4))
            continue
        stop = asyncio.Event()
        lost = asyncio.Event()
        async def heartbeat():
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), lease_seconds / 3)
                except asyncio.TimeoutError:
                    if not await asyncio.to_thread(store.renew, claim, lease_seconds):
                        lost.set()
                        return
        pulse = asyncio.create_task(heartbeat())
        grading = None
        try:
            try:
                fresh = await asyncio.to_thread(inputs, claim['job'], root)
            except Exception:
                fresh = None
            if fresh != claim['expected']:
                result = ({'schema_version': 1, 'job_id': claim['job'].id, 'status': 'input_stale'}, b'', b'', None)
            else:
                grading = asyncio.create_task(evaluate(claim['job'], root, time.monotonic()))
                lease_lost = asyncio.create_task(lost.wait())
                done, _ = await asyncio.wait([grading, lease_lost], return_when=asyncio.FIRST_COMPLETED)
                if lease_lost in done:
                    grading.cancel()
                lease_lost.cancel()
                await asyncio.gather(lease_lost, return_exceptions=True)
                result = await grading
                if result[0].get('input') not in (None, claim['expected']['input']):
                    result[0]['status'] = 'input_stale'
            await asyncio.to_thread(store.complete, claim, result)
        finally:
            if grading is not None and not grading.done():
                grading.cancel()
                await asyncio.gather(grading, return_exceptions=True)
            stop.set()
            await pulse


def worker_entry(path, owner, lease_seconds=5):
    asyncio.run(worker(Store(path), owner, lease_seconds))


def run_workers(path, workers=1, lease_seconds=5):
    import multiprocessing
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('workers must be 1..64')
    context = multiprocessing.get_context('spawn')
    children = [context.Process(target=worker_entry, args=(str(path), 'worker-' + str(i), lease_seconds)) for i in range(workers)]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join()
        if any(child.exitcode != 0 for child in children):
            raise RuntimeError('worker failed; state is retained for recovery')
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
                if child.is_alive():
                    child.kill()
                    child.join()


def audit_durable(output):
    output = Path(output).resolve()
    if any(p.is_symlink() for p in output.rglob('*')):
        raise ValueError('linked evidence')
    manifest = strict_json((output / 'manifest.json').read_bytes())
    actual = {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*') if p.is_file() and p != output / 'manifest.json'}
    if actual != manifest['files']:
        raise ValueError('manifest mismatch')
    state = strict_json((output / 'state.json').read_bytes())
    jobs = {j['id']: j for j in state['jobs']}
    if len(jobs) != len(state['jobs']) or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name) for name in jobs):
        raise ValueError('unsafe or duplicated job ID')
    if any(not re.fullmatch(r'[a-f0-9]{16}', j['snapshot']) for j in jobs.values()):
        raise ValueError('unsafe snapshot identifier')
    replay = {name: {'state': None, 'token': 0, 'effective': None} for name in jobs}
    attempt_states = {}
    owners = {}
    previous = '0' * 64
    for index, raw in enumerate(state['events']):
        event = dict(raw)
        claimed_hash = event.pop('event_sha256')
        if event['sequence'] != index or event['previous_hash'] != previous or digest(encode(event)) != claimed_hash:
            raise ValueError('event chain mismatch')
        previous = claimed_hash
        name, token, kind = event['job_id'], event['token'], event['kind']
        row = replay[name]
        if kind == 'enrolled':
            if row['state'] is not None or token != 0:
                raise ValueError('invalid enrollment')
            row['state'] = 'pending'
        elif kind == 'claimed':
            if row['state'] != 'pending' or token != row['token'] + 1 or token > jobs[name]['max_attempts']:
                raise ValueError('invalid claim transition')
            row.update(state='running', token=token)
            attempt_states[name, token] = 'claimed'
            owners[name, token] = event['details']['owner']
        elif kind == 'expired':
            if row['state'] != 'running' or token != row['token']:
                raise ValueError('invalid expiry')
            row['state'] = 'pending'
            if attempt_states[name, token] == 'claimed':
                attempt_states[name, token] = 'expired'
        elif kind == 'exhausted':
            if row['state'] != 'pending' or token != jobs[name]['max_attempts']:
                raise ValueError('invalid exhaustion')
            row.update(state='complete', effective='retry_exhausted')
        elif kind == 'invalidated':
            if row['state'] in {None, 'stale'} or row['token'] != token:
                raise ValueError('invalid freshness transition')
            row.update(state='stale', effective='input_stale')
        elif kind in {'committed', 'fenced'}:
            if (name, token) not in attempt_states or attempt_states[name, token] in {'committed', 'fenced'}:
                raise ValueError('completion without claim')
            if kind == 'committed':
                if row['state'] != 'running' or row['token'] != token:
                    raise ValueError('unfenced terminal transition')
                row.update(state='complete', effective=event['details']['status'])
            attempt_states[name, token] = kind
            directory = output / 'attempts' / name / str(token)
            record_bytes = (directory / 'record.json').read_bytes()
            record = strict_json(record_bytes)
            if (record.get('job_id') != name or record.get('status') not in STATUSES | {'input_stale'} or
                digest(record_bytes) != event['details']['record_sha256'] or record['status'] != event['details']['status']):
                raise ValueError('event/record mismatch')
            for stream in ['stdout', 'stderr']:
                if file_hash(directory / (stream + '.bin')) != record[stream + '_sha256']:
                    raise ValueError('stream mismatch')
            if record['status'] in {'candidate_passed', 'candidate_rejected'}:
                snapshot = output / 'snapshot' / jobs[name]['snapshot']
                task = strict_json((snapshot / 'task.json').read_bytes())
                hashes = {'task_json_sha256': file_hash(snapshot / 'task.json'), 'task_brief_sha256': file_hash(snapshot / 'TASK.md'),
                          'evaluator_sha256': file_hash(snapshot / 'grade.py')}
                lock = strict_json((snapshot / 'protocol.lock.json').read_bytes())
                if hashes != lock['locked_hashes']:
                    raise ValueError('snapshot lock mismatch')
                hashes['protocol_lock_sha256'] = file_hash(snapshot / 'protocol.lock.json')
                expected = jobs[name]['expected']
                data = strict_json((directory / 'stdout.bin').read_bytes())
                if (hashes != expected['protocol'] or record['task_id'] != expected['task_id'] or
                    task['task_id'] != lock['task_id'] or record['input'] != expected['input'] or record['protocol'] != hashes or
                    classify(data, record['exit_code'], task, hashes, expected['input']) != record['status']):
                    raise ValueError('accepted record does not bind to enrollment')
                if data != strict_json((directory / 'grader.json').read_bytes()):
                    raise ValueError('raw/parsed result mismatch')
        else:
            raise ValueError('unknown event')
    for name, row in replay.items():
        if row['state'] is None:
            raise ValueError('exported job was never enrolled')
        if row != {key: jobs[name][key] for key in row}:
            raise ValueError('exported job state does not replay')
    for (name, token), expected_state in attempt_states.items():
        attempt = strict_json((output / 'attempts' / name / str(token) / 'attempt.json').read_bytes())
        if (attempt['state'] != expected_state or attempt['job_id'] != name or attempt['token'] != token or
            attempt['owner'] != owners[name, token]):
            raise ValueError('attempt state mismatch')
    return {'status': 'verified', 'jobs': len(jobs), 'attempts': len(attempt_states),
            'events': len(state['events']), 'outcomes': dict(Counter(j['effective'] or j['state'] for j in jobs.values()))}


def main():
    import argparse
    import json
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('durable-init')
    create.add_argument('--plan', type=Path, required=True)
    create.add_argument('--state', type=Path, required=True)
    create.add_argument('--max-attempts', type=int, default=3)
    work = sub.add_parser('durable-work')
    work.add_argument('--state', type=Path, required=True)
    work.add_argument('--workers', type=int, default=1)
    work.add_argument('--lease-seconds', type=float, default=5)
    export = sub.add_parser('durable-export')
    export.add_argument('--state', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    verify = sub.add_parser('durable-audit')
    verify.add_argument('bundle', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'durable-init':
            path = args.plan.resolve()
            plan = strict_json(path.read_bytes())
            jobs = [Job(j['id'], (path.parent / j['task_root']).resolve(),
                        (path.parent / j['workspace']).resolve(), j.get('timeout_seconds', 30),
                        j.get('output_limit_bytes', 1048576)) for j in plan['jobs']]
            store = Store.create(args.state, jobs, max_attempts=args.max_attempts)
            result = {'status': 'enrolled', 'jobs': len(store.jobs())}
        elif args.command == 'durable-work':
            run_workers(args.state, args.workers, args.lease_seconds)
            result = {'status': 'finished', 'outcomes': dict(Counter(r['effective'] or r['state'] for r in Store(args.state).jobs()))}
        elif args.command == 'durable-export':
            result = Store(args.state).export(args.output)
        else:
            result = audit_durable(args.bundle)
        print(encode(result).decode(), end='')
        return 0 if args.command != 'durable-work' or all(k in {'candidate_passed', 'candidate_rejected'} for k in result['outcomes']) else 1
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, sqlite3.Error) as exc:
        print(json.dumps({'error': type(exc).__name__, 'message': str(exc)}), file=sys.stderr)
        return 1
