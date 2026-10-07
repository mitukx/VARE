"""Bounded POSIX grader execution. This is not a hostile-code sandbox."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {'candidate_passed', 'candidate_rejected', 'protocol_error', 'source_error',
            'timeout', 'output_limit', 'invalid_output', 'grader_error',
            'provenance_mismatch', 'workspace_changed', 'cancelled', 'internal_error'}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def floating(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('nonfinite JSON number')
        return number

    def constant(value):
        raise ValueError('nonfinite JSON constant')

    return json.loads(data, object_pairs_hook=pairs, parse_float=floating,
                      parse_constant=constant)


def atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_hash(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('required input is missing, linked, or oversized')
    return digest(path.read_bytes())


def inside(path: Path, root: Path):
    return path == root or root in path.parents


def owned_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('unsafe source path')
    path = root / relative
    # Reject links in intermediate directories as well as at the leaf.
    if any(p.is_symlink() for p in [path, *path.parents] if inside(p, root)):
        raise ValueError('symlinked input')
    if not inside(path.resolve(), root.resolve()):
        raise ValueError('input escapes root')
    return path


@dataclass(frozen=True)
class Job:
    id: str
    task_root: Path
    workspace: Path
    timeout_seconds: float = 30.0
    output_limit_bytes: int = 1024 * 1024


def validate_jobs(jobs, root, output, workers):
    if os.name != 'posix':
        raise ValueError('POSIX process groups are required')
    if type(workers) is not int or not 1 <= workers <= 64 or not jobs:
        raise ValueError('workers must be 1..64 and jobs nonempty')
    seen = set()
    for job in jobs:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', job.id) or job.id in seen:
            raise ValueError('job IDs must be unique safe names')
        seen.add(job.id)
        if not math.isfinite(job.timeout_seconds) or not 0 < job.timeout_seconds <= 900:
            raise ValueError('timeout must be finite and in (0,900]')
        if type(job.output_limit_bytes) is not int or not 128 <= job.output_limit_bytes <= 16 * 1024 * 1024:
            raise ValueError('output limit must be 128 bytes..16 MiB')
        task = job.task_root.resolve()
        workspace = job.workspace.resolve()
        if not inside(task, root) or inside(workspace, root) or inside(root, workspace):
            raise ValueError('task must be in trusted root; workspace must be outside it')
        if inside(output, workspace) or inside(workspace, output) or inside(task, output):
            raise ValueError('output must not overlap protected inputs or workspaces')


def protocol(job, root):
    task_root = job.task_root.resolve()
    task_path = owned_file(task_root, 'task.json')
    file_hash(task_path)
    task = strict_json(task_path.read_bytes())
    lock_path = owned_file(task_root, 'protocol.lock.json')
    file_hash(lock_path)
    lock = strict_json(lock_path.read_bytes())
    evaluator = owned_file(root, task['grader']['path'])
    actual = {'task_json_sha256': file_hash(task_path),
              'task_brief_sha256': file_hash(owned_file(task_root, 'TASK.md')),
              'evaluator_sha256': file_hash(evaluator)}
    if task['task_id'] != lock['task_id'] or actual != lock['locked_hashes']:
        raise ValueError('locked protocol mismatch')
    if not task['source']['files'] or len(set(task['source']['files'])) != len(task['source']['files']):
        raise ValueError('invalid source file list')
    actual['protocol_lock_sha256'] = file_hash(lock_path)
    return task, evaluator, actual


def git(workspace, args):
    return subprocess.run(['git', '--no-pager', '-C', str(workspace), *args],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True, timeout=10).stdout


def fingerprint(job, task):
    workspace = job.workspace.resolve()
    files = task['source']['files']
    revision = git(workspace, ['rev-parse', 'HEAD']).decode().strip()
    # Match the grader's byte-level Git diff contract, disabling external diff drivers.
    source_hashes = {name: file_hash(owned_file(workspace, name)) for name in files}
    diff = git(workspace, ['diff', '--no-ext-diff', '--binary', task['source']['base_revision'], '--', *files])
    return {'workspace_revision': revision, 'candidate_diff_sha256': digest(diff),
            'source_file_sha256': source_hashes}


async def execute(evaluator, job):
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-B', str(evaluator), '--workspace', str(job.workspace.resolve()),
        '--task-root', str(job.task_root.resolve()), stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, start_new_session=True)
    buffers = [bytearray(), bytearray()]
    total = 0
    overflow = asyncio.Event()

    async def pump(stream, index):
        nonlocal total
        while True:
            block = await stream.read(16384)
            if not block:
                return
            remaining = max(0, job.output_limit_bytes - total)
            buffers[index].extend(block[:remaining])
            total += len(block)
            if total > job.output_limit_bytes:
                overflow.set()

    pumps = [asyncio.create_task(pump(process.stdout, 0)),
             asyncio.create_task(pump(process.stderr, 1))]
    # Explicitly require EOF on both streams; Process.wait behavior differs by event loop.
    async def complete():
        await process.wait()
        await asyncio.gather(*pumps)

    waiter = asyncio.create_task(complete())
    limiter = asyncio.create_task(overflow.wait())
    status = None
    try:
        done, _ = await asyncio.wait([waiter, limiter], timeout=job.timeout_seconds,
                                    return_when=asyncio.FIRST_COMPLETED)
        if limiter in done:
            status = 'output_limit'
        elif waiter not in done:
            status = 'timeout'
    finally:
        # Also clean up ordinary descendants if the parent finished successfully.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        limiter.cancel()
        await asyncio.gather(limiter, return_exceptions=True)
        try:
            await asyncio.wait_for(asyncio.gather(waiter, *pumps), timeout=2)
        except asyncio.TimeoutError:
            for future in [waiter, *pumps]:
                future.cancel()
            await asyncio.gather(waiter, *pumps, return_exceptions=True)
            status = status or 'timeout'
    if overflow.is_set():
        status = 'output_limit'
    return status, process.returncode, bytes(buffers[0]), bytes(buffers[1])


def classify(data, code, task, hashes, before):
    if not isinstance(data, dict):
        return 'invalid_output'
    if 'grader_error' in data:
        return 'grader_error' if code == 1 else 'invalid_output'
    if (type(data.get('schema_version')) is not int or data['schema_version'] != 1 or
        type(data.get('passed')) is not bool or not isinstance(data.get('failures'), list) or
        not isinstance(data.get('checks'), dict)):
        return 'invalid_output'
    expected = {**before, 'task_id': task['task_id'],
                'evaluator_sha256': hashes['evaluator_sha256'],
                'task_descriptor_sha256': hashes['task_json_sha256']}
    if any(data.get(key) != value for key, value in expected.items()):
        return 'provenance_mismatch'
    if data['passed'] and code == 0 and not data['failures']:
        return 'candidate_passed'
    if not data['passed'] and code == 2 and data['failures']:
        return 'candidate_rejected'
    return 'invalid_output'


async def evaluate(job, root, queued):
    started = time.monotonic()
    record = {'schema_version': 1, 'job_id': job.id,
              'queue_seconds': started - queued, 'status': 'internal_error',
              'timeout_seconds': job.timeout_seconds, 'output_limit_bytes': job.output_limit_bytes}
    stdout = stderr = b''
    data = None
    try:
        try:
            task, evaluator, hashes = await asyncio.to_thread(protocol, job, root)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            record.update(status='protocol_error', error_type=type(exc).__name__)
            return record, stdout, stderr, data
        record.update(task_id=task['task_id'], protocol=hashes)
        try:
            before = await asyncio.to_thread(fingerprint, job, task)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            record.update(status='source_error', error_type=type(exc).__name__)
            return record, stdout, stderr, data
        record['input'] = before
        forced, code, stdout, stderr = await execute(evaluator, job)
        record['exit_code'] = code
        # Revalidate both protected assets and source after execution, even on timeout.
        try:
            _, _, after_protocol = await asyncio.to_thread(protocol, job, root)
            after = await asyncio.to_thread(fingerprint, job, task)
        except (ValueError, KeyError, OSError, TypeError, subprocess.SubprocessError):
            record['status'] = 'workspace_changed'
            return record, stdout, stderr, data
        if hashes != after_protocol or before != after:
            record['status'] = 'workspace_changed'
        elif forced:
            record['status'] = forced
        else:
            try:
                data = strict_json(stdout)
            except (ValueError, UnicodeError, RecursionError):
                record['status'] = 'invalid_output'
            else:
                record['status'] = classify(data, code, task, hashes, before)
    except asyncio.CancelledError:
        record['status'] = 'cancelled'
    except Exception as exc:
        record.update(status='internal_error', error_type=type(exc).__name__)
    finally:
        record['execution_seconds'] = time.monotonic() - started
    return record, stdout, stderr, data


def write_manifest(output):
    files = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob('*'))
             if p.is_file() and p != output / 'manifest.json'}
    atomic(output / 'manifest.json', encode({'schema_version': 1, 'files': files}))


def snapshot_key(job):
    path = owned_file(job.task_root.resolve(), 'task.json')
    file_hash(path)
    return digest(path.read_bytes())[:16]


async def run_campaign(jobs, output, workers=1, repository_root=ROOT):
    root, output = repository_root.resolve(), output.resolve()
    validate_jobs(jobs, root, output, workers)
    output.mkdir(parents=True, exist_ok=False)
    # Snapshot the runner and every protected input. No absolute local paths in artifacts.
    for name in ['__init__.py', '__main__.py', 'runner.py']:
        atomic(output / 'snapshot' / 'vare' / name, (ROOT / 'vare' / name).read_bytes())
    snapshots = set()
    for job in jobs:
        key = snapshot_key(job)
        if key not in snapshots:
            snapshots.add(key)
            for name in ['task.json', 'TASK.md', 'protocol.lock.json']:
                path = owned_file(job.task_root.resolve(), name)
                if path.is_file():
                    file_hash(path)
                    atomic(output / 'snapshot' / key / name, path.read_bytes())
            try:
                task = strict_json((job.task_root / 'task.json').read_bytes())
                path = owned_file(root, task['grader']['path'])
                file_hash(path)
                atomic(output / 'snapshot' / key / 'grade.py', path.read_bytes())
            except (ValueError, KeyError, OSError, TypeError):
                pass
    atomic(output / 'plan.json', encode({'workers': workers, 'jobs': [
        {'id': j.id, 'task_snapshot': snapshot_key(j),
         'timeout_seconds': j.timeout_seconds, 'output_limit_bytes': j.output_limit_bytes}
        for j in jobs]}))
    semaphore, commit_lock = asyncio.Semaphore(workers), asyncio.Lock()
    started = time.monotonic()
    records, ledger = [], []

    async def one(job):
        async with semaphore:
            record, stdout, stderr, data = await evaluate(job, root, started)
            async with commit_lock:
                base = output / 'jobs' / job.id
                atomic(base / 'stdout.bin', stdout)
                atomic(base / 'stderr.bin', stderr)
                if data is not None:
                    atomic(base / 'grader.json', encode(data))
                record['stdout_sha256'] = digest(stdout)
                record['stderr_sha256'] = digest(stderr)
                raw = encode(record)
                atomic(base / 'record.json', raw)
                event = {'sequence': len(ledger), 'job_id': job.id,
                         'record_sha256': digest(raw),
                         'previous_hash': ledger[-1]['event_sha256'] if ledger else '0' * 64}
                event['event_sha256'] = digest(encode(event))
                ledger.append(event)
                # Entire journal replaced atomically, so readers see only complete events.
                atomic(output / 'ledger.json', encode(ledger))
                records.append(record)

    tasks = [asyncio.create_task(one(job)) for job in jobs]
    interrupted = False
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        interrupted = True
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        summary = {'schema_version': 1, 'status': 'interrupted' if interrupted else 'complete',
                   'workers': workers, 'planned_jobs': len(jobs), 'recorded_jobs': len(records),
                   'outcomes': dict(sorted(Counter(r['status'] for r in records).items())),
                   'wall_seconds': time.monotonic() - started,
                   'runtime': {'python': platform.python_version(), 'platform': platform.platform()},
                   'resources': {'gpu_hours': 0, 'paid_api_calls': 0, 'external_compute_usd': 0}}
        if not ledger:
            atomic(output / 'ledger.json', encode([]))
        atomic(output / 'summary.json', encode(summary))
        write_manifest(output)
    return summary


def audit(output):
    output = output.resolve()
    if any(p.is_symlink() for p in output.rglob('*')):
        raise ValueError('bundle contains a symlink')
    manifest = strict_json((output / 'manifest.json').read_bytes())
    actual = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob('*'))
              if p.is_file() and p != output / 'manifest.json'}
    if manifest.get('schema_version') != 1 or actual != manifest['files']:
        raise ValueError('bundle file set or digest mismatch')
    plan = strict_json((output / 'plan.json').read_bytes())
    summary = strict_json((output / 'summary.json').read_bytes())
    ledger = strict_json((output / 'ledger.json').read_bytes())
    previous = '0' * 64
    seen, outcomes = set(), Counter()
    if any(not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', j['id']) or
           not re.fullmatch(r'[a-f0-9]{16}', j['task_snapshot']) for j in plan['jobs']):
        raise ValueError('unsafe plan path')
    planned = {j['id'] for j in plan['jobs']}
    if len(planned) != len(plan['jobs']):
        raise ValueError('duplicate plan ID')
    for index, event in enumerate(ledger):
        entry = dict(event)
        claimed = entry.pop('event_sha256')
        if digest(encode(entry)) != claimed or entry['previous_hash'] != previous or entry['sequence'] != index:
            raise ValueError('ledger chain mismatch')
        job_id = entry['job_id']
        if job_id not in planned or job_id in seen:
            raise ValueError('unexpected or duplicate ledger job')
        seen.add(job_id)
        base = output / 'jobs' / job_id
        raw = (base / 'record.json').read_bytes()
        record = strict_json(raw)
        if digest(raw) != entry['record_sha256'] or record['job_id'] != job_id or record['status'] not in STATUSES:
            raise ValueError('ledger record mismatch')
        for stream in ['stdout', 'stderr']:
            if file_hash(base / (stream + '.bin')) != record[stream + '_sha256']:
                raise ValueError('stream digest mismatch')
        if record['status'] in {'candidate_passed', 'candidate_rejected'}:
            data = strict_json((base / 'stdout.bin').read_bytes())
            snapshot = output / 'snapshot' / next(j['task_snapshot'] for j in plan['jobs'] if j['id'] == job_id)
            task = strict_json((snapshot / 'task.json').read_bytes())
            hashes = {'task_json_sha256': file_hash(snapshot / 'task.json'),
                      'task_brief_sha256': file_hash(snapshot / 'TASK.md'),
                      'evaluator_sha256': file_hash(snapshot / 'grade.py')}
            lock = strict_json((snapshot / 'protocol.lock.json').read_bytes())
            if lock['locked_hashes'] != hashes or lock['task_id'] != task['task_id']:
                raise ValueError('snapshot violates protocol lock')
            hashes['protocol_lock_sha256'] = file_hash(snapshot / 'protocol.lock.json')
            if record['protocol'] != hashes or set(record['input']['source_file_sha256']) != set(task['source']['files']):
                raise ValueError('record disagrees with protected snapshot')
            if strict_json((base / 'grader.json').read_bytes()) != data:
                raise ValueError('parsed grader result disagrees with raw stream')
            if classify(data, record['exit_code'], task, record['protocol'], record['input']) != record['status']:
                raise ValueError('accepted record violates grader contract')
        outcomes[record['status']] += 1
        previous = claimed
    if summary['recorded_jobs'] != len(seen) or summary['planned_jobs'] != len(plan['jobs']) or summary['outcomes'] != dict(outcomes):
        raise ValueError('summary disagrees with ledger')
    if summary['status'] == 'complete' and seen != planned:
        raise ValueError('complete campaign has missing jobs')
    if summary['status'] not in {'complete', 'interrupted'} or summary['workers'] != plan['workers']:
        raise ValueError('invalid campaign summary')
    return {'status': 'verified', 'recorded_jobs': len(seen), 'outcomes': dict(outcomes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--workers', type=int, default=1)
    verify = sub.add_parser('audit')
    verify.add_argument('bundle', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'audit':
            result = audit(args.bundle)
        else:
            path = args.plan.resolve()
            plan = strict_json(path.read_bytes())
            jobs = [Job(j['id'], (path.parent / j['task_root']).resolve(),
                        (path.parent / j['workspace']).resolve(),
                        j.get('timeout_seconds', 30), j.get('output_limit_bytes', 1048576))
                    for j in plan['jobs']]
            result = asyncio.run(run_campaign(jobs, args.output, args.workers))
        print(encode(result).decode(), end='')
        if args.command == 'audit':
            return 0
        return 0 if result.get('status') == 'complete' and not any(
            k not in {'candidate_passed', 'candidate_rejected'} for k in result.get('outcomes', {})) else 1
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(json.dumps({'error': type(exc).__name__, 'message': str(exc)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
