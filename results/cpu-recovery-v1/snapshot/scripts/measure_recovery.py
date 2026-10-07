#!/usr/bin/env python3
"""Frozen local process-recovery protocol; no model/API/accelerator use."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.scheduler.fixtures import make_fixture
from scripts.calibrate_task import checkout_revision
from vare.durable import Store, audit_durable, run_workers
from vare.runner import Job, atomic, encode, evaluate, file_hash, strict_json, write_manifest

PROTOCOL = ROOT / 'protocols/cpu_recovery_v1.json'


def die_claim(path, lease):
    Store(path).claim('lost-worker', lease)
    os._exit(23)


def die_transaction(path):
    with Store(path).connection(True) as db:
        db.execute("UPDATE jobs SET state='stale',effective='input_stale'")
        os._exit(24)


def die_ack(path, root, lease):
    store = Store(path)
    claim = store.claim('unacknowledged', lease)
    result = asyncio.run(evaluate(claim['job'], Path(root), time.monotonic()))
    if not store.complete(claim, result)['committed']:
        os._exit(26)
    os._exit(25)


def crash(target, args, expected):
    child = multiprocessing.get_context('spawn').Process(target=target, args=args)
    child.start()
    child.join(30)
    if child.is_alive():
        child.kill()
        child.join()
        raise ValueError('fault process hung')
    if child.exitcode != expected:
        raise ValueError('unexpected fault exit: ' + str(child.exitcode))
    return child.exitcode


def grade(claim, root):
    return asyncio.run(evaluate(claim['job'], root, time.monotonic()))


def run_case(name, output, parent, protocol):
    parent.mkdir()
    root, job = make_fixture(parent, 'valid')
    jobs = [replace(job, id='job-%02d' % i) for i in range(protocol['race_jobs'])] if name == 'claim_race' else [job]
    store = Store.create(parent / 'state.db', jobs, root, protocol['max_attempts'])
    lease, short = protocol['lease_seconds'], protocol['fault_lease_seconds']
    notes = {}
    if name == 'claim_race':
        run_workers(store.path, protocol['workers'], lease)
    elif name == 'death_after_claim':
        notes['fault_exit'] = crash(die_claim, (store.path, short), 23)
        time.sleep(short + .05)
        run_workers(store.path, protocol['workers'], lease)
    elif name == 'transaction_rollback':
        notes['fault_exit'] = crash(die_transaction, (store.path,), 24)
        notes['state_after_death'] = store.jobs()[0]['state']
        run_workers(store.path, protocol['workers'], lease)
    elif name == 'acknowledgement_loss':
        notes['fault_exit'] = crash(die_ack, (store.path, root, lease), 25)
        run_workers(store.path, protocol['workers'], lease)
    elif name == 'late_completion':
        old = store.claim('old-worker', short)
        old_result = grade(old, root)
        time.sleep(short + .05)
        new = store.claim('new-worker', lease)
        notes['new_committed'] = store.complete(new, grade(new, root))['committed']
        notes['old_committed'] = store.complete(old, old_result)['committed']
    elif name == 'stale_reuse':
        run_workers(store.path, protocol['workers'], lease)
        (job.workspace / 'source.txt').write_text('changed after terminal result\n')
    elif name == 'stale_pending':
        (job.workspace / 'source.txt').write_text('changed before grading\n')
        run_workers(store.path, protocol['workers'], lease)
    elif name == 'retry_budget':
        for _ in range(protocol['max_attempts']):
            if store.claim('lost-worker', short) is None:
                raise ValueError('budget exhausted early')
            time.sleep(short + .05)
        notes['extra_claim'] = store.claim('extra-worker', lease) is not None
    else:
        raise ValueError('unknown frozen case')
    store.export(output)
    audited = audit_durable(output)
    state = strict_json((output / 'state.json').read_bytes())
    rows = state['jobs']
    if name == 'claim_race':
        passed = audited['outcomes'] == {'candidate_passed': protocol['race_jobs']} and all(r['token'] == 1 for r in rows)
        passed = passed and audited['attempts'] == protocol['race_jobs']
    elif name in {'death_after_claim', 'late_completion'}:
        passed = audited['outcomes'] == {'candidate_passed': 1} and rows[0]['token'] == 2 and audited['attempts'] == 2
        if name == 'late_completion':
            passed = passed and notes['new_committed'] and not notes['old_committed']
    elif name in {'transaction_rollback', 'acknowledgement_loss'}:
        passed = audited['outcomes'] == {'candidate_passed': 1} and rows[0]['token'] == 1 and audited['attempts'] == 1
        if name == 'transaction_rollback':
            passed = passed and notes['state_after_death'] == 'pending'
    elif name == 'stale_reuse':
        record = strict_json((output / 'attempts/valid/1/record.json').read_bytes())
        notes['retained_historical_status'] = record['status']
        passed = audited['outcomes'] == {'input_stale': 1} and record['status'] == 'candidate_passed' and audited['attempts'] == 1
    elif name == 'stale_pending':
        passed = audited['outcomes'] == {'input_stale': 1} and audited['attempts'] == 0
    else:
        passed = audited['outcomes'] == {'retry_exhausted': 1} and rows[0]['token'] == protocol['max_attempts'] and not notes['extra_claim']
    return {'case': name, 'matched': passed, 'audit': audited, 'notes': notes}


def historical(output, scratch, protocol):
    jobs, expected = [], {}
    for index, relative in enumerate(['rvl_behavior_policy_parity', 'trl_grpo_accumulation_scale']):
        task_root = ROOT / 'benchmarks/historical' / relative
        task = strict_json((task_root / 'task.json').read_bytes())
        for label, revision in [('baseline', task['source']['base_revision']), ('fixed', task['source']['calibration_revision'])]:
            workspace = scratch / ('task-%d-%s' % (index, label))
            checkout_revision(task['source']['repository'], revision, task['source']['base_revision'], workspace, task['source']['files'])
            job_id = 'task-%d-%s' % (index, label)
            jobs.append(Job(job_id, task_root, workspace))
            expected[job_id] = 'candidate_rejected' if label == 'baseline' else 'candidate_passed'
    store = Store.create(scratch / 'historical.db', jobs, ROOT)
    crash(die_claim, (store.path, protocol['fault_lease_seconds']), 23)
    time.sleep(protocol['fault_lease_seconds'] + .05)
    run_workers(store.path, protocol['workers'], protocol['lease_seconds'])
    store.export(output)
    audited = audit_durable(output)
    state = strict_json((output / 'state.json').read_bytes())
    observed = {j['id']: j['effective'] for j in state['jobs']}
    matched = observed == expected and sum(j['token'] == 2 for j in state['jobs']) == 1 and audited['attempts'] == 5
    return {'matched': matched, 'expected': expected, 'observed': observed, 'audit': audited}


def measure(output):
    protocol = strict_json(PROTOCOL.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    for relative in ['protocols/cpu_recovery_v1.json', 'scripts/measure_recovery.py',
                     'benchmarks/scheduler/fixtures.py', 'benchmarks/scheduler/fixture_grader.py', 'tests/test_durable.py']:
        atomic(output / 'snapshot' / relative, (ROOT / relative).read_bytes())
    started = time.monotonic()
    try:
        results = []
        with tempfile.TemporaryDirectory(prefix='vare-recovery-') as directory:
            scratch = Path(directory)
            for repetition in range(protocol['replications']):
                for name in protocol['cases']:
                    case_output = output / ('replicate-' + str(repetition)) / name
                    result = run_case(name, case_output, scratch / ('replicate-%d-%s' % (repetition, name)), protocol)
                    result['replication'] = repetition
                    results.append(result)
                    atomic(output / ('replicate-' + str(repetition)) / (name + '.summary.json'), encode(result))
                    print('Replication %d %s: %s' % (repetition, name, result['matched']), flush=True)
                    if not result['matched'] or time.monotonic() - started > protocol['resources']['max_wall_seconds']:
                        raise ValueError('frozen recovery criterion or budget failed')
            integration = historical(output / 'historical', scratch, protocol)
            atomic(output / 'historical-summary.json', encode(integration))
            if not integration['matched']:
                raise ValueError('historical recovery mismatch')
        summary = {'schema_version': 1, 'status': 'complete', 'protocol_id': protocol['protocol_id'],
                   'protocol_sha256': file_hash(PROTOCOL), 'cases': results, 'matched_cases': len(results),
                   'historical_integration': integration, 'wall_seconds': time.monotonic() - started,
                   'runtime': {'python': platform.python_version(), 'platform': platform.platform(), 'sqlite': __import__('sqlite3').sqlite_version},
                   'resources': protocol['resources'], 'claim_limit': protocol['claim_limit']}
        atomic(output / 'summary.json', encode(summary))
        print('Recovery matched cases: ' + str(summary['matched_cases']), flush=True)
    except BaseException as exc:
        atomic(output / 'failure.json', encode({'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)}))
        raise
    finally:
        write_manifest(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.output.resolve()
    if path == ROOT or ROOT.is_relative_to(path) or ROOT in path.parents and not path.is_relative_to(ROOT / 'results'):
        parser.error('output must be new, outside inputs, or under results/')
    measure(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
