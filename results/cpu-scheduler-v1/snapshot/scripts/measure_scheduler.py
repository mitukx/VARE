#!/usr/bin/env python3
"""Reproduce the frozen CPU scheduler protocol into a new evidence directory."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.scheduler.fixtures import make_fixture
from scripts.calibrate_task import checkout_revision
from vare.runner import Job, atomic, audit, encode, file_hash, run_campaign, strict_json, write_manifest

PROTOCOL = ROOT / 'protocols/cpu_scheduler_v1.json'
TASKS = [ROOT / 'benchmarks/historical/rvl_behavior_policy_parity',
         ROOT / 'benchmarks/historical/trl_grpo_accumulation_scale']


def quantile(values, p):
    return sorted(values)[math.ceil(p * len(values)) - 1]


def reliability(output, protocol, scratch):
    jobs, expected = [], protocol['reliability_cases']
    for behavior in expected:
        fixture_behavior = 'valid' if behavior in {'missing_source', 'tampered_protocol'} else behavior
        # Separate directories allow distinct instances of the same fixture behavior.
        parent = scratch / behavior
        parent.mkdir()
        root, job = make_fixture(parent, fixture_behavior,
                                 timeout=.5 if behavior in {'timeout', 'inherited_pipe'} else 3,
                                 limit=1024 if behavior == 'flood' else 1048576)
        # Unify trusted roots before launching the independent candidate workspaces.
        unified = scratch / 'trusted'
        destination = unified / behavior
        destination.mkdir(parents=True)
        for source in job.task_root.iterdir():
            (destination / source.name).write_bytes(source.read_bytes())
        task_path = destination / 'task.json'
        task = strict_json(task_path.read_bytes())
        task['grader']['path'] = behavior + '/grade.py'
        atomic_rewrite(task_path, encode(task))
        lock_path = destination / 'protocol.lock.json'
        lock = strict_json(lock_path.read_bytes())
        lock['locked_hashes']['task_json_sha256'] = file_hash(task_path)
        atomic_rewrite(lock_path, encode(lock))
        if behavior == 'missing_source':
            (job.workspace / 'source.txt').unlink()
        if behavior == 'tampered_protocol':
            (destination / 'TASK.md').write_text('injected protected-input mutation\n')
        jobs.append(Job(behavior, destination, job.workspace, job.timeout_seconds, job.output_limit_bytes))
    summary = asyncio.run(run_campaign(jobs, output, 4, unified))
    audit(output)
    observed = {job.id: strict_json((output / 'jobs' / job.id / 'record.json').read_bytes())['status'] for job in jobs}
    # Verify child processes are dead or zombies (awaiting OS reaping), not running.
    cleanup = {}
    for name in ['child', 'inherited_pipe']:
        stderr = (output / 'jobs' / name / 'stderr.bin').read_text()
        pid = int(stderr.split('child_pid=')[1].split()[0])
        state = ''
        for _ in range(50):
            state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
            if not state or state.startswith('Z'):
                break
            time.sleep(.02)
        cleanup[name] = {'not_running': not state or state.startswith('Z'), 'observed_state': state or 'absent'}
    result = {'expected': expected, 'observed': observed,
              'matched': observed == expected, 'case_count': len(expected), 'child_cleanup': cleanup}
    if not result['matched'] or not all(row['not_running'] for row in cleanup.values()):
        raise ValueError('reliability protocol mismatch: ' + json.dumps(result))
    return result


def atomic_rewrite(path, data):
    # Fixture configuration before evaluation, never edits a retained evidence record.
    path.unlink()
    atomic(path, data)


def measure(output):
    protocol = strict_json(PROTOCOL.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'snapshot'
    for relative in ['protocols/cpu_scheduler_v1.json', 'scripts/measure_scheduler.py',
                     'scripts/calibrate_task.py', 'benchmarks/scheduler/fixtures.py',
                     'benchmarks/scheduler/fixture_grader.py', 'tests/test_runner.py']:
        atomic(snapshot / relative, (ROOT / relative).read_bytes())
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='vare-scheduler-') as directory:
            scratch = Path(directory)
            fault_root = scratch / 'faults'
            fault_root.mkdir()
            faults = reliability(output / 'reliability', protocol, fault_root)
            atomic(output / 'reliability-summary.json', encode(faults))
            print('Reliability cases matched: ' + str(faults['case_count']), flush=True)
            jobs, expected = [], {}
            for index, task_root in enumerate(TASKS):
                task = strict_json((task_root / 'task.json').read_bytes())
                for label, revision in [('baseline', task['source']['base_revision']),
                                        ('fixed', task['source']['calibration_revision'])]:
                    workspace = scratch / ('task-' + str(index) + '-' + label)
                    checkout_revision(task['source']['repository'], revision,
                                      task['source']['base_revision'], workspace, task['source']['files'])
                    for copy in range(2):
                        job_id = 'task-' + str(index) + '-' + label + '-' + str(copy)
                        jobs.append(Job(job_id, task_root, workspace,
                                        protocol['timeout_seconds'], protocol['output_limit_bytes']))
                        expected[job_id] = 'candidate_passed' if label == 'fixed' else 'candidate_rejected'
            paired = []
            for repetition, order in enumerate(protocol['order']):
                row = {'pair': repetition, 'order': order, 'campaigns': {}}
                for workers in order:
                    path = output / ('pair-' + str(repetition)) / ('workers-' + str(workers))
                    summary = asyncio.run(run_campaign(jobs, path, workers))
                    verified = audit(path)
                    records = [strict_json((path / 'jobs' / j.id / 'record.json').read_bytes()) for j in jobs]
                    observed = {r['job_id']: r['status'] for r in records}
                    if observed != expected:
                        raise ValueError('unexpected historical task result: ' + json.dumps(observed))
                    queue = [r['queue_seconds'] for r in records]
                    execution = [r['execution_seconds'] for r in records]
                    row['campaigns'][str(workers)] = {'wall_seconds': summary['wall_seconds'],
                        'jobs_per_second': len(jobs) / summary['wall_seconds'],
                        'queue_p50_seconds': quantile(queue, .5), 'queue_p95_seconds': quantile(queue, .95),
                        'execution_p50_seconds': quantile(execution, .5), 'execution_p95_seconds': quantile(execution, .95),
                        'outcomes': verified['outcomes']}
                    print('Pair %d workers %d: %.4fs' % (repetition, workers, summary['wall_seconds']), flush=True)
                    if time.monotonic() - started > protocol['budget']['max_local_wall_seconds']:
                        raise ValueError('local measurement wall budget exceeded')
                row['speedup'] = row['campaigns']['1']['wall_seconds'] / row['campaigns']['4']['wall_seconds']
                paired.append(row)
                atomic(output / ('pair-' + str(repetition)) / 'comparison.json', encode(row))
            summary = {'schema_version': 1, 'protocol_id': protocol['protocol_id'], 'status': 'complete',
                       'protocol_sha256': file_hash(PROTOCOL), 'paired': paired,
                       'median_paired_speedup': statistics.median(r['speedup'] for r in paired),
                       'total_historical_evaluations': 80, 'all_expected_outcomes': True,
                       'reliability_cases_matched': faults['case_count'],
                       'runtime': {'python': platform.python_version(), 'platform': platform.platform()},
                       'resources': protocol['budget'], 'claim_limit': protocol['claim_limit']}
            atomic(output / 'summary.json', encode(summary))
            print(encode(summary).decode(), end='')
    except BaseException as exc:
        atomic(output / 'failure.json', encode({'status': 'failed', 'error_type': type(exc).__name__,
                                              'message': str(exc), 'protocol_sha256': file_hash(PROTOCOL)}))
        raise
    finally:
        write_manifest(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents and not output.is_relative_to(ROOT / 'results') or ROOT.is_relative_to(output):
        parser.error('output must be new, outside protected inputs, or under results/')
    measure(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
