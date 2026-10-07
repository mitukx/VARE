#!/usr/bin/env python3
"""Reconstruct the scheduler report from retained campaign records; no source fetch."""
import argparse
from collections import Counter
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vare.runner import audit, encode, file_hash, strict_json


def load(path):
    return strict_json(path.read_bytes())


def q(values, p):
    return sorted(values)[math.ceil(len(values) * p) - 1]


def audit_measurement(output):
    output = output.resolve()
    if any(p.is_symlink() for p in output.rglob('*')):
        raise ValueError('linked artifact')
    manifest = load(output / 'manifest.json')
    actual = {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*')
              if p.is_file() and p != output / 'manifest.json'}
    if manifest['files'] != actual:
        raise ValueError('measurement manifest mismatch')
    protocol_path = output / 'snapshot/protocols/cpu_scheduler_v1.json'
    protocol = load(protocol_path)
    summary = load(output / 'summary.json')
    if summary['protocol_sha256'] != file_hash(protocol_path) or summary['status'] != 'complete':
        raise ValueError('protocol or completion mismatch')
    if protocol['paired_repetitions'] != len(protocol['order']):
        raise ValueError('protocol repetition mismatch')
    audit(output / 'reliability')
    faults = load(output / 'reliability-summary.json')
    observed = {name: load(output / 'reliability/jobs' / name / 'record.json')['status']
                for name in protocol['reliability_cases']}
    if faults['expected'] != protocol['reliability_cases'] or faults['observed'] != observed or observed != faults['expected']:
        raise ValueError('fault classifications mismatch')
    if not faults['matched'] or faults['case_count'] != len(observed) or not all(
            r['not_running'] for r in faults['child_cleanup'].values()):
        raise ValueError('fault summary mismatch')
    paired, total = [], 0
    for pair, order in enumerate(protocol['order']):
        row = {'pair': pair, 'order': order, 'campaigns': {}}
        for workers in order:
            bundle = output / ('pair-' + str(pair)) / ('workers-' + str(workers))
            checked = audit(bundle)
            campaign = load(bundle / 'summary.json')
            plan = load(bundle / 'plan.json')
            if campaign['status'] != 'complete' or campaign['workers'] != workers or len(plan['jobs']) != 8:
                raise ValueError('campaign count or configuration mismatch')
            records = []
            for index, task_id in enumerate(protocol['task_pack']):
                for label in ['baseline', 'fixed']:
                    for copy in range(2):
                        job_id = 'task-%d-%s-%d' % (index, label, copy)
                        record = load(bundle / 'jobs' / job_id / 'record.json')
                        job = next(j for j in plan['jobs'] if j['id'] == job_id)
                        task = load(bundle / 'snapshot' / job['task_snapshot'] / 'task.json')
                        revision = task['source']['base_revision' if label == 'baseline' else 'calibration_revision']
                        expected = 'candidate_rejected' if label == 'baseline' else 'candidate_passed'
                        if record['task_id'] != task_id or record['status'] != expected or record['input']['workspace_revision'] != revision:
                            raise ValueError('historical outcome or revision mismatch')
                        if record['timeout_seconds'] != protocol['timeout_seconds'] or record['output_limit_bytes'] != protocol['output_limit_bytes']:
                            raise ValueError('historical budget mismatch')
                        records.append(record)
            queue, execution = ([r[key] for r in records] for key in ['queue_seconds', 'execution_seconds'])
            row['campaigns'][str(workers)] = {'wall_seconds': campaign['wall_seconds'],
                'jobs_per_second': len(records) / campaign['wall_seconds'],
                'queue_p50_seconds': q(queue, .5), 'queue_p95_seconds': q(queue, .95),
                'execution_p50_seconds': q(execution, .5), 'execution_p95_seconds': q(execution, .95),
                'outcomes': checked['outcomes']}
            total += len(records)
        row['speedup'] = row['campaigns']['1']['wall_seconds'] / row['campaigns']['4']['wall_seconds']
        if row != load(output / ('pair-' + str(pair)) / 'comparison.json'):
            raise ValueError('pair summary does not reconstruct')
        paired.append(row)
    median = statistics.median(row['speedup'] for row in paired)
    if (paired != summary['paired'] or median != summary['median_paired_speedup'] or
        total != summary['total_historical_evaluations'] or not summary['all_expected_outcomes'] or
        summary['reliability_cases_matched'] != len(observed) or summary['resources'] != protocol['budget']):
        raise ValueError('headline summary does not reconstruct')
    return {'status': 'verified', 'historical_evaluations': total,
            'reliability_cases': len(observed), 'median_paired_speedup': median,
            'claim_limit': protocol['claim_limit']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    args = parser.parse_args()
    try:
        print(encode(audit_measurement(args.bundle)).decode(), end='')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
