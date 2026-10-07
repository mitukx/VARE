#!/usr/bin/env python3
"""Reconstruct frozen recovery acceptance from exported states and events."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vare.durable import audit_durable
from vare.runner import encode, file_hash, strict_json


def load(path):
    return strict_json(path.read_bytes())


def audit_recovery(output):
    output = output.resolve()
    if any(p.is_symlink() for p in output.rglob('*')):
        raise ValueError('linked evidence')
    manifest = load(output / 'manifest.json')
    actual = {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*') if p.is_file() and p != output / 'manifest.json'}
    if manifest['files'] != actual:
        raise ValueError('experiment manifest mismatch')
    protocol_path = output / 'snapshot/protocols/cpu_recovery_v1.json'
    protocol = load(protocol_path)
    summary = load(output / 'summary.json')
    if summary['status'] != 'complete' or summary['protocol_sha256'] != file_hash(protocol_path):
        raise ValueError('completion or protocol mismatch')
    reconstructed = []
    total_attempts, total_jobs = 0, 0
    for replication in range(protocol['replications']):
        for name in protocol['cases']:
            base = output / ('replicate-' + str(replication))
            bundle = base / name
            checked = audit_durable(bundle)
            state = load(bundle / 'state.json')
            rows = state['jobs']
            events = state['events']
            row = rows[0]
            saved = load(base / (name + '.summary.json'))
            notes = saved['notes']
            if name == 'claim_race':
                matched = checked['outcomes'] == {'candidate_passed': protocol['race_jobs']} and checked['attempts'] == protocol['race_jobs'] and all(j['token'] == 1 for j in rows)
                matched = matched and len({e['details']['owner'] for e in events if e['kind'] == 'claimed'}) <= protocol['workers']
            elif name == 'death_after_claim':
                matched = checked['outcomes'] == {'candidate_passed': 1} and row['token'] == 2 and checked['attempts'] == 2 and notes['fault_exit'] == 23
                matched = matched and any(e['kind'] == 'expired' and e['token'] == 1 for e in events)
            elif name in {'transaction_rollback', 'acknowledgement_loss'}:
                matched = checked['outcomes'] == {'candidate_passed': 1} and row['token'] == 1 and checked['attempts'] == 1
                matched = matched and notes['fault_exit'] == (24 if name == 'transaction_rollback' else 25)
                if name == 'transaction_rollback':
                    matched = matched and notes['state_after_death'] == 'pending'
            elif name == 'late_completion':
                first = load(bundle / 'attempts/valid/1/attempt.json')
                second = load(bundle / 'attempts/valid/2/attempt.json')
                matched = checked['outcomes'] == {'candidate_passed': 1} and row['token'] == 2 and checked['attempts'] == 2
                matched = matched and first['state'] == 'fenced' and second['state'] == 'committed' and notes['new_committed'] and not notes['old_committed']
            elif name == 'stale_reuse':
                record = load(bundle / 'attempts/valid/1/record.json')
                matched = checked['outcomes'] == {'input_stale': 1} and record['status'] == 'candidate_passed' and checked['attempts'] == 1
                matched = matched and notes['retained_historical_status'] == 'candidate_passed'
            elif name == 'stale_pending':
                matched = checked['outcomes'] == {'input_stale': 1} and checked['attempts'] == 0
            elif name == 'retry_budget':
                matched = checked['outcomes'] == {'retry_exhausted': 1} and row['token'] == protocol['max_attempts'] and not notes['extra_claim']
            else:
                raise ValueError('unknown frozen case')
            expected = {'case': name, 'matched': matched, 'audit': checked, 'notes': notes, 'replication': replication}
            if not matched or saved != expected:
                raise ValueError('case criterion or summary mismatch: ' + name)
            reconstructed.append(expected)
            total_attempts += checked['attempts']
            total_jobs += checked['jobs']
    integration = load(output / 'historical-summary.json')
    checked = audit_durable(output / 'historical')
    state = load(output / 'historical/state.json')
    observed, expected = {}, {}
    task_ids = ['rvl-hf-behavior-policy-parity-v1', 'trl-grpo-accumulation-window-normalizer-v1']
    for index, task_id in enumerate(task_ids):
        for label in ['baseline', 'fixed']:
            name = 'task-%d-%s' % (index, label)
            row = next(j for j in state['jobs'] if j['id'] == name)
            task = load(output / 'historical/snapshot' / row['snapshot'] / 'task.json')
            revision = task['source']['base_revision' if label == 'baseline' else 'calibration_revision']
            if row['expected']['task_id'] != task_id or row['expected']['input']['workspace_revision'] != revision:
                raise ValueError('historical enrollment mismatch')
            observed[name] = row['effective']
            expected[name] = 'candidate_rejected' if label == 'baseline' else 'candidate_passed'
    matched = observed == expected and sum(j['token'] == 2 for j in state['jobs']) == 1 and checked['attempts'] == 5
    rebuilt = {'matched': matched, 'expected': expected, 'observed': observed, 'audit': checked}
    if not matched or rebuilt != integration or rebuilt != summary['historical_integration']:
        raise ValueError('historical recovery criterion mismatch')
    if summary['cases'] != reconstructed or summary['matched_cases'] != len(reconstructed) or summary['resources'] != protocol['resources']:
        raise ValueError('headline does not reconstruct')
    return {'status': 'verified', 'matched_fault_cases': len(reconstructed),
            'synthetic_jobs': total_jobs, 'synthetic_attempts': total_attempts,
            'historical_jobs': checked['jobs'], 'historical_attempts': checked['attempts'],
            'claim_limit': protocol['claim_limit']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    args = p.parse_args()
    try:
        print(encode(audit_recovery(args.bundle)).decode(), end='')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
