"""Replay retained process-loss evidence and costs; does not launch graders."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from vare.durable import audit_durable
from vare.runner import audit

ROOT = Path(__file__).resolve().parents[1]


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    # Keep JSON parsing independent of the runner's parser, including ambiguity.
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('nonfinite JSON value: ' + value)

    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def verify(out):
    if out.is_file():
        require(not out.is_symlink(), 'linked archive')
        with tempfile.TemporaryDirectory(prefix='vare-owner-audit-') as temp:
            with zipfile.ZipFile(out) as archive:
                entries = archive.infolist()
                require(len(entries) <= 2000 and len({e.filename for e in entries}) == len(entries),
                        'oversized or duplicated archive inventory')
                require(sum(e.file_size for e in entries) <= 32 * 1024**2, 'oversized archive')
                for entry in entries:
                    path = Path(entry.filename)
                    require(not path.is_absolute() and '..' not in path.parts and
                            '\\' not in entry.filename and
                            (entry.external_attr >> 16) & 0o170000 != 0o120000, 'unsafe archive path')
                archive.extractall(temp)
            return verify_directory(Path(temp))
    return verify_directory(out)


def verify_directory(out):
    require(not out.is_symlink() and not any(p.is_symlink() for p in out.rglob('*')), 'linked evidence')
    manifest = read(out / 'manifest.json')
    require(manifest['schema_version'] == 1, 'manifest version')
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p != out / 'manifest.json'}
    require(files == manifest['files'], 'inventory mismatch')
    protocol = read(ROOT / 'protocols/grader_owner_loss_v1.lock.json')
    require(read(out / 'protocol.json') == protocol, 'frozen protocol mismatch')
    require(sha(out / 'protected-test.py.txt') == protocol['protected_test_sha256'], 'test changed')
    require(manifest['baseline_commit'] == protocol['baseline_commit'], 'baseline revision mismatch')
    for name in ['runner.py', 'durable.py']:
        raw = subprocess.check_output(['git', 'show', protocol['baseline_commit'] + ':src/vare/' + name], cwd=ROOT)
        require(raw == (out / 'baseline/source' / (name + '.txt')).read_bytes(), 'baseline source mismatch')
    summary = {}
    for label in ['baseline', 'candidate']:
        candidate = label == 'candidate'
        cases = ET.parse(out / label / 'tests.xml').findall('.//testcase')
        require({c.attrib['name'].split('[')[-1].rstrip(']') for c in cases} == set(protocol['faults'])
                and len(cases) == 3, 'fault test set mismatch')
        require(not any(c.find('error') is not None or c.find('skipped') is not None for c in cases), 'test error/skip')
        require(sum(c.find('failure') is not None for c in cases) == (0 if candidate else 3), 'fault verdict mismatch')
        require(read(out / label / 'exit.json') == {'exit_code': 0 if candidate else 1}, 'pytest exit mismatch')
        for fault in protocol['faults']:
            folder = out / label / 'faults' / fault
            observed = read(folder / 'observation.json')
            require(observed['fault'] == fault and observed['stopped'] is candidate, 'owner-loss result mismatch')
            require(observed['owner_exit'] == {'SIGKILL': -9, 'SIGTERM': -15, 'exit': 73}[fault], 'fault not injected')
            receipt = observed['receipt']
            require(all(type(receipt[key]) is int and receipt[key] > 0
                        for key in ['grader', 'child', 'group', 'parent']), 'invalid PID receipt')
            require(receipt['grader'] != receipt['child'], 'not a real descendant')
            require((receipt['group'] == receipt['parent'] and receipt['group'] != receipt['grader'])
                    if candidate else receipt['group'] == receipt['grader'], 'grader group ownership mismatch')
            require(number(observed['stop_seconds']), 'invalid stop measurement')
            if candidate:
                require(observed['stop_seconds'] < protocol['stop_timeout_seconds'], 'stop deadline exceeded')
                replay = audit_durable(folder / 'export')
                require(replay == observed['replay'] and replay['attempts'] == 2
                        and replay['outcomes'] == {'candidate_passed': 1}, 'recovery replay mismatch')
                state = read(folder / 'export/state.json')
                require(state['jobs'][0]['token'] == 2 and [e['kind'] for e in state['events']] ==
                        ['enrolled', 'claimed', 'expired', 'claimed', 'committed'], 'recovery transitions mismatch')
                for name in ['runner.py', 'durable.py', '_grader_guard.py']:
                    require((folder / 'export/snapshot/vare' / name).read_bytes() ==
                            (out / label / 'source' / (name + '.txt')).read_bytes(), 'recovery executor changed')
            else:
                require(observed['stop_seconds'] >= protocol['stop_timeout_seconds'], 'baseline did not wait')
        rows = []
        for pair in range(protocol['timing_pairs']):
            directory = out / label / ('pair-%d' % pair)
            process = read(out / label / ('pair-%d-process.json' % pair))
            order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
            require(process['exit_code'] == 0 and process['order'] == order, 'cost execution mismatch')
            row = read(directory / 'timing.json')
            require(all(number(v) for k, v in row.items() if k != 'replay'), 'invalid cost')
            replay = audit(directory / 'campaign')
            require(row['replay'] == replay and replay['outcomes'] == {'candidate_passed': protocol['timing_jobs']},
                    'cost workload mismatch')
            plan = read(directory / 'campaign/plan.json')
            require(plan['workers'] == protocol['timing_workers'] and
                    {j['id'] for j in plan['jobs']} == {'cost-%02d' % i for i in range(protocol['timing_jobs'])},
                    'cost plan mismatch')
            records = [read(directory / 'campaign/jobs' / j['id'] / 'record.json') for j in plan['jobs']]
            latencies = [r['execution_seconds'] for r in records]
            require(row['latency_p50_seconds'] == statistics.median(latencies)
                    and row['latency_p95_seconds'] == statistics.quantiles(latencies, n=20, method='inclusive')[18],
                    'latency summary mismatch')
            require(row['jobs_per_second'] == protocol['timing_jobs'] / row['wall_seconds'], 'throughput mismatch')
            for name in ['runner.py'] + (['_grader_guard.py'] if candidate else []):
                require((directory / 'campaign/snapshot/vare' / name).read_bytes() ==
                        (out / label / 'source' / (name + '.txt')).read_bytes(), 'cost executor changed')
            for task in (directory / 'campaign/snapshot').glob('*/task.json'):
                descriptor = read(task)
                require(descriptor['fixture_behavior'] == 'valid' and
                        descriptor['fixture_delay'] == protocol['grader_delay_seconds'], 'cost grader mismatch')
            rows.append(row)
        summary[label] = {key: statistics.median(row[key] for row in rows)
                          for key in rows[0] if key != 'replay'}
    require(summary == read(out / 'summary.json'), 'cost medians mismatch')
    return {'status': 'verified', 'faults': 3, 'recovered_jobs': 3, 'cost_pairs': protocol['timing_pairs'],
            'scope': 'same-host process correctness and descriptive costs; no model learning', 'summary': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    print(json.dumps(verify(parser.parse_args().bundle.absolute()), indent=2))
