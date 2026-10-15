"""Offline replay of source-pin refusals, restored execution and raw costs."""
import argparse
import json
from pathlib import Path
import statistics
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from audit_grader_owner_loss_v1 import number, require, sha
from vare.durable import audit_durable
from vare.runner import strict_json

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SOURCE_SHA256 = {
    '__init__.py': '548fea93dbd5c551b72d1649f3a41d2109428d7cb24b1ff524345e18cf262025',
    '__main__.py': 'db44dd5bcafe53bb1e9e20e9b816d441a8d91a24d07ea48aac6f79efd47c9dcd',
    'runner.py': '5aaaf6ce099dbc262c7c622336d77828b90fa73af9e436664f24c391ad7aa391',
    'durable.py': '3e521487fa0c288908e69940c8ce83f7c53fae7bf07773fd0c9ead66dc6efa9d',
    '_grader_guard.py': 'b64f104d62bcd59d7c50e4b8abc5530f4db526b4071a79e082c42fac4c7f2d7c',
}
ASSETS = ['__init__.py', '__main__.py', 'runner.py', 'durable.py', '_grader_guard.py']


def read(path):
    return strict_json(path.read_bytes())


def same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def verify(out):
    require(not out.is_symlink(), 'linked evidence')
    if out.is_file():
        with tempfile.TemporaryDirectory(prefix='vare-runtime-audit-') as temp:
            with zipfile.ZipFile(out) as archive:
                entries = archive.infolist()
                require(len(entries) <= 4096 and len({e.filename for e in entries}) == len(entries)
                        and sum(e.file_size for e in entries) <= 32 * 1024**2, 'invalid archive inventory')
                for entry in entries:
                    path = Path(entry.filename)
                    require(not path.is_absolute() and '..' not in path.parts and '\\' not in entry.filename
                            and (entry.external_attr >> 16) & 0o170000 != 0o120000, 'unsafe archive member')
                archive.extractall(temp)
            return verify_directory(Path(temp))
    return verify_directory(out)


def verify_directory(out):
    require(not any(p.is_symlink() for p in out.rglob('*')), 'linked evidence member')
    manifest = read(out / 'manifest.json')
    actual = {str(p.relative_to(out)): sha(p) for p in out.rglob('*')
              if p.is_file() and p != out / 'manifest.json'}
    require(type(manifest['schema_version']) is int and manifest['schema_version'] == 1 and
            actual == manifest['files'], 'inventory mismatch')
    protocol = read(ROOT / 'protocols/durable_runtime_source_v1.lock.json')
    require(same(read(out / 'protocol.json'), protocol) and manifest['baseline_commit'] == protocol['baseline_commit'],
            'protocol mismatch')
    require(sha(out / 'protected-test.py.txt') == protocol['protected_test_sha256'], 'protected test mismatch')
    require({name: sha(out / 'baseline/source' / (name + '.txt')) for name in ASSETS}
            == BASELINE_SOURCE_SHA256, 'baseline source mismatch')
    require(number(manifest['wall_seconds']) and manifest['wall_seconds'] <= protocol['max_wall_seconds'], 'wall budget')
    rebuilt = {}
    for label in ['baseline', 'candidate']:
        candidate = label == 'candidate'
        cases = ET.parse(out / label / 'tests.xml').findall('.//testcase')
        require(len(cases) == protocol['cases'] and
                len({c.attrib['name'] for c in cases}) == protocol['cases'] and
                sum(c.find('failure') is not None for c in cases) == protocol[label + '_expected']['failures'] and
                not any(c.find('error') is not None or c.find('skipped') is not None for c in cases), 'test verdicts mismatch')
        require(read(out / label / 'exit.json') == {'exit_code': 0 if candidate else 1}, 'pytest exit mismatch')
        expected_sources = {name: sha(out / label / 'source' / (name + '.txt')) for name in ASSETS}
        for case in ['changed-source', 'matching-control']:
            folder = out / label / 'faults' / case
            records = read(folder / 'processes.json')
            require(all(type(r['exit_code']) is int for r in records), 'invalid process exit code')
            changed = case == 'changed-source'
            require(len(records) == (3 if candidate and changed else 2), 'resume process set mismatch')
            require(records[0]['runtime'] == 'a' and records[0]['mode'] == 'crash' and
                    records[0]['exit_code'] == 42 and same(strict_json(records[0]['stdout']), {'token': 1}), 'crash missing')
            require(records[1]['runtime'] == 'b' and records[1]['mode'] == 'resume', 'resume identity mismatch')
            resumed = strict_json(records[1]['stdout'])
            require(type(resumed['token']) is int, 'invalid resume token')
            require(resumed['loaded_marker'] == ('runtime-b' if changed else None), 'changed code did not load')
            replay = audit_durable(folder / 'after-b')
            require(same(replay, resumed['audit']), 'resume export mismatch')
            require(resumed['exported_runner_sha256'] == expected_sources['runner.py'], 'enrolled runner changed')
            require((resumed['used_runner_sha256'] != resumed['exported_runner_sha256']) if changed else
                    (resumed['used_runner_sha256'] == resumed['exported_runner_sha256']), 'runtime change mismatch')
            if candidate and changed:
                require(records[1]['exit_code'] == 17 and resumed['status'] == 'refused' and
                        resumed['error_type'] == 'RuntimeMismatchError' and resumed['token'] == 1 and
                        replay['outcomes'] == {'running': 1} and replay['attempts'] == 1, 'changed source was not refused')
                state = read(folder / 'after-b/state.json')
                require([e['kind'] for e in state['events']] == ['enrolled', 'claimed'], 'refusal mutated job state')
                restored = records[2]
                require(restored['runtime'] == 'a' and restored['mode'] == 'resume' and restored['exit_code'] == 0,
                        'enrolled runtime failed to restore')
                resumed = strict_json(restored['stdout'])
                require(type(resumed['token']) is int, 'invalid restored token')
                replay = audit_durable(folder / 'after-a')
                require(same(resumed['audit'], replay) and resumed['used_runner_sha256'] == resumed['exported_runner_sha256'],
                        'restored provenance mismatch')
            else:
                require(records[1]['exit_code'] == 0 and resumed['status'] == 'completed', 'control/baseline changed')
            require(resumed['token'] == 2 and replay['outcomes'] == {'candidate_passed': 1} and
                    replay['attempts'] == 2 and replay['events'] == 5, 'terminal recovery mismatch')
            final = folder / ('after-a' if candidate and changed else 'after-b')
            for name in ASSETS:
                require(sha(final / 'snapshot/vare' / name) == expected_sources[name], 'export executor changed')
        rows = []
        for pair in range(protocol['timing_pairs']):
            directory = out / label / ('pair-%d' % pair)
            process = read(out / label / ('pair-%d-process.json' % pair))
            order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
            require(type(process['exit_code']) is int and process['exit_code'] == 0 and
                    process['order'] == order, 'cost process mismatch')
            row = read(directory / 'timing.json')
            require(all(number(v) for k, v in row.items() if k not in {'replay', 'used_executor_sha256'}), 'invalid cost')
            replay = audit_durable(directory / 'export')
            require(same(row['replay'], replay) and replay['outcomes'] == {'candidate_passed': protocol['timing_jobs']}
                    and replay['attempts'] == protocol['timing_jobs'], 'cost workload mismatch')
            state = read(directory / 'export/state.json')
            require({j['id'] for j in state['jobs']} == {'cost-%02d' % i for i in range(protocol['timing_jobs'])}
                    and all(j['token'] == 1 for j in state['jobs']), 'cost job set mismatch')
            require(row['used_executor_sha256'] == expected_sources, 'cost runtime source mismatch')
            for name in ASSETS:
                require(sha(directory / 'export/snapshot/vare' / name) == expected_sources[name], 'cost snapshot mismatch')
            latencies = [read(directory / 'export/attempts' / j['id'] / '1/record.json')['execution_seconds']
                         for j in state['jobs']]
            require(row['latency_p50_seconds'] == statistics.median(latencies) and
                    row['latency_p95_seconds'] == statistics.quantiles(latencies, n=20, method='inclusive')[18],
                    'cost latency mismatch')
            require(row['jobs_per_second'] == protocol['timing_jobs'] / row['wall_seconds'], 'throughput mismatch')
            for task in (directory / 'export/snapshot').glob('*/task.json'):
                descriptor = read(task)
                require(descriptor['fixture_behavior'] == 'valid' and
                        descriptor['fixture_delay'] == protocol['grader_delay_seconds'], 'cost fixture mismatch')
            rows.append(row)
        rebuilt[label] = {key: statistics.median(row[key] for row in rows)
                          for key in rows[0] if key not in {'replay', 'used_executor_sha256'}}
    require(rebuilt == read(out / 'summary.json'), 'summary mismatch')
    return {'status': 'verified', 'cases': 15, 'cost_pairs': protocol['timing_pairs'], 'summary': rebuilt,
            'scope': 'declared executor source consistency and descriptive CPU costs; no model learning'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    print(json.dumps(verify(parser.parse_args().bundle.absolute()), indent=2))
