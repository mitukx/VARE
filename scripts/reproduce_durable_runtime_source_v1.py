"""Pinned-source durable execution comparison; no model inference or training."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
import xml.etree.ElementTree as ET

from reproduce_grader_owner_loss_v1 import require, sha, write

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'protocols/durable_runtime_source_v1.lock.json'
ASSETS = ['__init__.py', '__main__.py', 'runner.py', 'durable.py', '_grader_guard.py']


def timing(out, protocol):
    from benchmarks.scheduler.fixtures import make_fixture
    from vare import durable
    out.mkdir()
    with tempfile.TemporaryDirectory(prefix='vare-runtime-cost-') as temp:
        root, job = make_fixture(Path(temp), 'valid', delay=protocol['grader_delay_seconds'])
        jobs = [replace(job, id='cost-%02d' % i) for i in range(protocol['timing_jobs'])]
        store = durable.Store.create(Path(temp) / 'state.db', jobs, root)
        tracemalloc.start()
        started = time.perf_counter()
        durable.run_workers(store.path, protocol['timing_workers'], 10)
        wall = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        started = time.perf_counter()
        store.export(out / 'export')
        export_seconds = time.perf_counter() - started
        started = time.perf_counter()
        replay = durable.audit_durable(out / 'export')
        audit_seconds = time.perf_counter() - started
        require(replay['outcomes'] == {'candidate_passed': len(jobs)} and replay['attempts'] == len(jobs),
                'cost work did not complete once per job')
        records = [json.loads((out / 'export/attempts' / j.id / '1/record.json').read_text()) for j in jobs]
        latencies = [r['execution_seconds'] for r in records]
        factor = 1 if sys.platform == 'darwin' else 1024
        write(out / 'timing.json', {'wall_seconds': wall, 'jobs_per_second': len(jobs) / wall,
            'latency_p50_seconds': statistics.median(latencies),
            'latency_p95_seconds': statistics.quantiles(latencies, n=20, method='inclusive')[18],
            'export_seconds': export_seconds, 'audit_seconds': audit_seconds,
            'controller_python_peak_bytes': peak,
            'controller_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * factor,
            'largest_reaped_child_peak_rss_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * factor,
            'used_executor_sha256': {name: sha(Path(durable.__file__).with_name(name)) for name in ASSETS},
            'replay': replay})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timing-worker', action='store_true')
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.timing_worker:
        timing(args.output, protocol)
        return
    started = time.monotonic()

    def budget(maximum):
        remaining = protocol['max_wall_seconds'] - (time.monotonic() - started)
        require(remaining > 0, 'wall budget exhausted; retain output')
        return min(maximum, remaining)

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    test = ROOT / protocol['protected_test']
    require(sha(test) == protocol['protected_test_sha256'], 'protected test changed')
    (out / 'protocol.json').write_bytes(PROTOCOL.read_bytes())
    (out / 'protected-test.py.txt').write_bytes(test.read_bytes())
    (out / 'reproducer.py.txt').write_bytes(Path(__file__).read_bytes())
    revision = protocol['baseline_commit']
    if subprocess.run(['git', 'cat-file', '-e', revision + '^{commit}'], cwd=ROOT,
                      stderr=subprocess.DEVNULL, timeout=budget(10)).returncode:
        subprocess.run(['git', 'fetch', '--filter=blob:none', 'origin', revision], cwd=ROOT,
                       check=True, timeout=budget(120))
    with tempfile.TemporaryDirectory(prefix='vare-runtime-compare-') as temp:
        baseline = Path(temp) / 'baseline'
        paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision, 'src'],
                                        cwd=ROOT, timeout=budget(10)).decode().splitlines()
        for name in paths:
            target = baseline / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(['git', 'show', revision + ':' + name],
                                                       cwd=ROOT, timeout=budget(20)))
        sources = {'baseline': baseline, 'candidate': ROOT}
        for label, source in sources.items():
            snapshot = out / label / 'source'
            snapshot.mkdir(parents=True)
            for name in ASSETS:
                (snapshot / (name + '.txt')).write_bytes((source / 'src/vare' / name).read_bytes())
            env = {**os.environ, 'PYTHONPATH': str(source / 'src') + os.pathsep + str(ROOT),
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
                   'VARE_RUNTIME_SOURCE_EVIDENCE_DIR': str(out / label / 'faults')}
            with (out / label / 'pytest.txt').open('w') as log:
                result = subprocess.run([sys.executable, '-m', 'pytest', '-c', '/dev/null',
                    '-p', 'no:cacheprovider', '-o', 'pythonpath=' + str(source / 'src'), str(test),
                    '--junitxml=' + str(out / label / 'tests.xml')], cwd=temp, env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=budget(45))
            write(out / label / 'exit.json', {'exit_code': result.returncode})
        for pair in range(protocol['timing_pairs']):
            order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
            for label in order:
                env = {**os.environ, 'PYTHONPATH': str(sources[label] / 'src') + os.pathsep + str(ROOT),
                       'PYTHONDONTWRITEBYTECODE': '1'}
                run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--timing-worker',
                    '--output', str(out / label / ('pair-%d' % pair))], cwd=temp, env=env,
                    capture_output=True, timeout=budget(40))
                write(out / label / ('pair-%d-process.json' % pair), {'exit_code': run.returncode,
                      'stdout': run.stdout.decode(), 'stderr': run.stderr.decode(), 'order': order})
                require(run.returncode == 0, 'cost worker failed; retain its raw output')
    summary = {}
    for label in ['baseline', 'candidate']:
        rows = [json.loads((out / label / ('pair-%d/timing.json' % i)).read_text())
                for i in range(protocol['timing_pairs'])]
        summary[label] = {key: statistics.median(row[key] for row in rows)
                          for key in rows[0] if key not in {'replay', 'used_executor_sha256'}}
    write(out / 'summary.json', summary)
    write(out / 'manifest.json', {'schema_version': 1, 'baseline_commit': revision,
        'candidate_parent': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        'wall_seconds': time.monotonic() - started,
        'runtime': {'python': platform.python_version(), 'platform': platform.platform()},
        'files': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    for label in ['baseline', 'candidate']:
        cases = ET.parse(out / label / 'tests.xml').findall('.//testcase')
        require(len(cases) == protocol['cases'] and
                sum(c.find('failure') is not None for c in cases) == protocol[label + '_expected']['failures'] and
                not any(c.find('error') is not None or c.find('skipped') is not None for c in cases),
                'regression mismatch; retain raw results')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
