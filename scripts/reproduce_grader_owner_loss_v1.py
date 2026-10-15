"""Compare a pinned runner with owner-loss containment on real CPU processes."""
import argparse
import asyncio
from dataclasses import replace
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'protocols/grader_owner_loss_v1.lock.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def timing(out, protocol):
    from benchmarks.scheduler.fixtures import make_fixture
    from vare.runner import audit, run_campaign
    out.mkdir()
    with tempfile.TemporaryDirectory(prefix='vare-grader-cost-') as temp:
        parent = Path(temp)
        root, job = make_fixture(parent, 'valid', delay=protocol['grader_delay_seconds'])
        jobs = [replace(job, id='cost-%02d' % i) for i in range(protocol['timing_jobs'])]
        tracemalloc.start()
        started = time.perf_counter()
        summary = asyncio.run(run_campaign(jobs, out / 'campaign', protocol['timing_workers'], root))
        wall = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        audit_started = time.perf_counter()
        replay = audit(out / 'campaign')
        audit_seconds = time.perf_counter() - audit_started
        require(summary['outcomes'] == {'candidate_passed': len(jobs)}, 'cost workload failed')
        records = [json.loads((out / 'campaign/jobs' / j.id / 'record.json').read_text()) for j in jobs]
        factor = 1 if sys.platform == 'darwin' else 1024
        result = {'wall_seconds': wall, 'audit_seconds': audit_seconds,
                  'jobs_per_second': len(jobs) / wall, 'python_peak_bytes': peak,
                  'controller_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * factor,
                  'largest_reaped_child_peak_rss_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * factor,
                  'latency_p50_seconds': statistics.median(r['execution_seconds'] for r in records),
                  'latency_p95_seconds': statistics.quantiles(
                      [r['execution_seconds'] for r in records], n=20, method='inclusive')[18],
                  'replay': replay}
        write(out / 'timing.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timing-worker', action='store_true')
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.timing_worker:
        timing(args.output, protocol)
        return
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    test = ROOT / protocol['protected_test']
    require(sha(test) == protocol['protected_test_sha256'], 'protected regression changed')
    revision = protocol['baseline_commit']
    if subprocess.run(['git', 'cat-file', '-e', revision + '^{commit}'], cwd=ROOT,
                      stderr=subprocess.DEVNULL).returncode:
        subprocess.run(['git', 'fetch', '--filter=blob:none', 'origin', revision], cwd=ROOT,
                       check=True, timeout=120)
    (out / 'protocol.json').write_bytes(PROTOCOL.read_bytes())
    (out / 'protected-test.py.txt').write_bytes(test.read_bytes())
    sources = {}
    with tempfile.TemporaryDirectory(prefix='vare-owner-comparison-') as temp:
        baseline = Path(temp) / 'baseline'
        paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision, 'src'],
                                        cwd=ROOT).decode().splitlines()
        for name in paths:
            dest = baseline / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(subprocess.check_output(['git', 'show', revision + ':' + name], cwd=ROOT))
        for label, source in [('baseline', baseline), ('candidate', ROOT)]:
            sources[label] = source
            snapshot = out / label / 'source'
            snapshot.mkdir(parents=True)
            for name in ['runner.py', 'durable.py', '_grader_guard.py']:
                path = source / 'src/vare' / name
                if path.exists():
                    (snapshot / (name + '.txt')).write_bytes(path.read_bytes())
            env = {**os.environ, 'PYTHONPATH': str(source / 'src') + os.pathsep + str(ROOT),
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
                   'VARE_OWNER_LOSS_EVIDENCE_DIR': str(out / label / 'faults')}
            with (out / label / 'pytest.txt').open('w') as log:
                result = subprocess.run([sys.executable, '-m', 'pytest', '-c', '/dev/null',
                    '-p', 'no:cacheprovider', '-o', 'pythonpath=' + str(source / 'src'), str(test),
                    '--junitxml=' + str(out / label / 'tests.xml')], cwd=temp, env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=45)
            write(out / label / 'exit.json', {'exit_code': result.returncode})
        for pair in range(protocol['timing_pairs']):
            order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
            for label in order:
                source = sources[label]
                env = {**os.environ, 'PYTHONPATH': str(source / 'src') + os.pathsep + str(ROOT),
                       'PYTHONDONTWRITEBYTECODE': '1'}
                destination = out / label / ('pair-%d' % pair)
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    '--timing-worker', '--output', str(destination)], cwd=temp, env=env,
                    capture_output=True, timeout=30)
                write(out / label / ('pair-%d-process.json' % pair),
                      {'exit_code': result.returncode, 'stdout': result.stdout.decode(),
                       'stderr': result.stderr.decode(), 'order': order})
                require(result.returncode == 0, 'timing worker failed; retain artifacts')
    summary = {}
    for label in ['baseline', 'candidate']:
        rows = [json.loads((out / label / ('pair-%d/timing.json' % i)).read_text())
                for i in range(protocol['timing_pairs'])]
        summary[label] = {key: statistics.median(row[key] for row in rows)
                          for key in rows[0] if key != 'replay'}
    write(out / 'summary.json', summary)
    write(out / 'manifest.json', {'schema_version': 1, 'baseline_commit': revision,
        'candidate_parent': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        'runtime': {'python': platform.python_version(), 'platform': platform.platform()},
        'files': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    for label in ['baseline', 'candidate']:
        cases = ET.parse(out / label / 'tests.xml').findall('.//testcase')
        expected = protocol[label + '_expected']
        require(len(cases) == 3 and sum(c.find('failure') is not None for c in cases) == expected['failures']
                and not any(c.find('error') is not None or c.find('skipped') is not None for c in cases),
                label + ' regression mismatch; retain artifacts')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
