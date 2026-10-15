"""Offline audit must reject rehashed false provenance and ambiguous records."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'results/durable-runtime-source-v1/comparison-v1.zip'
AUDITOR = ROOT / 'scripts/audit_durable_runtime_source_v1.py'


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('mutation', ['inventory', 'refusal', 'used-source', 'restored-source',
                                    'baseline-source', 'cost-runtime', 'summary', 'bool-token', 'duplicate'])
def test_rehashed_false_provenance_is_rejected(tmp_path, optimized, mutation):
    target = tmp_path / 'bundle'
    with zipfile.ZipFile(BUNDLE) as archive:
        archive.extractall(target)
    if mutation == 'inventory':
        (target / 'extra').write_text('unlisted')
    else:
        if mutation in {'refusal', 'used-source', 'restored-source', 'bool-token', 'duplicate'}:
            path = target / 'candidate/faults/changed-source/processes.json'
            records = json.loads(path.read_text())
            value = json.loads(records[1]['stdout'])
            if mutation == 'refusal':
                value['status'] = 'completed'
                records[1]['exit_code'] = 0
            elif mutation == 'used-source':
                value['used_runner_sha256'] = value['exported_runner_sha256']
            elif mutation == 'bool-token':
                value['token'] = True
            elif mutation == 'restored-source':
                restored = json.loads(records[2]['stdout'])
                restored['used_runner_sha256'] = value['used_runner_sha256']
                records[2]['stdout'] = json.dumps(restored)
            records[1]['stdout'] = json.dumps(value)
            if mutation == 'duplicate':
                records[1]['stdout'] = records[1]['stdout'][:-1] + ', "status": "refused"}'
            path.write_text(json.dumps(records))
        elif mutation == 'baseline-source':
            path = target / 'baseline/source/durable.py.txt'
            path.write_bytes(path.read_bytes() + b'\n# not the pinned source\n')
        elif mutation == 'cost-runtime':
            path = target / 'candidate/pair-0/timing.json'
            value = json.loads(path.read_text())
            value['used_executor_sha256']['durable.py'] = '0' * 64
            path.write_text(json.dumps(value))
        else:
            path = target / 'summary.json'
            value = json.loads(path.read_text())
            value['candidate']['wall_seconds'] += 1
            path.write_text(json.dumps(value))
        manifest_path = target / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files'] = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in target.rglob('*') if p.is_file() and p != manifest_path}
        manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run([sys.executable, *(['-O'] if optimized else []), str(AUDITOR), str(target)],
                            cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and 'ValueError' in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize('optimized', [False, True])
def test_positive_replay_is_offline(tmp_path, optimized):
    result = subprocess.run([sys.executable, *(['-O'] if optimized else []), str(AUDITOR), str(BUNDLE)],
                            cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src'),
                                              'PATH': '', 'GIT_DIR': str(tmp_path / 'missing')},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0 and json.loads(result.stdout)['status'] == 'verified', result.stderr
