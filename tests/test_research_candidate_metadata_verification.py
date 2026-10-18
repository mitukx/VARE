"""Metadata evidence checks must survive interpreter optimization."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


COLLECTOR = Path(os.environ.get('VARE_METADATA_COLLECTOR',
                 Path(__file__).resolve().parents[1] / 'reproducers/research_candidate_metadata_20261016.py')).resolve()
EXPECTED_REVISION = '467ff55dbf6accf0a59e1f0c7f5741ac53ce21de'


def seal(root):
    sums = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.iterdir() if p.name != 'SHA256SUMS.json'}
    (root / 'SHA256SUMS.json').write_text(json.dumps(sums))


@pytest.mark.parametrize('mode', ['normal', 'flag', 'environment'])
@pytest.mark.parametrize('corruption', ['clean', 'inventory', 'checksum', 'model_inference_calls',
                                      'optimizer_calls', 'task_payload_downloads', 'request_length', 'request_digest',
                                      'source_revision'])
def test_verification_survives_optimized_python(tmp_path, mode, corruption):
    payload = b'{"metadata":true}\n'
    (tmp_path / 'response.json').write_bytes(payload)
    capture = {'repository_revision':EXPECTED_REVISION,
               'model_inference_calls':0, 'optimizer_calls':0, 'task_payload_downloads':0,
               'requests':[{'retained_file':'response.json', 'response_bytes':len(payload),
                            'response_sha256':hashlib.sha256(payload).hexdigest()}]}
    if corruption in ('model_inference_calls', 'optimizer_calls', 'task_payload_downloads'):
        capture[corruption] = 1
    elif corruption == 'request_length':
        capture['requests'][0]['response_bytes'] += 1
    elif corruption == 'request_digest':
        capture['requests'][0]['response_sha256'] = '0' * 64
    elif corruption == 'source_revision':
        capture['repository_revision'] = '0' * 40
    (tmp_path / 'capture.json').write_text(json.dumps(capture))
    seal(tmp_path)
    if corruption == 'checksum':
        (tmp_path / 'response.json').write_bytes(payload + b' ')
    elif corruption == 'inventory':
        (tmp_path / 'unexpected.json').write_text('{}')
    command = [sys.executable, *(['-O'] if mode == 'flag' else []), str(COLLECTOR), '--verify', str(tmp_path)]
    env = {key:value for key,value in os.environ.items() if key != 'PYTHONOPTIMIZE'}
    if mode == 'environment':
        env['PYTHONOPTIMIZE'] = '1'
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    if corruption == 'clean':
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['metadata_integrity'] == 'PASS'
    else:
        assert result.returncode != 0, result.stdout
        assert 'metadata_integrity' not in result.stdout
