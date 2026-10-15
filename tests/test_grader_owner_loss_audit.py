"""Changed artifacts must fail replay, including after manifest recomputation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'results/grader-owner-loss-v1/comparison-v2.zip'


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('mutation', ['outcome', 'exit', 'deadline', 'group', 'summary', 'latency', 'inventory'])
def test_rehashed_corruption_is_rejected(tmp_path, optimized, mutation):
    target = tmp_path / 'bundle'
    with zipfile.ZipFile(BUNDLE) as archive:
        archive.extractall(target)
    if mutation == 'inventory':
        (target / 'extra').write_text('unexpected')
    else:
        if mutation in {'outcome', 'exit', 'deadline', 'group'}:
            path = target / 'candidate/faults/SIGKILL/observation.json'
            data = json.loads(path.read_text())
            if mutation == 'outcome':
                data['stopped'] = False
            elif mutation == 'exit':
                data['owner_exit'] = 0
            elif mutation == 'deadline':
                data['stop_seconds'] = 3
            else:
                data['receipt']['group'] = data['receipt']['grader']
        elif mutation == 'summary':
            path = target / 'summary.json'
            data = json.loads(path.read_text())
            data['candidate']['wall_seconds'] += 1
        else:
            path = target / 'candidate/pair-0/timing.json'
            data = json.loads(path.read_text())
            data['latency_p50_seconds'] += 1
        path.write_text(json.dumps(data) + '\n')
        manifest_path = target / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files'] = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in target.rglob('*') if p.is_file() and p != manifest_path}
        manifest_path.write_text(json.dumps(manifest) + '\n')
    result = subprocess.run([sys.executable, *(['-O'] if optimized else []),
                             str(ROOT / 'scripts/audit_grader_owner_loss_v1.py'), str(target)],
                            cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and 'ValueError' in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize('mutation', ['escape', 'symlink', 'duplicate'])
def test_archive_paths_are_checked_before_extraction(tmp_path, mutation):
    import runpy
    verify = runpy.run_path(str(ROOT / 'scripts/audit_grader_owner_loss_v1.py'))['verify']
    target = tmp_path / 'bad.zip'
    with zipfile.ZipFile(target, 'w') as archive:
        if mutation == 'escape':
            archive.writestr('../escape', 'outside')
        elif mutation == 'symlink':
            entry = zipfile.ZipInfo('link')
            entry.external_attr = 0o120777 << 16
            archive.writestr(entry, '/tmp')
        else:
            archive.writestr('same', 'one')
            with pytest.warns(UserWarning):
                archive.writestr('same', 'two')
    with pytest.raises(ValueError, match='unsafe|duplicated'):
        verify(target)
    assert not (tmp_path / 'escape').exists()


@pytest.mark.parametrize('optimized', [False, True])
def test_positive_replay_needs_no_git_history(tmp_path, optimized):
    # A broken/missing Git history must not mask the mutation tests above.
    # The reviewed frozen source hashes remain required, even without Git.
    result = subprocess.run([sys.executable, *(['-O'] if optimized else []),
                             str(ROOT / 'scripts/audit_grader_owner_loss_v1.py'), str(BUNDLE)],
                            cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src'),
                                              'PATH': '', 'GIT_DIR': str(tmp_path / 'missing.git')},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['status'] == 'verified'
