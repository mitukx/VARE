"""Adversarial E0 binding checks against a real local Git workspace; no code runs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_record_coding_agent_baseline import fixture, dump, digest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/record_coding_agent_baseline.py'


def git(root: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def git_fixture(tmp_path: Path):
    data = fixture(tmp_path)
    work = tmp_path / 'isolated-source'
    work.mkdir()
    git(work, 'init', '-q')
    git(work, 'config', 'user.name', 'Fixture')
    git(work, 'config', 'user.email', 'fixture@example.invalid')
    source = work / 'src/pkg/x.py'
    source.parent.mkdir(parents=True)
    source.write_text('old\n')
    git(work, 'add', '.')
    git(work, 'commit', '-qm', 'base fixture')
    base = git(work, 'rev-parse', 'HEAD')
    data['protocol']['task']['base_revision'] = base
    task_path = data['task_root'] / 'task.json'
    task = json.loads(task_path.read_text())
    task['source']['base_revision'] = base
    dump(task_path, task)
    task_sha = hashlib.sha256(task_path.read_bytes()).hexdigest()
    lock_path = data['task_root'] / 'protocol.lock.json'
    lock = json.loads(lock_path.read_text())
    lock['locked_hashes']['task_json_sha256'] = task_sha
    dump(lock_path, lock)
    data['protocol']['task']['hashes']['task.json'] = task_sha
    data['protocol']['task']['hashes']['protocol.lock.json'] = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    data['agent']['task_input_hashes'] = copy.deepcopy(data['protocol']['task']['hashes'])
    data['agent']['task_revisions']['base'] = base
    data['agent']['workspace_revision'] = base
    data['grade']['workspace_revision'] = base
    data['grade']['task_descriptor_sha256'] = task_sha
    source.write_text('new\n')
    diff = git(work, 'diff', '--binary', base, '--', 'src/pkg/x.py')
    data['agent']['candidate_diff'] = diff
    data['agent']['candidate_diff_sha256'] = digest(diff.encode())
    data['grade']['candidate_diff_sha256'] = digest(diff.encode())
    data['agent']['source_hashes_before'] = {'src/pkg/x.py': digest(b'old\n')}
    data['agent']['source_hashes_after'] = {'src/pkg/x.py': digest(b'new\n')}
    data['grade']['source_file_sha256'] = {'src/pkg/x.py': digest(b'new\n')}
    data['protocol']['interaction'] = {'allowed_tools': ['apply_patch', 'finish']}
    data['protocol']['workspace_verification'] = {'required': True}
    return data, work


def audit(tmp_path: Path, data: dict, workspace: Path | None, label: str):
    records = tmp_path / f'input-{label}'
    dump(records / 'protocol.json', data['protocol'])
    dump(records / 'agent.json', data['agent'])
    dump(records / 'grade.json', data['grade'])
    output = tmp_path / f'audit-{label}'
    command = [sys.executable, str(SCRIPT), '--protocol', str(records / 'protocol.json'),
               '--task-root', str(data['task_root']),
               '--agent-record', str(records / 'agent.json'),
               '--grader-record', str(records / 'grade.json'), '--output-dir', str(output)]
    if workspace is not None:
        command += ['--workspace', str(workspace)]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    assert (output / 'summary.json').exists(), proc.stderr
    return proc.returncode, json.loads((output / 'summary.json').read_text()), output


def test_actual_git_state_binds_agent_grader_and_source(tmp_path: Path):
    data, work = git_fixture(tmp_path)
    code, result, out = audit(tmp_path, data, work, 'ok')
    assert code == 0, result
    assert result['record_integrity'] == 'valid'
    assert result['workspace_audit']['verified_source_sha256'] == data['agent']['source_hashes_after']
    assert (out / 'manifest.json').exists()


def test_v2_requires_a_verifiable_workspace(tmp_path: Path):
    data, _ = git_fixture(tmp_path)
    code, result, _ = audit(tmp_path, data, None, 'no-workspace')
    assert code == 1 and result['record_integrity'] == 'invalid'


@pytest.mark.parametrize('mutation', [
    'disk_tamper', 'source_base', 'staged', 'untracked', 'source_symlink',
    'out_of_scope', 'forged_after', 'forged_diff', 'no_successful_patch',
    'unknown_tool', 'workspace_is_task_root',
])
def test_v2_fails_closed_on_workspace_and_record_attacks(tmp_path: Path, mutation: str):
    data, work = git_fixture(tmp_path)
    src = work / 'src/pkg/x.py'
    if mutation == 'disk_tamper':
        src.write_text('different\n')
    elif mutation == 'source_base':
        data['agent']['source_hashes_before']['src/pkg/x.py'] = '0' * 64
    elif mutation == 'staged':
        git(work, 'add', 'src/pkg/x.py')
    elif mutation == 'untracked':
        (work / 'unknown.txt').write_text('untracked')
    elif mutation == 'source_symlink':
        src.unlink()
        src.symlink_to('/tmp/x-does-not-exist')
    elif mutation == 'out_of_scope':
        file = work / 'other.py'
        file.write_text('initial')
        git(work, 'add', 'other.py')
        git(work, 'commit', '-qm', 'temp extra tracked')
    elif mutation == 'forged_after':
        data['agent']['source_hashes_after']['src/pkg/x.py'] = '0' * 64
        data['grade']['source_file_sha256']['src/pkg/x.py'] = '0' * 64
    elif mutation == 'forged_diff':
        different = data['agent']['candidate_diff'].replace('+new', '+NEW')
        data['agent']['candidate_diff'] = different
        data['agent']['candidate_diff_sha256'] = digest(different.encode())
        data['grade']['candidate_diff_sha256'] = digest(different.encode())
    elif mutation == 'no_successful_patch':
        data['agent']['transcript'][0]['tool_calls'][0]['result'] = {'applied': False}
    elif mutation == 'unknown_tool':
        data['agent']['transcript'][0]['tool_calls'][0]['name'] = 'run_shell'
    elif mutation == 'workspace_is_task_root':
        work = data['task_root']
    code, result, _ = audit(tmp_path, data, work, mutation)
    assert code == 1 and result['record_integrity'] == 'invalid', result


def test_workspace_negative_grader_remains_valid_negative(tmp_path: Path):
    data, work = git_fixture(tmp_path)
    data['grade'].update({'passed': False, 'failures': [{'case_id': 'regression'}]})
    code, result, _ = audit(tmp_path, data, work, 'valid-negative')
    assert code == 0 and result['task_success'] is False
    assert result['failure_category'] == 'independent_grader_reject'
