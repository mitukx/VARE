"""Local Git fixture factory; no network, package installation, or model use."""
import json
from pathlib import Path
import subprocess

from vare.runner import Job, digest, encode

FIXTURE = Path(__file__).with_name('fixture_grader.py')


def make_fixture(parent, behavior, *, delay=0, timeout=3, limit=1048576):
    root = parent / 'trusted'
    task_root = root / behavior
    task_root.mkdir(parents=True)
    evaluator = task_root / 'grade.py'
    evaluator.write_bytes(FIXTURE.read_bytes())
    workspace = parent / ('candidate-' + behavior)
    workspace.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(workspace), *args], stderr=subprocess.DEVNULL)
    git('init', '--quiet')
    (workspace / 'source.txt').write_text('original\n')
    git('add', 'source.txt')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '--quiet', '-m', 'Fixture source')
    revision = git('rev-parse', 'HEAD').decode().strip()
    task = {'task_id': 'fixture-' + behavior, 'source': {'files': ['source.txt'], 'base_revision': revision},
            'grader': {'path': behavior + '/grade.py'}, 'fixture_behavior': behavior, 'fixture_delay': delay}
    (task_root / 'task.json').write_bytes(encode(task))
    (task_root / 'TASK.md').write_text('Synthetic runner fault injection.\n')
    lock = {'task_id': task['task_id'], 'locked_hashes': {
        'task_json_sha256': digest((task_root / 'task.json').read_bytes()),
        'task_brief_sha256': digest((task_root / 'TASK.md').read_bytes()),
        'evaluator_sha256': digest(evaluator.read_bytes())}}
    (task_root / 'protocol.lock.json').write_bytes(encode(lock))
    return root, Job(behavior, task_root, workspace, timeout, limit)
