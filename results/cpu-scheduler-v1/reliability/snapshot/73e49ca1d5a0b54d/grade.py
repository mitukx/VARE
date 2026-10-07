"""Fault injection only; synthetic outcomes are not task or model capability."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--task-root', type=Path, required=True)
a = p.parse_args()
task = json.loads((a.task_root / 'task.json').read_text())
behavior = task['fixture_behavior']
if behavior == 'timeout':
    time.sleep(30)
if behavior == 'flood':
    while True:
        os.write(1, b'x' * 16384)
if behavior == 'crash':
    print(json.dumps({'grader_error': 'InjectedError'}))
    sys.exit(1)
if behavior == 'malformed':
    print('not JSON')
    sys.exit(0)
if behavior == 'duplicate':
    print('{"passed":true,"passed":false}')
    sys.exit(0)
if behavior == 'nonfinite':
    print('{"value":1e999}')
    sys.exit(0)
if behavior == 'mutate':
    (a.workspace / 'source.txt').write_text('changed')
if behavior == 'child':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('child_pid=' + str(child.pid), file=sys.stderr, flush=True)
if behavior == 'inherited_pipe':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    print('child_pid=' + str(child.pid), file=sys.stderr, flush=True)
if behavior in {'valid', 'reject'}:
    time.sleep(task.get('fixture_delay', 0))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
def git(args):
    return subprocess.check_output(['git', '-C', str(a.workspace), *args])
result = {'schema_version': 1, 'task_id': task['task_id'],
          'workspace_revision': git(['rev-parse', 'HEAD']).decode().strip(),
          'candidate_diff_sha256': hashlib.sha256(git(['diff', '--binary', task['source']['base_revision'], '--', 'source.txt'])).hexdigest(),
          'source_file_sha256': {'source.txt': sha(a.workspace / 'source.txt')},
          'evaluator_sha256': sha(Path(__file__)),
          'task_descriptor_sha256': sha(a.task_root / 'task.json'),
          'passed': behavior != 'reject', 'checks': {'synthetic': True},
          'failures': ['injected candidate failure'] if behavior == 'reject' else []}
if behavior == 'wrong_task':
    result['task_id'] = 'wrong'
if behavior == 'fake_hash':
    result['candidate_diff_sha256'] = '0' * 64
if behavior == 'wrong_exit':
    result['passed'] = True
print(json.dumps(result), flush=True)
sys.exit(2 if behavior in {'reject', 'wrong_exit'} else 0)
