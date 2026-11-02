#!/usr/bin/env python3
"""Independently grade one recorded Transformers #50219 agent workspace."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TASK_ID = 'transformers-50219-posix-paths-agent-diagnostic-001'
SOURCE_PATH = 'utils/tests_fetcher.py'
TRUSTED_TEST_SHA256 = 'd61b904a44f89da35578b4db22391c0e4e79aa18957617725b7f24f4326d87e1'
QUALIFICATION_RUNNER_SHA256 = '2bb58cede057b6f51a19dbcd9ab85435a1f92964dce845ba377de8e0fa6f9cf2'
MAX_CPU_SECONDS = 120
MAX_WALL_SECONDS = 180
MAX_RSS_BYTES = 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_qualification_runner(path: Path) -> Any:
    if sha256(path) != QUALIFICATION_RUNNER_SHA256:
        raise ValueError('source-qualification runner differs from its frozen hash')
    spec = importlib.util.spec_from_file_location('frozen_transformers_50219_qualification', path)
    if spec is None or spec.loader is None:
        raise ImportError('could not load the frozen source-qualification helpers')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--agent-record', type=Path, required=True)
    parser.add_argument('--trusted-test', type=Path, required=True)
    parser.add_argument('--qualification-runner', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve(strict=True)
    record_path = args.agent_record.resolve(strict=True)
    test_path = args.trusted_test.resolve(strict=True)
    runner_path = args.qualification_runner.resolve(strict=True)
    output_path = args.output.resolve()
    source_path = workspace / SOURCE_PATH
    if not source_path.is_file() or source_path.is_symlink():
        raise SystemExit('candidate workspace must contain the regular allowlisted source file')
    if test_path.is_relative_to(workspace) or record_path.is_relative_to(workspace):
        raise SystemExit('trusted inputs must remain outside the candidate workspace')
    if not output_path.parent.is_dir():
        raise SystemExit('precreated grade output directory is missing')
    if sha256(test_path) != TRUSTED_TEST_SHA256:
        raise SystemExit('trusted regression test differs from the frozen hash')
    agent = json.loads(record_path.read_text(encoding='utf-8'))
    if agent.get('task_id') != TASK_ID or agent.get('candidate_diff_validation_error') is not None:
        raise SystemExit('agent record is not a valid result for the frozen task')
    source_hashes = agent.get('source_hashes_after')
    if not isinstance(source_hashes, dict) or source_hashes.get(SOURCE_PATH) != sha256(source_path):
        raise SystemExit('candidate source does not match the durable agent record')
    diff = agent.get('candidate_diff')
    if not isinstance(diff, str) or hashlib.sha256(diff.encode('utf-8')).hexdigest() != agent.get('candidate_diff_sha256'):
        raise SystemExit('agent diff does not match its recorded hash')
    tracked = subprocess.check_output(['git', '-C', str(workspace), 'ls-files'], text=True).splitlines()
    if tracked != [SOURCE_PATH]:
        raise SystemExit('candidate workspace tracked-file inventory differs from the task allowlist')
    status = subprocess.check_output(
        ['git', '-C', str(workspace), 'status', '--porcelain=v1', '--untracked-files=all'], text=True,
    ).splitlines()
    if any(line[:3] != ' M ' or line[3:] != SOURCE_PATH for line in status):
        raise SystemExit('candidate workspace has staged, untracked, or non-allowlisted changes')
    changed = subprocess.check_output(
        ['git', '-C', str(workspace), 'diff', '--name-only', 'HEAD', '--'], text=True,
    ).splitlines()
    if any(name != SOURCE_PATH for name in changed):
        raise SystemExit('candidate changed a non-allowlisted path')

    qualification = load_qualification_runner(runner_path)
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))
    signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(TimeoutError('wall limit exceeded')))
    signal.alarm(MAX_WALL_SECONDS)
    started = time.monotonic()
    upstream = None
    behavioral = []
    execution_error = None
    try:
        upstream = qualification.run_upstream_regression(test_path, workspace)
        behavioral = qualification.run_windows_path_behavior(source_path.read_text(encoding='utf-8'))
    except BaseException as exc:
        execution_error = f'{type(exc).__name__}: {exc}'
    finally:
        signal.alarm(0)
    elapsed = time.monotonic() - started
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(usage if sys.platform == 'darwin' else usage * 1024)
    resource_breach = peak_rss_bytes > MAX_RSS_BYTES
    if upstream and isinstance(upstream.get('error'), str) and len(upstream['error']) > 4000:
        full_error = upstream['error']
        upstream['error_sha256'] = hashlib.sha256(full_error.encode('utf-8')).hexdigest()
        upstream['error_chars'] = len(full_error)
        upstream['error'] = full_error[:4000]
        upstream['error_truncated'] = True
    grade_pass = (
        execution_error is None and not resource_breach
        and upstream is not None and upstream.get('status') == 'PASS'
        and [item.get('status') for item in behavioral] == ['PASS', 'PASS']
    )
    result = {
        'schema_version': 1,
        'task_id': TASK_ID,
        'status': 'PASS' if grade_pass else 'FAIL',
        'candidate_source_sha256': sha256(source_path),
        'agent_attempt_sha256': sha256(record_path),
        'candidate_diff_sha256': agent['candidate_diff_sha256'],
        'edit_applied': agent.get('edit_applied') is True,
        'trusted_test_sha256': TRUSTED_TEST_SHA256,
        'qualification_runner_sha256': QUALIFICATION_RUNNER_SHA256,
        'upstream_regression': upstream,
        'independent_windows_path_behavior': behavioral,
        'execution_error': execution_error,
        'resource_breach': resource_breach,
        'elapsed_seconds': elapsed,
        'peak_rss_bytes': peak_rss_bytes,
        'limits': {
            'cpu_seconds': MAX_CPU_SECONDS,
            'wall_seconds': MAX_WALL_SECONDS,
            'observed_peak_rss_bytes_max': MAX_RSS_BYTES,
            'rss_enforcement': 'post-run measurement; outer process-tree limiter also required',
        },
        'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'platform': sys.platform,
    }
    with output_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'task_id': TASK_ID, 'status': result['status'], 'upstream': upstream, 'behavior': behavioral, 'execution_error': execution_error}))
    return 0 if grade_pass else 2


if __name__ == '__main__':
    raise SystemExit(main())
