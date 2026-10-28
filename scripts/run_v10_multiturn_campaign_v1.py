#!/usr/bin/env python3
"""Frozen-protocol guard for the v9/v10 multi-turn ABBA campaign.

Verifies the protocol hash and every frozen file/model identity, performs the
resource preflight, runs the KV validity probe and the counterbalanced
schedule one process per trajectory under host resource limits, grades each
final source in a separate grader process, and applies the preregistered
stopping rules. Outputs are append-only; an existing run directory is never
overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

REPO = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=1)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def append_event(path: Path, **payload: Any) -> None:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'unix_time': round(time.time(), 3), **payload}, sort_keys=True) + '\n')


def schedule(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    runs = []
    for block_index, block in enumerate(protocol['design']['blocks']):
        for position, letter in enumerate(block['order']):
            runs.append({
                'block': block_index + 1, 'position': position + 1, 'variant': block['variant'],
                'horizon': block['horizon'], 'order': block['order'],
                'arm': protocol['design']['arm_letters'][letter],
            })
    return runs


def terminate(proc: subprocess.Popen[bytes]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def tree_rss(proc: psutil.Process) -> int:
    total = 0
    try:
        for item in [proc, *proc.children(recursive=True)]:
            try:
                total += item.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return total


def guarded(command: list[str], *, timeout_s: float, limits: dict[str, int], swap_start: int,
            stdout: Path, stderr: Path, samples: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                'PYTHONPATH': os.pathsep.join((str(REPO), str(REPO / 'src')))})
    started = time.monotonic()
    swap_at_start = psutil.swap_memory().used
    peak_rss = 0
    max_swap_used = swap_at_start
    min_available = psutil.virtual_memory().available
    abort = None
    with stdout.open('xb') as out, stderr.open('xb') as err, samples.open('x', encoding='utf-8') as sample_stream:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=out, stderr=err, start_new_session=True)
        handle = psutil.Process(proc.pid)
        while proc.poll() is None:
            elapsed = time.monotonic() - started
            rss = tree_rss(handle)
            swap = psutil.swap_memory()
            available = psutil.virtual_memory().available
            peak_rss = max(peak_rss, rss)
            max_swap_used = max(max_swap_used, swap.used)
            min_available = min(min_available, available)
            sample_stream.write(json.dumps({'t': round(elapsed, 2), 'rss': rss, 'swap_used': swap.used,
                                            'available': available}) + '\n')
            if rss > limits['max_process_tree_rss_bytes']:
                abort = 'process_rss_limit'
            elif swap.used - swap_start > limits['max_swap_growth_bytes']:
                abort = 'swap_growth_limit'
            elif available < limits['min_available_memory_bytes']:
                abort = 'available_memory_floor'
            elif swap.free < limits['min_free_swap_bytes']:
                abort = 'free_swap_floor'
            elif elapsed > timeout_s:
                abort = 'timeout'
            if abort:
                terminate(proc)
                break
            time.sleep(0.5)
        proc.wait()
    return {
        'exit_code': proc.returncode, 'abort': abort,
        'guard_wall_seconds': round(time.monotonic() - started, 3),
        'peak_process_tree_rss_bytes': peak_rss,
        'swap_used_start_bytes': swap_at_start, 'swap_used_max_bytes': max_swap_used,
        'swap_growth_during_run_bytes': max_swap_used - swap_at_start,
        'min_available_memory_bytes': min_available,
    }


def verify_identities(protocol: dict[str, Any], python: Path, model: Path, *, hash_weights: bool) -> list[str]:
    problems = []
    for relative, expected in protocol['implementation']['file_sha256'].items():
        path = REPO / relative
        if not path.is_file() or digest(path) != expected:
            problems.append(f'file identity mismatch: {relative}')
    if str(python) != protocol['runtime']['python_executable']:
        problems.append('python executable differs from protocol')
    if str(model) != protocol['model']['snapshot_path']:
        problems.append('model snapshot path differs from protocol')
    for name, expected in protocol['model']['file_sha256'].items():
        if name == 'model.safetensors' and not hash_weights:
            continue
        if digest(model / name) != expected:
            problems.append(f'model file identity mismatch: {name}')
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    expected_sha = (protocol_path.parent / 'protocol.sha256').read_text(encoding='utf-8').split()[0]
    if digest(protocol_path) != expected_sha or protocol.get('status') != 'FROZEN_BEFORE_MODEL_INFERENCE':
        raise SystemExit('refusing to run: protocol hash or frozen status mismatch')
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise SystemExit(f'refusing to overwrite {run_dir}')
    run_dir.mkdir(parents=True)
    events = run_dir / 'events.jsonl'
    python = Path(protocol['runtime']['python_executable'])
    model = Path(protocol['model']['snapshot_path'])
    limits = protocol['resources']
    evidence = protocol_path.parent

    preflight: dict[str, Any] = {'protocol_sha256': expected_sha, 'started_unix': time.time()}
    problems = verify_identities(protocol, python, model, hash_weights=True)
    vm, swap = psutil.virtual_memory(), psutil.swap_memory()
    disk = psutil.disk_usage(str(run_dir))
    heavy = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            rss = proc.info['memory_info'].rss
        except (psutil.Error, AttributeError):
            continue
        if rss > 1 << 30:
            heavy.append({'pid': proc.info['pid'], 'name': proc.info['name'], 'rss_bytes': rss})
    git_head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True).stdout.strip()
    preflight.update({
        'repository_head': git_head, 'tracked_modifications': bool(dirty),
        'identity_problems': problems,
        'total_memory_bytes': vm.total, 'available_memory_bytes': vm.available,
        'swap_total_bytes': swap.total, 'swap_used_bytes': swap.used, 'swap_free_bytes': swap.free,
        'disk_free_bytes': disk.free, 'load_average': os.getloadavg(),
        'processes_over_1GiB_rss': sorted(heavy, key=lambda item: -item['rss_bytes'])[:10],
    })
    checks = {
        'identities_verified': not problems,
        'clean_tracked_tree': not dirty,
        'available_memory_ok': vm.available >= limits['preflight_min_available_memory_bytes'],
        'free_swap_ok': swap.free >= limits['preflight_min_free_swap_bytes'],
        'disk_ok': disk.free >= limits['preflight_min_disk_free_bytes'],
    }
    preflight['checks'] = checks
    preflight['decision'] = 'GO' if all(checks.values()) else 'NO_GO'
    write_json(run_dir / 'preflight.json', preflight)
    if preflight['decision'] != 'GO' or args.preflight_only:
        return 0 if preflight['decision'] == 'GO' else 75

    swap_start = swap.used
    campaign_started = time.monotonic()
    worker = REPO / protocol['implementation']['worker']
    grader = REPO / protocol['implementation']['grader']
    fixtures = evidence / 'fixtures.json'
    budget = protocol['design']['generation']

    probe_dir = run_dir / 'probe'
    probe_dir.mkdir()
    probe_guard = guarded(
        [str(python), str(worker), '--mode', 'probe', '--model', str(model), '--fixtures', str(fixtures),
         '--output', str(probe_dir / 'probe.json')],
        timeout_s=limits['probe_timeout_seconds'], limits=limits, swap_start=swap_start,
        stdout=probe_dir / 'stdout.log', stderr=probe_dir / 'stderr.log', samples=probe_dir / 'resource-samples.jsonl')
    write_json(probe_dir / 'guard.json', probe_guard)
    append_event(events, event='probe_finished', **probe_guard)
    if probe_guard['abort'] and probe_guard['abort'] != 'timeout':
        append_event(events, event='campaign_stopped', reason=f"resource_abort:{probe_guard['abort']}")
        return 3

    stop_reason = None
    timeouts = crashes = 0
    rules = protocol['stopping_rules']
    block_minutes = {str(h): float(m) for h, m in rules['default_block_minutes_estimate'].items()}
    block_started = time.monotonic()
    previous_horizon = None
    for index, run in enumerate(schedule(protocol), start=1):
        if run['position'] == 1:
            if index > 1:
                block_minutes[str(previous_horizon)] = (time.monotonic() - block_started) / 60
            elapsed_min = (time.monotonic() - campaign_started) / 60
            projected = elapsed_min + block_minutes[str(run['horizon'])]
            append_event(events, event='block_budget_check', elapsed_minutes=round(elapsed_min, 2),
                         projected_minutes=round(projected, 2), block=run['block'])
            if projected > rules['campaign_wall_minutes_max']:
                stop_reason = 'campaign_wall_budget'
                break
            block_started = time.monotonic()
        previous_horizon = run['horizon']
        label = f"{index:03d}-b{run['block']}-{run['variant']}-h{run['horizon']}-{run['arm']}"
        run_path = run_dir / 'runs' / label
        run_path.mkdir(parents=True)
        trajectory = run_path / 'trajectory.json'
        append_event(events, event='run_started', label=label, **run)
        guard = guarded(
            [str(python), str(worker), '--mode', 'trajectory', '--arm', run['arm'], '--variant', run['variant'],
             '--horizon', str(run['horizon']), '--run-label', label, '--model', str(model),
             '--fixtures', str(fixtures), '--output', str(trajectory),
             '--turn-token-budget', str(budget['turn_token_budget']),
             '--max-context-tokens', str(budget['max_context_tokens'])],
            timeout_s=limits['trajectory_timeout_seconds'][str(run['horizon'])], limits=limits,
            swap_start=swap_start, stdout=run_path / 'stdout.log', stderr=run_path / 'stderr.log',
            samples=run_path / 'resource-samples.jsonl')
        write_json(run_path / 'guard.json', {**run, 'label': label, **guard})
        append_event(events, event='run_finished', label=label, **guard)
        if guard['abort'] == 'timeout':
            timeouts += 1
        elif guard['abort']:
            stop_reason = f"resource_abort:{guard['abort']}"
            break
        elif guard['exit_code'] != 0 or not trajectory.is_file():
            crashes += 1
        if trajectory.is_file():
            grade = subprocess.run(
                [str(python), str(grader), '--trajectory', str(trajectory), '--fixtures', str(fixtures),
                 '--hidden-tests', str(evidence / 'grader' / 'hidden_tests.json'),
                 '--output', str(run_path / 'grade.json')],
                cwd=REPO, env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(REPO / 'src')},
                capture_output=True, text=True, timeout=120)
            append_event(events, event='graded', label=label, exit_code=grade.returncode)
            record = json.loads(trajectory.read_text(encoding='utf-8'))
            violations = [v for turn in record.get('turns', []) for v in
                          ((turn.get('prefix_audit') or {}).get('violations') or [])]
            if violations:
                stop_reason = 'kv_prefix_audit_violation'
                append_event(events, event='integrity_violation', label=label, violations=violations[:20])
                break
        if timeouts >= rules['max_timeouts'] or crashes >= rules['max_worker_failures']:
            stop_reason = 'repeated_worker_failures'
            break
    write_json(run_dir / 'campaign-summary.json', {
        'stop_reason': stop_reason or 'schedule_completed',
        'campaign_wall_seconds': round(time.monotonic() - campaign_started, 3),
        'timeouts': timeouts, 'worker_failures': crashes,
        'swap_used_end_bytes': psutil.swap_memory().used, 'swap_used_campaign_start_bytes': swap_start,
    })
    append_event(events, event='campaign_finished', stop_reason=stop_reason or 'schedule_completed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
