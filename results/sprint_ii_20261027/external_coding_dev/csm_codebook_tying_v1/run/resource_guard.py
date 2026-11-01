#!/usr/bin/env python3
"""Run one agent process under a frozen macOS swap/RSS/wall budget."""
import argparse
import json
import os
import pathlib
import re
import signal
import subprocess
import time

import psutil

MiB = 1024 * 1024


def _sysctl_value(label):
    text = subprocess.run(['sysctl', 'vm.swapusage'], capture_output=True, text=True, check=True).stdout
    match = re.search(rf'{label}\s*=\s*([0-9.]+)M', text)
    if not match:
        raise RuntimeError(f'could not parse vm.swapusage {label}: {text}')
    return float(match.group(1))


def swap_used_mib():
    return _sysctl_value('used')


def swap_free_mib():
    return _sysctl_value('free')


def pressure_free_pct():
    text = subprocess.run(['memory_pressure'], capture_output=True, text=True, check=True).stdout
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', text)
    if not match:
        raise RuntimeError('could not parse system memory pressure')
    return int(match.group(1))


def tree_rss_mib(pid):
    import psutil
    try:
        proc = psutil.Process(pid)
        items = [proc, *proc.children(recursive=True)]
        return sum(p.memory_info().rss for p in items if p.is_running()) / MiB
    except psutil.NoSuchProcess:
        return 0.0


def terminate_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=pathlib.Path, required=True)
    parser.add_argument('--swap-growth-cap-mib', type=float, default=768.0)
    parser.add_argument('--swap-free-preflight-mib', type=float, default=1024.0)
    parser.add_argument('--memory-free-preflight-percent', type=int, default=65)
    parser.add_argument('--rss-cap-mib', type=float, default=12288.0)
    parser.add_argument('--wall-cap-seconds', type=float, default=1800.0)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('command is required after --')
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / 'resource-summary.json'
    if summary_path.exists():
        parser.error(f'refusing to overwrite {summary_path}')
    baseline = swap_used_mib()
    free_swap = swap_free_mib()
    free_memory = pressure_free_pct()
    start_wall = time.time()
    start_mono = time.monotonic()
    summary = {
        'command': command, 'started_unix': start_wall,
        'preflight': {'swap_used_mib': baseline, 'swap_free_mib': free_swap,
                      'memory_pressure_free_percent': free_memory},
        'swap_growth_cap_mib': args.swap_growth_cap_mib,
        'swap_abort_threshold_mib': baseline + args.swap_growth_cap_mib,
        'rss_cap_mib': args.rss_cap_mib, 'wall_cap_seconds': args.wall_cap_seconds,
        'resource_abort': None, 'returncode': None, 'elapsed_seconds': None,
        'sample_count': 0, 'max_sampled_process_tree_rss_mib': 0.0,
        'swap_used_final_mib': baseline, 'swap_growth_final_mib': 0.0,
        'model_process_started': False,
    }
    if free_swap < args.swap_free_preflight_mib or free_memory < args.memory_free_preflight_percent:
        summary['resource_abort'] = 'preflight_below_frozen_minimum'
        summary['elapsed_seconds'] = 0.0
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
        print(json.dumps(summary, sort_keys=True))
        return 0

    env = os.environ.copy()
    env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'DOCKER_CONTEXT': 'colima'})
    stdout_path, stderr_path = run_dir / 'stdout.log', run_dir / 'stderr.log'
    samples_path = run_dir / 'resource-samples.jsonl'
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr, samples_path.open('xb') as sample_file:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        summary['model_process_started'] = True
        try:
            while True:
                elapsed = time.monotonic() - start_mono
                used = swap_used_mib()
                rss = tree_rss_mib(proc.pid)
                sample = {'elapsed_seconds': round(elapsed, 3), 'swap_used_mib': used,
                          'swap_growth_mib': round(used - baseline, 3),
                          'process_tree_rss_mib': round(rss, 3)}
                sample_file.write((json.dumps(sample, sort_keys=True) + '\n').encode())
                sample_file.flush()
                os.fsync(sample_file.fileno())
                summary['sample_count'] += 1
                summary['max_sampled_process_tree_rss_mib'] = max(summary['max_sampled_process_tree_rss_mib'], rss)
                summary['swap_used_final_mib'] = used
                summary['swap_growth_final_mib'] = round(used - baseline, 3)
                if used > baseline + args.swap_growth_cap_mib:
                    summary['resource_abort'] = 'host_swap_growth_cap'
                    terminate_group(proc)
                    break
                if rss > args.rss_cap_mib:
                    summary['resource_abort'] = 'sampled_process_tree_rss_cap'
                    terminate_group(proc)
                    break
                if elapsed > args.wall_cap_seconds:
                    summary['resource_abort'] = 'external_wall_cap'
                    terminate_group(proc)
                    break
                rc = proc.poll()
                if rc is not None:
                    summary['returncode'] = rc
                    break
                time.sleep(1.0)
        finally:
            if proc.poll() is None:
                terminate_group(proc)
            if summary['returncode'] is None:
                summary['returncode'] = proc.returncode
            summary['elapsed_seconds'] = round(time.monotonic() - start_mono, 3)
            summary['swap_used_final_mib'] = swap_used_mib()
            summary['swap_growth_final_mib'] = round(summary['swap_used_final_mib'] - baseline, 3)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary['returncode'] == 0 and summary['resource_abort'] is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
