#!/usr/bin/env python3
"""Apply per-process MLX memory limits, then run the pinned local-agent CLI."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory-limit-bytes', type=int, required=True)
    parser.add_argument('--wired-limit-bytes', type=int, required=True)
    parser.add_argument('--agent-runner', type=Path, required=True)
    parser.add_argument('agent_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    agent_args = args.agent_args[1:] if args.agent_args[:1] == ['--'] else args.agent_args
    if not agent_args:
        parser.error('agent arguments after -- are required')
    if args.memory_limit_bytes <= 0 or args.wired_limit_bytes <= 0:
        parser.error('MLX memory limits must be positive')
    runner = args.agent_runner.resolve(strict=True)
    import mlx.core as mx

    previous_memory_limit = mx.set_memory_limit(args.memory_limit_bytes)
    previous_wired_limit = mx.set_wired_limit(args.wired_limit_bytes)
    if mx.get_memory_limit() != args.memory_limit_bytes:
        raise RuntimeError('MLX memory limit was not applied')
    print(
        f'MLX limits applied: graph={mx.get_memory_limit()} bytes; '
        f'wired={args.wired_limit_bytes} bytes; '
        f'previous_graph={previous_memory_limit}; previous_wired={previous_wired_limit}',
        flush=True,
    )
    sys.argv = [str(runner), *agent_args]
    runpy.run_path(str(runner), run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
