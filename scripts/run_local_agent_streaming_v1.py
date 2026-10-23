#!/usr/bin/env python3
"""Durably retain each model response while reusing the frozen agent runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_local_agent as agent  # noqa: E402


def _append_record(path: Path, record: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("raw-turn destination is not a regular file")
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def instrument_generate_turn(
    original: Callable[..., tuple[str, int, int]], raw_turn_log: Path
) -> Callable[..., tuple[str, int, int]]:
    """Append exact raw generations before returning control to the agent loop."""
    sequence = 0

    def wrapped(*args: Any, **kwargs: Any) -> tuple[str, int, int]:
        nonlocal sequence
        started = time.monotonic()
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            sequence += 1
            _append_record(raw_turn_log, {
                "sequence": sequence,
                "event": "generation_exception",
                "exception_type": type(exc).__name__,
                "exception_text": str(exc)[:1000],
                "elapsed_seconds": round(time.monotonic() - started, 6),
            })
            raise
        raw, generated_tokens, prompt_tokens = result
        sequence += 1
        _append_record(raw_turn_log, {
            "sequence": sequence,
            "event": "generation_completed",
            "model_output": raw,
            "generated_tokens": generated_tokens,
            "prompt_tokens": prompt_tokens,
            "generation_seconds": round(time.monotonic() - started, 6),
        })
        return result

    return wrapped


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--raw-turn-log", type=Path, required=True)
    captured, forwarded = parser.parse_known_args()
    log = captured.raw_turn_log
    if log.is_symlink() or not log.is_file() or log.stat().st_size != 0:
        parser.error("raw-turn log must be a pre-created empty regular file")
    agent.generate_turn = instrument_generate_turn(agent.generate_turn, log)
    sys.argv = [sys.argv[0], *forwarded]
    return agent.main()


if __name__ == "__main__":
    raise SystemExit(main())
