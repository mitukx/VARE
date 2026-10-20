"""Hidden boundary checks for the warmup-cosine schedule.

The candidate runs through the harness adapter. Pass or fail is decided here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_PACK = Path(__file__).resolve().parents[3]
if str(_PACK) not in sys.path:
    sys.path.insert(0, str(_PACK))

from harness_adapter import HarnessRequest, run_harness


def _reference(step, *, total_steps, warmup_steps, base_lr, min_lr):
    if step < 0:
        raise ValueError("step out of range")
    if step < warmup_steps:
        return base_lr * (step + 1) / float(warmup_steps)
    span = total_steps - warmup_steps
    if span == 1:
        return float(min_lr)
    progress = (step - warmup_steps) / float(span - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _generator(index: int):
    if index < 10:
        return (
            (index,),
            {
                "total_steps": 10,
                "warmup_steps": 4,
                "base_lr": 0.2,
                "min_lr": 0.02,
            },
        )
    if index == 10:
        return (
            (4,),
            {"total_steps": 5, "warmup_steps": 4, "base_lr": 0.2, "min_lr": 0.02},
        )
    return (
        (-1,),
        {"total_steps": 4, "warmup_steps": 1, "base_lr": 0.1, "min_lr": 0.0},
    )


def _checker(actual, expected, _positional, _keywords, _args, _kwargs):
    if not math.isfinite(actual) or abs(actual - expected) > 1e-12:
        raise AssertionError(f"{actual} != {expected}")


def main() -> None:
    report = run_harness(
        HarnessRequest(
            workspace=Path(sys.argv[1]).resolve(),
            entrypoint="warmup_cosine:learning_rate",
            task_type="correctness",
            trusted_task_path=Path(__file__).resolve().parents[1] / "task.json",
            input_generator=_generator,
            reference=_reference,
            checker=_checker,
            repeats=12,
            timeout_s=10,
        )
    )
    if not report.passed:
        raise SystemExit(report.error or "hidden checks failed")


if __name__ == "__main__":
    main()
