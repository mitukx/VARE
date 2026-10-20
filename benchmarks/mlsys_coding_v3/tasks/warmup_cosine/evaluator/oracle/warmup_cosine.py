"""Reference implementation used only by harness calibration."""

from __future__ import annotations

import math


def _validate(step, total_steps, warmup_steps, base_lr, min_lr):
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("total_steps must be an integer >= 1")
    if type(warmup_steps) is not int or not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup_steps out of range")
    if type(step) is not int or not 0 <= step < total_steps:
        raise ValueError("step out of range")
    if (
        isinstance(base_lr, bool)
        or isinstance(min_lr, bool)
        or not isinstance(base_lr, (int, float))
        or not isinstance(min_lr, (int, float))
        or not math.isfinite(base_lr)
        or not math.isfinite(min_lr)
        or min_lr < 0
        or base_lr < min_lr
    ):
        raise ValueError("learning rates must be finite and ordered")


def learning_rate(step, *, total_steps, warmup_steps, base_lr, min_lr):
    _validate(step, total_steps, warmup_steps, base_lr, min_lr)
    if step < warmup_steps:
        return base_lr * (step + 1) / float(warmup_steps)
    span = total_steps - warmup_steps
    if span == 1:
        return float(min_lr)
    progress = (step - warmup_steps) / float(span - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
