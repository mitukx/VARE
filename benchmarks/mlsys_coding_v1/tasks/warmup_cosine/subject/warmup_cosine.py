"""Warmup-cosine learning-rate schedule.

``learning_rate(step, *, total_steps, warmup_steps, base_lr, min_lr)``

Preconditions, each raising ``ValueError`` when violated:

- ``total_steps`` is an integer >= 1
- ``warmup_steps`` is an integer and ``0 <= warmup_steps <= total_steps``
- ``step`` is an integer and ``0 <= step < total_steps``
- ``base_lr`` and ``min_lr`` are finite and ``0 <= min_lr <= base_lr``

Value:

- If ``step < warmup_steps``, return ``base_lr * (step + 1) / warmup_steps``.
  Step ``warmup_steps - 1`` therefore returns ``base_lr``.
- Otherwise the cosine region contains ``span = total_steps - warmup_steps``
  steps, starting at ``step == warmup_steps``.
  If ``span == 1``, return ``min_lr``.
  If ``span > 1``, let ``progress = (step - warmup_steps) / (span - 1)``
  and return
  ``min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))``.
  Progress is 0 on the first cosine step and 1 on step ``total_steps - 1``.
"""

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
        return base_lr * step / float(warmup_steps)
    progress = step / float(total_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
