"""Hidden boundary checks for the warmup-cosine schedule."""
from __future__ import annotations

import math
import sys
from pathlib import Path


def _expected(step, total_steps, warmup_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / float(warmup_steps)
    span = total_steps - warmup_steps
    if span == 1:
        return float(min_lr)
    progress = (step - warmup_steps) / float(span - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    workspace = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(workspace))
    from warmup_cosine import learning_rate

    total_steps = 10
    warmup_steps = 4
    base_lr = 0.2
    min_lr = 0.02
    for step in range(total_steps):
        actual = learning_rate(
            step, total_steps=total_steps, warmup_steps=warmup_steps, base_lr=base_lr, min_lr=min_lr,
        )
        expected = _expected(step, total_steps, warmup_steps, base_lr, min_lr)
        if not math.isfinite(actual) or abs(actual - expected) > 1e-12:
            raise AssertionError(f"step {step}: {actual} != {expected}")
    only = learning_rate(4, total_steps=5, warmup_steps=4, base_lr=base_lr, min_lr=min_lr)
    if abs(only - min_lr) > 1e-12:
        raise AssertionError("single cosine step must be min_lr")
    try:
        learning_rate(-1, total_steps=4, warmup_steps=1, base_lr=0.1, min_lr=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative step must fail")


if __name__ == "__main__":
    main()
