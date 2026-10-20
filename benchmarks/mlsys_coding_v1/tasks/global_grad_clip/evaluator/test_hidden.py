"""Hidden L2 checks for global-norm clipping.

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


def _reference(grads, max_norm):
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    values = [float(value) for row in grads for value in row]
    norm = math.sqrt(sum(value * value for value in values))
    scale = 1.0 if norm == 0.0 or norm <= max_norm else max_norm / norm
    clipped = []
    cursor = 0
    for row in grads:
        clipped.append([values[cursor + offset] * scale for offset in range(len(row))])
        cursor += len(row)
    return clipped


def _generator(index: int):
    if index == 0:
        return (([[3.0, 4.0], [0.0, 12.0]], 2.5), {})
    if index == 1:
        return (([[3.0, 4.0]], 5.0), {})
    if index == 2:
        return (([[0.0, 0.0]], 1.0), {})
    return (([[1.0, 2.0]], 0.0), {})


def _checker(actual, expected, positional, _keywords, args, _kwargs):
    for actual_row, expected_row in zip(actual, expected):
        for left, right in zip(actual_row, expected_row):
            if abs(left - right) > 1e-12:
                raise AssertionError(f"{actual} != {expected}")
    if positional[0] != args[0]:
        raise AssertionError("input gradient was mutated")


def main() -> None:
    report = run_harness(
        HarnessRequest(
            workspace=Path(sys.argv[1]).resolve(),
            entrypoint="global_grad_clip:clip_by_global_norm",
            task_type="correctness",
            trusted_task_path=Path(__file__).resolve().parents[1] / "task.json",
            input_generator=_generator,
            reference=_reference,
            checker=_checker,
            repeats=4,
            timeout_s=10,
        )
    )
    if not report.passed:
        raise SystemExit(report.error or "hidden checks failed")


if __name__ == "__main__":
    main()
