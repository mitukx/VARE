"""Hidden causal-mask checks.

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


def _softmax(values):
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    normalizer = sum(exponentials)
    return [value / normalizer for value in exponentials]


def _reference(scores):
    if not scores or any(len(row) != len(scores) for row in scores):
        raise ValueError("scores must be square")
    weights = []
    for row_index, row in enumerate(scores):
        allowed = _softmax(row[: row_index + 1])
        weights.append(allowed + [0.0] * (len(row) - row_index - 1))
    return weights


def _generator(index: int):
    cases = (
        ([[0.0]],),
        ([[1000.0]],),
        ([[0.0, 5.0], [0.0, 40.0]],),
        ([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-2.0, 1.0, 4.0]],),
        ([[1.0, 2.0]],),
    )
    return (cases[index], {})


def _checker(actual, expected, _positional, _keywords, _args, _kwargs):
    if len(actual) != len(expected):
        raise AssertionError("shape changed")
    for actual_row, expected_row in zip(actual, expected):
        if len(actual_row) != len(expected_row):
            raise AssertionError("shape changed")
        for left, right in zip(actual_row, expected_row):
            if not math.isfinite(left) or abs(left - right) > 1e-9:
                raise AssertionError(f"{actual} != {expected}")
        if abs(sum(actual_row) - 1.0) > 1e-8:
            raise AssertionError(f"row does not sum to 1: {actual_row}")


def main() -> None:
    report = run_harness(
        HarnessRequest(
            workspace=Path(sys.argv[1]).resolve(),
            entrypoint="causal_attention:causal_attention_weights",
            task_type="correctness",
            trusted_task_path=Path(__file__).resolve().parents[1] / "task.json",
            input_generator=_generator,
            reference=_reference,
            checker=_checker,
            repeats=5,
            timeout_s=10,
        )
    )
    if not report.passed:
        raise SystemExit(report.error or "hidden checks failed")


if __name__ == "__main__":
    main()
