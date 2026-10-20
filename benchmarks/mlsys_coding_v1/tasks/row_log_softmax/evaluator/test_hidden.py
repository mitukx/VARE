"""Hidden correctness checks for row-wise log-softmax.

The candidate runs through the harness adapter. This process compares the
decoded return value with its own reference.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_PACK = Path(__file__).resolve().parents[3]
if str(_PACK) not in sys.path:
    sys.path.insert(0, str(_PACK))

from harness_adapter import HarnessRequest, run_harness


def _reference(rows):
    if not rows or any(
        not row or any(not math.isfinite(value) for value in row) for row in rows
    ):
        raise ValueError("invalid row")
    result = []
    for row in rows:
        peak = max(row)
        total = sum(math.exp(value - peak) for value in row)
        log_total = peak + math.log(total)
        result.append([value - log_total for value in row])
    return result


def _generator(index: int):
    cases = (
        ([[0.0, math.log(3.0)], [1.0, 1.0, 1.0]],),
        ([[1000.0, 1000.0, 999.0], [-1000.0, -1001.0, -1000.5]],),
        ([[]],),
        ([[math.inf]],),
    )
    return (cases[index], {})


def _checker(actual, expected, _positional, _keywords, _args, _kwargs):
    if len(actual) != len(expected):
        raise AssertionError("row count changed")
    for actual_row, expected_row in zip(actual, expected):
        if len(actual_row) != len(expected_row):
            raise AssertionError("row length changed")
        for left, right in zip(actual_row, expected_row):
            if not math.isfinite(left) or abs(left - right) > 1e-9:
                raise AssertionError(f"{actual} != {expected}")
        total = sum(math.exp(value) for value in actual_row)
        if abs(total - 1.0) > 1e-8:
            raise AssertionError(f"probabilities sum to {total}")


def main() -> None:
    report = run_harness(
        HarnessRequest(
            workspace=Path(sys.argv[1]).resolve(),
            entrypoint="row_log_softmax:row_log_softmax",
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
