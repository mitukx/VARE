"""Hidden short-window checks for gradient accumulation.

The candidate runs through the harness adapter. Pass or fail is decided here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACK = Path(__file__).resolve().parents[3]
if str(_PACK) not in sys.path:
    sys.path.insert(0, str(_PACK))

from harness_adapter import HarnessRequest, run_harness


def _reference(microbatches, accumulation_steps):
    if not microbatches or accumulation_steps <= 0:
        raise ValueError("invalid window")
    rows = len(microbatches[0])
    width = len(microbatches[0][0])
    count = len(microbatches)
    return [
        [sum(batch[row][col] for batch in microbatches) / count for col in range(width)]
        for row in range(rows)
    ]


def _generator(index: int):
    if index == 0:
        return (([[[2.0, -4.0], [6.0, 8.0]]],), {"accumulation_steps": 4})
    if index == 1:
        mixed = [
            [[1.0, 1.0, 1.0]],
            [[3.0, 5.0, 7.0]],
            [[5.0, 9.0, 13.0]],
        ]
        return ((mixed,), {"accumulation_steps": 8})
    return (([], 2), {})


def _checker(actual, expected, positional, _keywords, args, _kwargs):
    if actual != expected:
        raise AssertionError(f"short window diverged: {actual}")
    if positional[0] != args[0]:
        raise AssertionError("input gradient was mutated")


def main() -> None:
    report = run_harness(
        HarnessRequest(
            workspace=Path(sys.argv[1]).resolve(),
            entrypoint="grad_accum:average_microbatch_grads",
            task_type="correctness",
            trusted_task_path=Path(__file__).resolve().parents[1] / "task.json",
            input_generator=_generator,
            reference=_reference,
            checker=_checker,
            repeats=3,
            timeout_s=10,
        )
    )
    if not report.passed:
        raise SystemExit(report.error or "hidden checks failed")


if __name__ == "__main__":
    main()
