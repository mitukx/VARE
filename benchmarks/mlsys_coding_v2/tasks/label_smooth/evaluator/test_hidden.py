"""Hidden checks for the per-class label-smoothing mass."""

from __future__ import annotations

import sys
from pathlib import Path

_PACK = Path(__file__).resolve().parents[4] / "mlsys_coding_v1"
if str(_PACK) not in sys.path:
    sys.path.insert(0, str(_PACK))

from harness_adapter import HarnessRequest, run_harness


def _reference(label, num_classes, epsilon):
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes")
    if type(label) is not int or not 0 <= label < num_classes:
        raise ValueError("label")
    if not 0.0 <= float(epsilon) < 1.0:
        raise ValueError("epsilon")
    share = float(epsilon) / num_classes
    target = [share] * num_classes
    target[label] = (1.0 - float(epsilon)) + share
    return target


def _generator(index: int):
    cases = (
        (1, 4, 0.2),
        (0, 2, 0.0),
        (3, 5, 0.5),
        (4, 4, 0.1),
        (0, 3, 1.0),
    )
    return (cases[index], {})


def _checker(actual, expected, _positional, _keywords, _args, _kwargs):
    if len(actual) != len(expected):
        raise AssertionError("class count changed")
    for left, right in zip(actual, expected):
        if abs(left - right) > 1e-12:
            raise AssertionError(f"{actual} != {expected}")
    if abs(sum(actual) - 1.0) > 1e-12:
        raise AssertionError("distribution does not sum to 1")


def main() -> None:
    report = run_harness(
        HarnessRequest(
            workspace=Path(sys.argv[1]).resolve(),
            entrypoint="label_smooth:smooth_targets",
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
