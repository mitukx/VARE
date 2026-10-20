"""Hidden pair-integrity checks for the mini-batch loader.

The candidate runs through the harness adapter. Pass or fail is decided here.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_PACK = Path(__file__).resolve().parents[3]
if str(_PACK) not in sys.path:
    sys.path.insert(0, str(_PACK))

from harness_adapter import HarnessRequest, run_harness


def _reference(features, labels, batch_size, seed, shuffle, drop_last):
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    order = list(range(len(features)))
    if shuffle:
        random.Random(seed).shuffle(order)
    batches = []
    start = 0
    while start < len(order):
        stop = min(start + batch_size, len(order))
        if drop_last and stop - start < batch_size:
            break
        indexes = order[start:stop]
        batches.append(
            (
                [features[index] for index in indexes],
                [labels[index] for index in indexes],
            )
        )
        start = stop
    return batches


def _generator(index: int):
    features = [f"f{i}" for i in range(10)]
    labels = [100 + i for i in range(10)]
    if index == 5:
        return (([1], [1, 2], 1, 0, False, False), {})
    batch_size, seed, shuffle, drop_last = (
        (3, 7, True, False),
        (4, 11, True, True),
        (10, 3, True, False),
        (1, 99, True, False),
        (6, 0, False, True),
    )[index]
    return ((features, labels, batch_size, seed, shuffle, drop_last), {})


def _checker(actual, expected, _positional, _keywords, args, _kwargs):
    if actual != expected:
        raise AssertionError(f"batches diverged: {actual}")
    features, labels = args[0], args[1]
    for feature_batch, label_batch in actual:
        for feature, label in zip(feature_batch, label_batch):
            index = features.index(feature)
            if label != labels[index]:
                raise AssertionError("label was detached from its feature")


def main() -> None:
    report = run_harness(
        HarnessRequest(
            workspace=Path(sys.argv[1]).resolve(),
            entrypoint="paired_loader:iter_batches",
            task_type="correctness",
            trusted_task_path=Path(__file__).resolve().parents[1] / "task.json",
            input_generator=_generator,
            reference=_reference,
            checker=_checker,
            repeats=6,
            timeout_s=10,
        )
    )
    if not report.passed:
        raise SystemExit(report.error or "hidden checks failed")


if __name__ == "__main__":
    main()
