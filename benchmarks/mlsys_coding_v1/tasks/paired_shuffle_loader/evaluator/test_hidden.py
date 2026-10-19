"""Hidden pair-integrity checks for the mini-batch loader."""
from __future__ import annotations

import random
import sys
from pathlib import Path


def _expected(features, labels, batch_size, seed, shuffle, drop_last):
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
        batches.append(([features[index] for index in indexes], [labels[index] for index in indexes]))
        start = stop
    return batches


def main() -> None:
    workspace = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(workspace))
    from paired_loader import iter_batches

    features = [f"f{i}" for i in range(10)]
    labels = [100 + i for i in range(10)]
    cases = [
        (3, 7, True, False),
        (4, 11, True, True),
        (10, 3, True, False),
        (1, 99, True, False),
        (6, 0, False, True),
    ]
    for batch_size, seed, shuffle, drop_last in cases:
        actual = iter_batches(features, labels, batch_size, seed, shuffle, drop_last)
        expected = _expected(features, labels, batch_size, seed, shuffle, drop_last)
        if actual != expected:
            raise AssertionError(f"case {(batch_size, seed, shuffle, drop_last)} diverged")
        for feature_batch, label_batch in actual:
            for feature, label in zip(feature_batch, label_batch):
                index = features.index(feature)
                if label != labels[index]:
                    raise AssertionError("label was detached from its feature")

    try:
        iter_batches([1], [1, 2], 1, 0, False, False)
    except ValueError:
        pass
    else:
        raise AssertionError("unequal lengths must fail")


if __name__ == "__main__":
    main()
