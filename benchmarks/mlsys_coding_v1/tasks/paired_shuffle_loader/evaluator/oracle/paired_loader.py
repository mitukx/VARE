"""Reference implementation used only by harness calibration."""
from __future__ import annotations

import random


def iter_batches(features, labels, batch_size, seed, shuffle, drop_last):
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
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
            ([features[index] for index in indexes], [labels[index] for index in indexes])
        )
        start = stop
    return batches
