"""Paired mini-batch loader.

``iter_batches(features, labels, batch_size, seed, shuffle, drop_last)``
yields ``(feature_batch, label_batch)`` lists.

``features[i]`` is paired with ``labels[i]``. Every yielded item must keep
that pair. Shuffle reorders pairs; it does not reassign labels. When
``shuffle`` is true, the pair order is the permutation produced by one
``random.Random(seed)`` shuffle of the indexes. When ``shuffle`` is false,
the input order is preserved.

``batch_size`` is a positive integer. A trailing batch shorter than
``batch_size`` is omitted when ``drop_last`` is true and yielded otherwise.
Unequal input lengths or a non-positive batch size raise ``ValueError``.
"""
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
        label_order = list(range(len(labels)))
        random.Random(seed + 1).shuffle(label_order)
    else:
        label_order = order
    batches = []
    start = 0
    while start < len(order):
        stop = min(start + batch_size, len(order))
        if drop_last and stop - start < batch_size:
            break
        feature_batch = [features[index] for index in order[start:stop]]
        label_batch = [labels[index] for index in label_order[start:stop]]
        batches.append((feature_batch, label_batch))
        start = stop
    return batches
