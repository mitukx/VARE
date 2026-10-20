"""Reference implementation used only by harness calibration."""
from __future__ import annotations


def range_sums(values, queries):
    if any(type(value) is not int for value in values):
        raise ValueError("values must be ints")
    size = len(values)
    prefix = [0]
    for value in values:
        prefix.append(prefix[-1] + value)
    totals = []
    for query in queries:
        if not isinstance(query, tuple) or len(query) != 2:
            raise ValueError("query must be a (left, right) tuple")
        left, right = query
        if type(left) is not int or type(right) is not int or not 0 <= left <= right <= size:
            raise ValueError("query out of range")
        totals.append(prefix[right] - prefix[left])
    return totals
