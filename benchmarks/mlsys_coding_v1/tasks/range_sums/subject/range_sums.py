"""Exact integer range sums.

``range_sums(values, queries)`` takes a list of ints and a list of
``(left, right)`` half-open intervals. It returns one int per query,
equal to ``sum(values[left:right])``, which is 0 when ``left == right``.

Every query must satisfy ``0 <= left <= right <= len(values)``. A query
outside that range, or a non-integer value, raises ``ValueError``.
``values`` and ``queries`` are not mutated.
"""

from __future__ import annotations


def range_sums(values, queries):
    if any(type(value) is not int for value in values):
        raise ValueError("values must be ints")
    size = len(values)
    totals = []
    for query in queries:
        if not isinstance(query, tuple) or len(query) != 2:
            raise ValueError("query must be a (left, right) tuple")
        left, right = query
        if (
            type(left) is not int
            or type(right) is not int
            or not 0 <= left <= right <= size
        ):
            raise ValueError("query out of range")
        total = 0
        for index in range(left, right):
            total += values[index]
        totals.append(total)
    return totals
