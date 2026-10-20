"""Operator-side hidden reference and generated cases for label smoothing.

This module is imported by the trusted adapter. It is never copied to the
candidate workspace and does not execute candidate code on the host.
"""
from __future__ import annotations

import math


def _reference(label, num_classes, epsilon):
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes")
    if type(label) is not int or not 0 <= label < num_classes:
        raise ValueError("label")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(epsilon)
        or not 0.0 <= float(epsilon) < 1.0
    ):
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
        (0, 3, 0.999),
        (0, 1, 0.7),
        (2, 7, 0.25),
    )
    return (cases[index], {})


def _checker(actual, expected, _positional, _keywords, _args, _kwargs):
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise AssertionError("class count or return type changed")
    if any(not math.isfinite(value) for value in actual):
        raise AssertionError("non-finite target")
    if any(abs(left - right) > 1e-12 for left, right in zip(actual, expected)):
        raise AssertionError(f"{actual} != {expected}")
    if abs(sum(actual) - 1.0) > 1e-12:
        raise AssertionError("distribution does not sum to 1")
