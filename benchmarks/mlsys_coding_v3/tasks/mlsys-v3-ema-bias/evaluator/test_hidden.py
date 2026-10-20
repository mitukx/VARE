"""Operator-side hidden reference and generated cases for EMA correction.

This module is imported by the trusted adapter. It is never copied to the
candidate workspace and does not execute candidate code on the host.
"""
from __future__ import annotations

import math


def _reference(previous, observation, beta, step):
    values = (previous, observation, beta)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("EMA values must be finite numbers")
    if not 0.0 <= float(beta) < 1.0:
        raise ValueError("beta")
    if type(step) is not int or step < 1:
        raise ValueError("step")
    raw = float(beta) * float(previous) + (1.0 - float(beta)) * float(observation)
    return raw / (1.0 - float(beta) ** step)


def _generator(index: int):
    cases = (
        (0.0, 10.0, 0.9, 1),
        (1.0, 10.0, 0.9, 2),
        (0.0, 4.0, 0.0, 1),
        (1.0, 1.0, 0.5, 1),
        (2.5, -3.0, 0.75, 4),
    )
    return (cases[index], {})


def _checker(actual, expected, _positional, _keywords, _args, _kwargs):
    if not isinstance(actual, (int, float)) or not math.isfinite(actual):
        raise AssertionError("EMA output must be finite")
    if abs(actual - expected) > 1e-12:
        raise AssertionError(f"{actual} != {expected}")
