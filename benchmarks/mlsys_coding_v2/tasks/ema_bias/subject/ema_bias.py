"""Bias-corrected exponential moving average.

``bias_corrected_ema(previous, observation, beta, step)`` takes the
previous corrected-or-raw state, one new finite observation, a decay
``beta``, and ``step``.

``step`` is the 1-based count of observations included so far.
``beta`` is finite and ``0 <= beta < 1``. ``previous`` and
``observation`` are finite and are not booleans.

Let ``raw = beta * previous + (1 - beta) * observation``. The return
value is ``raw / (1 - beta ** step)``. Invalid arguments raise
``ValueError``.
"""

from __future__ import annotations

import math


def _finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")


def bias_corrected_ema(previous, observation, beta, step):
    _finite(previous, "previous")
    _finite(observation, "observation")
    _finite(beta, "beta")
    if not 0.0 <= float(beta) < 1.0:
        raise ValueError("beta must be in [0, 1)")
    if type(step) is not int or step < 1:
        raise ValueError("step must be an integer >= 1")
    return float(beta) * float(previous) + (1.0 - float(beta)) * float(observation)
