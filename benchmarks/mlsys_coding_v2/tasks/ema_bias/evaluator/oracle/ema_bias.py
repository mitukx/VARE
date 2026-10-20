"""Reference implementation used only by harness calibration."""

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
    raw = float(beta) * float(previous) + (1.0 - float(beta)) * float(observation)
    return raw / (1.0 - float(beta) ** step)
