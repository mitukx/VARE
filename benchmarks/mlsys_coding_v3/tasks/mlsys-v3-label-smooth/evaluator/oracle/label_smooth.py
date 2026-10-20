"""Reference implementation used only by harness calibration."""

from __future__ import annotations

import math


def _validate(label, num_classes, epsilon):
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be an integer >= 1")
    if type(label) is not int or not 0 <= label < num_classes:
        raise ValueError("label out of range")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(epsilon)
        or not 0.0 <= float(epsilon) < 1.0
    ):
        raise ValueError("epsilon must be in [0, 1)")


def smooth_targets(label, num_classes, epsilon):
    _validate(label, num_classes, epsilon)
    share = float(epsilon) / num_classes
    target = [share] * num_classes
    target[label] = (1.0 - float(epsilon)) + share
    return target
