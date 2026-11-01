"""Label smoothing for a single integer class id.

``smooth_targets(label, num_classes, epsilon)`` returns a new list of
``num_classes`` finite floats.

``num_classes`` is an integer >= 1. ``label`` is an integer in
``[0, num_classes)``. ``epsilon`` is finite and ``0 <= epsilon < 1``.

Class ``label`` gets ``(1 - epsilon) + epsilon / num_classes``. Every
other class gets ``epsilon / num_classes``. The entries sum to 1.
Invalid arguments raise ``ValueError``.
"""

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
    if num_classes == 1:
        return [1.0]
    off = float(epsilon) / (num_classes - 1)
    target = [off] * num_classes
    target[label] = 1.0 - float(epsilon)
    return target
