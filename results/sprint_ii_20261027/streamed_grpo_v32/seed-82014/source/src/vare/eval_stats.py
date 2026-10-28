"""Estimators for paired task comparisons.

These functions report uncertainty. They do not accept or reject a promotion,
and they do not read a base-gate decision.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of the chance that one of k draws is correct.

    ``n`` samples were drawn and ``c`` were correct. The estimate is
    ``1 - C(n - c, k) / C(n, k)`` when ``n - c >= k``, and 1 otherwise.
    """
    if type(n) is not int or type(c) is not int or type(k) is not int:
        raise TypeError("n, c, and k must be integers")
    if n < 1 or k < 1 or k > n or not 0 <= c <= n:
        raise ValueError("n, c, and k are outside the estimator domain")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean_unbiased_pass_at_k(correct_counts: Sequence[tuple[int, int]], k: int) -> float:
    if not correct_counts:
        raise ValueError("pass@k requires tasks")
    return sum(unbiased_pass_at_k(n, c, k) for n, c in correct_counts) / len(correct_counts)


def paired_bootstrap_interval(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Percentile interval for the paired mean of ``right - left``.

    Tasks are resampled with replacement. The interval interpolates the
    sorted bootstrap means at ``alpha / 2`` and ``1 - alpha / 2``.
    """
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equal non-empty series")
    if samples < 1 or not 0 < alpha < 1:
        raise ValueError("bootstrap settings are outside the domain")
    diffs = [float(b) - float(a) for a, b in zip(left, right, strict=True)]
    rng = random.Random(seed)
    width = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(width)] for _ in range(width)) / width
        for _ in range(samples)
    )
    observed = sum(diffs) / width
    return {
        "alpha": alpha,
        "high": _quantile(means, 1.0 - alpha / 2.0),
        "low": _quantile(means, alpha / 2.0),
        "mean": observed,
        "n": width,
        "samples": samples,
    }


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low]) * (1.0 - weight) + float(sorted_values[high]) * weight


def mcnemar_exact(b: int, c: int) -> dict[str, float | int | str | None]:
    """Two-sided exact test on discordant pairs.

    ``b`` is the count where the first arm is wrong and the second is right.
    ``c`` is the opposite count. The p-value is twice the lower binomial tail
    at probability 1/2, clamped to 1. No discordant pairs returns p = 1.
    """
    if type(b) is not int or type(c) is not int or b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative integers")
    total = b + c
    if total == 0:
        return {"b": b, "c": c, "method": "no_discordant_pairs", "p_value": 1.0, "statistic": None}
    smaller = min(b, c)
    tail = sum(math.comb(total, k) for k in range(smaller + 1))
    p_value = min(1.0, 2.0 * tail / (2**total))
    return {
        "b": b,
        "c": c,
        "method": "two_times_lower_binomial_tail",
        "p_value": p_value,
        "statistic": (b - c) ** 2 / total,
    }


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mcnemar_power(n: int, p10: float, p01: float, *, alpha: float = 0.05) -> float:
    """Normal-approximation power for a paired discordant difference.

    ``p10`` and ``p01`` are the two discordant probabilities. A zero
    difference returns ``alpha``. This is an approximation, not a simulation.
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if not 0 < alpha < 1 or p10 < 0 or p01 < 0 or p10 + p01 > 1:
        raise ValueError("probabilities are outside the domain")
    delta = p10 - p01
    if delta == 0.0:
        return alpha
    psi = p10 + p01
    z_alpha = _inverse_phi(1.0 - alpha / 2.0)
    z_beta = (abs(delta) * math.sqrt(n) - z_alpha * math.sqrt(psi)) / math.sqrt(psi - delta * delta)
    return _phi(z_beta)


def _inverse_phi(probability: float) -> float:
    if not 0 < probability < 1:
        raise ValueError("probability must be in (0, 1)")
    low = -20.0
    high = 20.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if _phi(mid) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0
