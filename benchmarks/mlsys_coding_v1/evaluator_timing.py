"""Paired-ratio timing statistic for the MLSys coding pack.

This is the only copy. Performance graders import it from the pack root.
Task directories do not carry a second ``timing.py``.
"""

from __future__ import annotations

import math

REPEATS = 7
T_CRIT_95_DF6 = 2.446911851144167
THRESHOLD = 2.0


def _samples(samples: list[float]) -> list[float]:
    if len(samples) != REPEATS:
        raise ValueError(f"expected {REPEATS} timing samples")
    if any(
        type(sample) is bool
        or not isinstance(sample, (int, float))
        or not math.isfinite(float(sample))
        or float(sample) < 0
        for sample in samples
    ):
        raise ValueError("timing samples must be finite and non-negative")
    return [float(sample) for sample in samples]


def summarize(samples: list[float]) -> dict[str, float]:
    """Mean, sample standard deviation, and a two-sided 95% t interval."""
    values = _samples(samples)
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    stdev = math.sqrt(variance)
    half = T_CRIT_95_DF6 * stdev / math.sqrt(count)
    return {
        "mean_s": mean,
        "stdev_s": stdev,
        "ci95_low_s": mean - half,
        "ci95_high_s": mean + half,
    }


def conservative_speedup(
    baseline: list[float], candidate: list[float]
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Lower 95% bound of the paired ratios, plus the two time summaries.

    ``baseline[i]`` and ``candidate[i]`` are one interleaved pair. The bound
    is not ``baseline_ci_low / candidate_ci_high``.
    """
    if len(baseline) != REPEATS or len(candidate) != REPEATS:
        raise ValueError(f"expected {REPEATS} paired timing samples")
    ratios: list[float] = []
    for left, right in zip(baseline, candidate, strict=True):
        if (
            type(left) is bool
            or type(right) is bool
            or not isinstance(left, (int, float))
            or not isinstance(right, (int, float))
            or not math.isfinite(float(left))
            or not math.isfinite(float(right))
            or float(left) < 0
            or float(right) <= 0
        ):
            raise ValueError(
                "paired times must be finite and the candidate time positive"
            )
        ratios.append(float(left) / float(right))
    ratio_stats = summarize(ratios)
    return ratio_stats["ci95_low_s"], summarize(baseline), summarize(candidate)
