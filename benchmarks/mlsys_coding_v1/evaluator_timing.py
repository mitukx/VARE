"""Repeated-timing helper copied into each performance task's evaluator.

The repository grader mounts one task directory. Keeping this file at the
pack root gives the protocol one canonical copy; each performance task has
a byte-identical ``evaluator/timing.py``.
"""
from __future__ import annotations

import math

REPEATS = 7
T_CRIT_95_DF6 = 2.446911851144167
THRESHOLD = 2.0


def summarize(samples: list[float]) -> dict[str, float]:
    if len(samples) != REPEATS:
        raise ValueError(f"expected {REPEATS} timing samples")
    count = len(samples)
    mean = sum(samples) / count
    variance = sum((sample - mean) ** 2 for sample in samples) / (count - 1)
    stdev = math.sqrt(variance)
    half = T_CRIT_95_DF6 * stdev / math.sqrt(count)
    return {
        "mean_s": mean,
        "stdev_s": stdev,
        "ci95_low_s": mean - half,
        "ci95_high_s": mean + half,
    }


def conservative_speedup(baseline: list[float], candidate: list[float]) -> tuple[float, dict[str, float], dict[str, float]]:
    baseline_stats = summarize(baseline)
    candidate_stats = summarize(candidate)
    low = baseline_stats["ci95_low_s"]
    high = candidate_stats["ci95_high_s"]
    if low <= 0 or high <= 0:
        raise ValueError("confidence bound must be positive")
    return low / high, baseline_stats, candidate_stats
