"""Protocol reward for the frozen MLSys coding pack.

The grader's ``passed`` bit is the pre-registered success gate: hidden
correctness, and, for performance tasks, a noise-aware speedup threshold.
This module recomputes that speedup from the grade command's retained
stdout samples. It does not trust a boolean printed by the grade script.
"""
from __future__ import annotations

import json
import math
from typing import Any

REPEATS = 7
# Student-t two-sided 95% critical value at df = REPEATS - 1 = 6.
T_CRIT_95_DF6 = 2.446911851144167
SUCCESS_THRESHOLD = 2.0
# Shaped reward reaches 1 at twice the success threshold.
REWARD_SATURATION = 4.0
MEASUREMENT = "student_t_conservative_speedup"


def summarize(samples: list[float]) -> dict[str, float]:
    """Mean, sample standard deviation, and a two-sided 95% t interval."""
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
    values = [float(sample) for sample in samples]
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


def conservative_speedup(baseline_samples: list[float], candidate_samples: list[float]) -> float:
    """Baseline lower 95% bound divided by the candidate upper 95% bound.

    A value above 1 means the candidate's pessimistic time beats the
    reference implementation's optimistic time. The success threshold is
    applied by the caller.
    """
    baseline = summarize(baseline_samples)
    candidate = summarize(candidate_samples)
    if baseline["ci95_low_s"] <= 0 or candidate["ci95_high_s"] <= 0:
        raise ValueError("confidence bound must be positive")
    return baseline["ci95_low_s"] / candidate["ci95_high_s"]


def shaped_reward(speedup: float) -> float:
    if not math.isfinite(speedup) or speedup <= 0:
        return 0.0
    return min(1.0, speedup / REWARD_SATURATION)


def _command_passed(test: dict[str, Any]) -> bool:
    """Command success as stored on ``EnvironmentResult.to_dict()``.

    ``CommandResult.passed`` is a property and is not part of the serialized
    result. The retained fields are the return code and the limit flags.
    """
    return (
        test.get("returncode") == 0
        and test.get("timed_out") is False
        and test.get("output_limited") is False
    )


def _measurement(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("grade stdout is not JSON") from exc
    if not isinstance(payload, dict) or payload.get("measurement") != MEASUREMENT:
        return None
    return payload


def decide(result: dict[str, Any]) -> dict[str, Any]:
    """Map one ``EnvironmentResult.to_dict()`` record to the protocol reward.

    Correctness tasks with a passing grade and no timing record score 1.
    Performance tasks score ``min(1, speedup / 4)`` when hidden correctness
    passed and the samples recompute cleanly, including when the speedup
    stays below the success threshold. Success still requires the grade
    command to have exited 0, which the grade script does only at or above
    the threshold.
    """
    decision = {
        "success": False,
        "reward": 0.0,
        "conservative_speedup": None,
        "baseline_stdev_s": None,
        "candidate_stdev_s": None,
        "reason": "unclassified",
    }
    if result.get("integrity_ok") is not True:
        decision["reason"] = "integrity"
        return decision
    tests = result.get("tests")
    if not isinstance(tests, list) or len(tests) != 1:
        decision["reason"] = "expected_one_grade_command"
        return decision
    test = tests[0]
    command_passed = _command_passed(test)
    measurement = _measurement(str(test.get("stdout", "")))
    if measurement is None:
        if command_passed and result.get("passed") is True and result.get("correctness") == 1.0:
            decision["success"] = True
            decision["reward"] = 1.0
            decision["reason"] = "correctness"
            return decision
        decision["reason"] = "correctness"
        return decision
    if measurement.get("correctness_pass") is not True:
        decision["reason"] = "correctness"
        return decision
    speedup = conservative_speedup(
        list(measurement["baseline_samples_s"]),
        list(measurement["candidate_samples_s"]),
    )
    baseline_stats = summarize(list(measurement["baseline_samples_s"]))
    candidate_stats = summarize(list(measurement["candidate_samples_s"]))
    decision["conservative_speedup"] = speedup
    decision["baseline_stdev_s"] = baseline_stats["stdev_s"]
    decision["candidate_stdev_s"] = candidate_stats["stdev_s"]
    decision["reward"] = shaped_reward(speedup)
    if speedup >= SUCCESS_THRESHOLD and command_passed and result.get("passed") is True:
        decision["success"] = True
        decision["reason"] = "performance"
        return decision
    if speedup >= SUCCESS_THRESHOLD:
        decision["reward"] = 0.0
        decision["reason"] = "threshold_met_but_grade_failed"
        return decision
    decision["reason"] = "performance_below_threshold"
    return decision
