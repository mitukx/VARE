"""Protocol reward for the frozen MLSys coding pack.

The grader's ``passed`` bit is the pre-registered success gate: hidden
correctness, and, for performance tasks, a noise-aware speedup threshold.
This module recomputes that speedup from the grade command's retained
stdout samples. It does not trust a boolean printed by the grade script.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

_TIMING_PATH = Path(__file__).resolve().with_name("evaluator_timing.py")
_TIMING_SPEC = importlib.util.spec_from_file_location(
    "mlsys_coding_v1_evaluator_timing", _TIMING_PATH
)
if _TIMING_SPEC is None or _TIMING_SPEC.loader is None:
    raise ImportError(_TIMING_PATH)
_timing = importlib.util.module_from_spec(_TIMING_SPEC)
_TIMING_SPEC.loader.exec_module(_timing)

REPEATS = _timing.REPEATS
# Student-t two-sided 95% critical value at df = REPEATS - 1 = 6.
T_CRIT_95_DF6 = _timing.T_CRIT_95_DF6
SUCCESS_THRESHOLD = _timing.THRESHOLD
# Shaped reward reaches 1 at twice the success threshold.
REWARD_SATURATION = 4.0
MEASUREMENT = "paired_ratio_student_t_lower"


def summarize(samples: list[float]) -> dict[str, float]:
    """Mean, sample standard deviation, and a two-sided 95% t interval."""
    return _timing.summarize(samples)


def conservative_speedup(
    baseline_samples: list[float], candidate_samples: list[float]
) -> float:
    """Lower 95% bound of the paired ratios ``baseline / candidate``.

    The ratio math lives in ``evaluator_timing.py``. Sample ``i`` is one
    interleaved pair. A non-positive lower bound is returned as that bound.
    """
    low, _baseline, _candidate = _timing.conservative_speedup(
        baseline_samples, candidate_samples
    )
    return low


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


def decide(result: dict[str, Any], *, task_type: str) -> dict[str, Any]:
    """Map one ``EnvironmentResult.to_dict()`` record to the protocol reward.

    ``task_type`` comes from trusted task metadata. It is not inferred from
    the grade stdout. A correctness task scores 1 only when the grade command
    passed. A performance task scores ``min(1, speedup / 4)`` only when the
    parent measurement JSON parses as one object and hidden correctness
    passed. Prefixed, missing, or malformed measurement output scores 0.
    The speedup is recomputed from the paired samples. A printed speedup
    field is ignored. Success still requires the grade command to have
    exited 0, which the grade script does only at or above the threshold.
    """
    decision = {
        "success": False,
        "reward": 0.0,
        "conservative_speedup": None,
        "baseline_stdev_s": None,
        "candidate_stdev_s": None,
        "reason": "unclassified",
    }
    if task_type not in {"correctness", "performance"}:
        decision["reason"] = "task_type"
        return decision
    if result.get("integrity_ok") is not True:
        decision["reason"] = "integrity"
        return decision
    tests = result.get("tests")
    if not isinstance(tests, list) or len(tests) != 1:
        decision["reason"] = "expected_one_grade_command"
        return decision
    test = tests[0]
    command_passed = _command_passed(test)
    if task_type == "correctness":
        if (
            command_passed
            and result.get("passed") is True
            and result.get("correctness") == 1.0
        ):
            decision["success"] = True
            decision["reward"] = 1.0
        decision["reason"] = "correctness"
        return decision
    try:
        measurement = _measurement(str(test.get("stdout", "")))
    except ValueError:
        measurement = None
    if measurement is None:
        decision["reason"] = "malformed_measurement"
        return decision
    if measurement.get("correctness_pass") is not True:
        decision["reason"] = "correctness"
        return decision
    try:
        speedup = conservative_speedup(
            list(measurement["baseline_samples_s"]),
            list(measurement["candidate_samples_s"]),
        )
    except (KeyError, TypeError, ValueError):
        decision["reason"] = "malformed_measurement"
        return decision
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
