"""Hidden correctness plus repeated-timing grade for dataset standardization."""
from __future__ import annotations

import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import timing

WORKLOAD_N = 2000
WORKLOAD_SEED = 20261019


def _reference(rows):
    """Slow scan used as the paired timing baseline. One pass per row."""
    count = len(rows)

    def mean_and_variance():
        mean = sum(rows) / count
        variance = sum((value - mean) ** 2 for value in rows) / count
        return mean, variance

    standardized = []
    for value in rows:
        mean, variance = mean_and_variance()
        if variance == 0.0:
            standardized.append(0.0)
        else:
            standardized.append((value - mean) / math.sqrt(variance))
    return standardized


def _close(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("length changed")
    for left, right in zip(actual, expected):
        if not math.isfinite(left) or abs(left - right) > 1e-9:
            raise AssertionError("standardized values diverged")


def _check_small(candidate) -> None:
    _close(candidate([1.0, 2.0, 3.0]), _reference([1.0, 2.0, 3.0]))
    if candidate([5.0, 5.0, 5.0]) != [0.0, 0.0, 0.0]:
        raise AssertionError("zero variance must return zeros")
    try:
        candidate([])
    except ValueError:
        return
    raise AssertionError("empty input must fail")


def _time_call(function, rows, expected, sink):
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        function(rows)
    samples = []
    for _ in range(timing.REPEATS):
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            started = time.perf_counter()
            actual = function(rows)
            elapsed = time.perf_counter() - started
        _close(actual, expected)
        samples.append(elapsed)
    return samples


def main() -> int:
    report = {
        "measurement": "student_t_conservative_speedup",
        "correctness_pass": False,
        "improvement_pass": False,
        "workload": {"n": WORKLOAD_N, "seed": WORKLOAD_SEED},
    }
    try:
        workspace = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(workspace))
        from dataset_standardize import standardized_rows

        _check_small(standardized_rows)
        generator = random.Random(WORKLOAD_SEED)
        rows = [generator.uniform(-1.0, 1.0) for _ in range(WORKLOAD_N)]
        expected = _reference(rows)
        report["correctness_pass"] = True
        with open(os.devnull, "w", encoding="utf-8") as sink:
            baseline_samples = _time_call(_reference, rows, expected, sink)
            candidate_samples = _time_call(standardized_rows, rows, expected, sink)
        speedup, baseline_stats, candidate_stats = timing.conservative_speedup(
            baseline_samples, candidate_samples,
        )
        report.update({
            "baseline_samples_s": baseline_samples,
            "candidate_samples_s": candidate_samples,
            "baseline": baseline_stats,
            "candidate": candidate_stats,
            "conservative_speedup": speedup,
            "threshold": timing.THRESHOLD,
            "improvement_pass": speedup >= timing.THRESHOLD,
        })
    except Exception as exc:  # noqa: BLE001 - candidate failures must still emit JSON
        report["error"] = f"{type(exc).__name__}: {exc}"[:500]
        print(json.dumps(report, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0 if report["improvement_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
