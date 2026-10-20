"""Parent-owned grade for dataset standardization.

The slow per-row rescan is the paired timing baseline. Speedup is computed
from parent clocks. This process exits 0 only when that speedup reaches the
threshold.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

_PACK = Path(__file__).resolve().parents[3]
if str(_PACK) not in sys.path:
    sys.path.insert(0, str(_PACK))

import timing
from harness_adapter import HarnessRequest, run_harness

WORKLOAD_N = 2000
WORKLOAD_SEED = 20261019


def _reference(rows):
    """Slow scan used as the paired timing baseline. One pass per row."""
    if not rows:
        raise ValueError("rows must be non-empty")
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


def _generator(index: int):
    if index == 0:
        return (([1.0, 2.0, 3.0],), {})
    if index == 1:
        return (([5.0, 5.0, 5.0],), {})
    if index == 2:
        return (([],), {})
    generator = random.Random(WORKLOAD_SEED + index * 1_000_003)
    rows = [generator.uniform(-1.0, 1.0) for _ in range(WORKLOAD_N)]
    return ((rows,), {})


def _checker(actual, expected, positional, _keywords, args, _kwargs):
    _close(actual, expected)
    if positional[0] != list(args[0]):
        raise AssertionError("input changed")


def main() -> int:
    report = {
        "measurement": "student_t_conservative_speedup",
        "correctness_pass": False,
        "improvement_pass": False,
        "workload": {"n": WORKLOAD_N, "seed": WORKLOAD_SEED},
    }
    try:
        outcome = run_harness(
            HarnessRequest(
                workspace=Path(sys.argv[1]).resolve(),
                entrypoint="dataset_standardize:standardized_rows",
                task_type="performance",
                trusted_task_path=Path(__file__).resolve().parents[1] / "task.json",
                input_generator=_generator,
                reference=_reference,
                checker=_checker,
                repeats=3,
                timing_repeats=timing.REPEATS,
                timeout_s=60,
            )
        )
        report["harness_id"] = outcome.harness_id
        report["nonce_enforced"] = outcome.nonce_enforced
        report["replaced_by"] = outcome.replaced_by
        if not outcome.passed:
            report["error"] = outcome.error or "parent checks failed"
            print(json.dumps(report, sort_keys=True))
            return 1
        speedup, baseline_stats, candidate_stats = timing.conservative_speedup(
            outcome.baseline_samples_s, outcome.candidate_samples_s
        )
        report.update(
            {
                "baseline_samples_s": outcome.baseline_samples_s,
                "candidate_samples_s": outcome.candidate_samples_s,
                "baseline": baseline_stats,
                "candidate": candidate_stats,
                "conservative_speedup": speedup,
                "correctness_pass": True,
                "threshold": timing.THRESHOLD,
                "improvement_pass": speedup >= timing.THRESHOLD,
            }
        )
    except Exception as exc:  # noqa: BLE001 - a failed grade still emits JSON
        report["error"] = f"{type(exc).__name__}: {exc}"[:500]
        print(json.dumps(report, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0 if report["improvement_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
