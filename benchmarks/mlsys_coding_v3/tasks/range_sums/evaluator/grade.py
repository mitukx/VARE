"""Parent-owned grade for range sums.

Samples and the speedup decision come from the harness adapter. This process
prints one JSON object. It exits 0 only when the parent-computed speedup
reaches the threshold. The child's exit code is not that decision.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_PACK = Path(__file__).resolve().parents[3]
if str(_PACK) not in sys.path:
    sys.path.insert(0, str(_PACK))

import evaluator_timing as timing
from harness_adapter import HarnessRequest, run_harness

WORKLOAD_N = 5000
WORKLOAD_Q = 5000
WORKLOAD_SEED = 20261019


def _reference(values, queries):
    totals = []
    for left, right in queries:
        if (
            type(left) is not int
            or type(right) is not int
            or not 0 <= left <= right <= len(values)
        ):
            raise ValueError("query out of range")
        total = 0
        for index in range(left, right):
            total += values[index]
        totals.append(total)
    return totals


def _workload(draw: int):
    """New values on every draw. A cache of one input cannot cover the repeats."""
    generator = random.Random(WORKLOAD_SEED + draw * 1_000_003)
    values = [generator.randint(-5, 5) for _ in range(WORKLOAD_N)]
    queries = []
    for _ in range(WORKLOAD_Q):
        left = generator.randrange(WORKLOAD_N + 1)
        right = generator.randrange(WORKLOAD_N + 1)
        if left > right:
            left, right = right, left
        queries.append((left, right))
    return values, queries


def _generator(index: int):
    if index == 0:
        values = [1, -2, 4, 8]
        queries = [(0, 2), (2, 2), (1, 4), (0, 4)]
        return ((values, queries), {})
    if index == 1:
        return (([1, 2], [(0, 3)]), {})
    return (_workload(index), {})


def _checker(actual, expected, positional, _keywords, args, _kwargs):
    if actual != expected:
        raise AssertionError("range sums diverged")
    if positional[0] != list(args[0]):
        raise AssertionError("input changed")


def main() -> int:
    report = {
        "measurement": "paired_ratio_student_t_lower",
        "correctness_pass": False,
        "improvement_pass": False,
        "workload": {"n": WORKLOAD_N, "q": WORKLOAD_Q, "seed": WORKLOAD_SEED},
    }
    try:
        outcome = run_harness(
            HarnessRequest(
                workspace=Path(sys.argv[1]).resolve(),
                entrypoint="range_sums:range_sums",
                task_type="performance",
                trusted_task_path=Path(__file__).resolve().parents[1] / "task.json",
                input_generator=_generator,
                reference=_reference,
                checker=_checker,
                repeats=2,
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
