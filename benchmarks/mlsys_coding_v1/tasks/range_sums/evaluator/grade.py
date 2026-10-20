"""Hidden correctness plus repeated-timing grade for range sums.

The process exits 0 only when the hidden checks pass and the noise-aware
speedup against the reference scan reaches the pre-registered threshold.
Stdout is one JSON object. Candidate prints are discarded.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path

import timing

WORKLOAD_N = 5000
WORKLOAD_Q = 5000
WORKLOAD_SEED = 20261019


def _reference(values, queries):
    totals = []
    for left, right in queries:
        total = 0
        for index in range(left, right):
            total += values[index]
        totals.append(total)
    return totals


def _workload():
    generator = random.Random(WORKLOAD_SEED)
    values = [generator.randint(-5, 5) for _ in range(WORKLOAD_N)]
    queries = []
    for _ in range(WORKLOAD_Q):
        left = generator.randrange(WORKLOAD_N + 1)
        right = generator.randrange(WORKLOAD_N + 1)
        if left > right:
            left, right = right, left
        queries.append((left, right))
    return values, queries


def _check_small(candidate) -> None:
    values = [1, -2, 4, 8]
    queries = [(0, 2), (2, 2), (1, 4), (0, 4)]
    if candidate(values, queries) != [ -1, 0, 10, 11]:
        raise AssertionError("small range sums diverged")
    snapshot = values[:]
    if candidate(values, queries) != _reference(values, queries) or values != snapshot:
        raise AssertionError("result or input changed")
    try:
        candidate([1, 2], [(0, 3)])
    except ValueError:
        return
    raise AssertionError("out-of-range query must fail")


def _time_call(function, values, queries, expected, sink):
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        function(values, queries)
    samples = []
    for _ in range(timing.REPEATS):
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            started = time.perf_counter()
            actual = function(values, queries)
            elapsed = time.perf_counter() - started
        if actual != expected:
            raise AssertionError("timed result diverged from the reference scan")
        samples.append(elapsed)
    return samples


def main() -> int:
    report = {
        "measurement": "student_t_conservative_speedup",
        "correctness_pass": False,
        "improvement_pass": False,
        "workload": {"n": WORKLOAD_N, "q": WORKLOAD_Q, "seed": WORKLOAD_SEED},
    }
    try:
        workspace = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(workspace))
        from range_sums import range_sums

        _check_small(range_sums)
        values, queries = _workload()
        expected = _reference(values, queries)
        report["correctness_pass"] = True
        with open(os.devnull, "w", encoding="utf-8") as sink:
            baseline_samples = _time_call(_reference, values, queries, expected, sink)
            candidate_samples = _time_call(range_sums, values, queries, expected, sink)
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
