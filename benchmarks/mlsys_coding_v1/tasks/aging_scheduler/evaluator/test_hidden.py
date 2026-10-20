"""Hidden aging and tie-break checks for the scheduler.

The candidate runs through the harness adapter. Pass or fail is decided here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACK = Path(__file__).resolve().parents[3]
if str(_PACK) not in sys.path:
    sys.path.insert(0, str(_PACK))

from harness_adapter import HarnessRequest, run_harness


def _reference(jobs):
    if any(job["release"] < 0 for job in jobs):
        raise ValueError("release out of range")
    remaining = [dict(job) for job in jobs]
    clock = 0
    result = []
    while remaining:
        ready = [job for job in remaining if job["release"] <= clock]
        if not ready:
            clock = min(job["release"] for job in remaining)
            continue
        chosen = min(
            ready,
            key=lambda job: (-(job["priority"] + (clock - job["release"])), job["id"]),
        )
        remaining.remove(chosen)
        result.append(
            {"id": chosen["id"], "start": clock, "finish": clock + chosen["service"]}
        )
        clock += chosen["service"]
    return result


def _generator(index: int):
    cases = (
        (
            [
                {"id": "z", "release": 0, "priority": 100, "service": 10},
                {"id": "a", "release": 0, "priority": 1, "service": 1},
                {"id": "b", "release": 10, "priority": 6, "service": 1},
            ],
        ),
        (
            [
                {"id": "b", "release": 0, "priority": 3, "service": 2},
                {"id": "a", "release": 0, "priority": 3, "service": 2},
            ],
        ),
        ([{"id": "early", "release": 4, "priority": 1, "service": 3}],),
        ([{"id": "a", "release": -1, "priority": 1, "service": 1}],),
    )
    return (cases[index], {})


def _checker(actual, expected, _positional, _keywords, _args, _kwargs):
    if actual != expected:
        raise AssertionError(f"schedule diverged: {actual}")


def main() -> None:
    report = run_harness(
        HarnessRequest(
            workspace=Path(sys.argv[1]).resolve(),
            entrypoint="aging_scheduler:schedule",
            task_type="correctness",
            trusted_task_path=Path(__file__).resolve().parents[1] / "task.json",
            input_generator=_generator,
            reference=_reference,
            checker=_checker,
            repeats=4,
            timeout_s=10,
        )
    )
    if not report.passed:
        raise SystemExit(report.error or "hidden checks failed")


if __name__ == "__main__":
    main()
