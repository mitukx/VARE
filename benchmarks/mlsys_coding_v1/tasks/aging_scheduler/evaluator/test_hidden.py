"""Hidden aging and tie-break checks for the scheduler."""
from __future__ import annotations

import sys
from pathlib import Path


def _expected(jobs):
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
        result.append({"id": chosen["id"], "start": clock, "finish": clock + chosen["service"]})
        clock += chosen["service"]
    return result


def main() -> None:
    workspace = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(workspace))
    from aging_scheduler import schedule

    aging = [
        {"id": "z", "release": 0, "priority": 100, "service": 10},
        {"id": "a", "release": 0, "priority": 1, "service": 1},
        {"id": "b", "release": 10, "priority": 6, "service": 1},
    ]
    if schedule(aging) != _expected(aging):
        raise AssertionError(f"aging decision diverged: {schedule(aging)}")

    ties = [
        {"id": "b", "release": 0, "priority": 3, "service": 2},
        {"id": "a", "release": 0, "priority": 3, "service": 2},
    ]
    if [item["id"] for item in schedule(ties)] != ["a", "b"]:
        raise AssertionError("tie must prefer the smaller id")

    delayed = [
        {"id": "early", "release": 4, "priority": 1, "service": 3},
    ]
    if schedule(delayed) != [{"id": "early", "start": 4, "finish": 7}]:
        raise AssertionError("idle machine must jump to the next release")

    try:
        schedule([{"id": "a", "release": -1, "priority": 1, "service": 1}])
    except ValueError:
        pass
    else:
        raise AssertionError("negative release must fail")


if __name__ == "__main__":
    main()
