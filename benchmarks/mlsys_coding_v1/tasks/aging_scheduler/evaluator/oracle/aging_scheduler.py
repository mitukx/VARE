"""Reference implementation used only by harness calibration."""

from __future__ import annotations


def _validate(jobs):
    if not jobs:
        raise ValueError("jobs must be non-empty")
    seen = set()
    for job in jobs:
        if set(job) != {"id", "release", "priority", "service"}:
            raise ValueError("job fields are id, release, priority, service")
        if not isinstance(job["id"], str) or not job["id"] or job["id"] in seen:
            raise ValueError("ids must be unique non-empty strings")
        seen.add(job["id"])
        for name in ("release", "priority", "service"):
            if type(job[name]) is not int:
                raise ValueError(f"{name} must be an integer")
        if job["release"] < 0 or job["service"] < 1:
            raise ValueError("release and service out of range")


def schedule(jobs):
    _validate(jobs)
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
