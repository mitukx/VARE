"""Non-preemptive priority scheduling with linear aging.

``schedule(jobs)`` takes jobs of the form
``{"id", "release", "priority", "service"}`` and returns
``{"id", "start", "finish"}`` records in execution order.

``release`` and ``service`` are integers, ``release >= 0`` and
``service >= 1``. ``priority`` is an integer. Ids are unique strings.
Time is an integer. The machine starts at time 0 and runs one job at a
time to completion.

When the machine is free at time ``t`` and no waiting job has
``release <= t``, advance ``t`` to the next release. Otherwise start the
waiting job with the greatest effective priority
``priority + (t - release)``. Break ties by the lexicographically smaller
``id``. The chosen job starts at ``t`` and finishes at ``t + service``.
Invalid jobs raise ``ValueError``.
"""
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
        chosen = max(ready, key=lambda job: (job["priority"] + (clock - job["release"]), job["id"]))
        remaining.remove(chosen)
        start = clock
        finish = clock + chosen["service"]
        result.append({"id": chosen["id"], "start": start, "finish": finish})
        clock = finish
    return result
