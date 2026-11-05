# ResearchGym execution backend

This slice runs a prepared experiment and recovers committed results after the
worker is interrupted. It does not define research tasks, grade them, or
install `vare research` commands. Those belong to other slices.

The programmatic entry point is `vare.research_lab.ExperimentRunner`.
`scripts/research_lab_exec.py` is only a thin way to exercise that API.

## What a run records

`prepare` stores the canonical spec, source fingerprint, and protected-file
fingerprint. `execute` runs each command in its own child and commits that
child's stdout, stderr, and record before starting the next command. `resume`
after the owner process is gone reruns only steps that were not committed.
Calling `execute` or `resume` on a terminal run returns the stored manifest
and does not start the commands again.

A second terminal commit with different bytes raises. The same bytes are a
duplicate, not a new result. `status` reads the store. It does not invent a
summary for a step that was still running when the worker died.

A wall-clock timeout is `wall_timeout`. It sets `candidate_incorrect` only
when the metric contract says `timeout_counts_as_incorrect`. It is never
acceptance.

`accepted` is true only when every step exited 0 inside its limits, protected
files and declared source files still match the enrolled fingerprints, and
the trusted metric contract holds. Keys such as `passed` or `accepted` inside
a candidate metrics file are ignored.

## Isolation

Default isolation is `process_group`. It reuses `vare._grader_guard` so the
child group is signalled when the owner pipe closes, including after the
worker is killed. Optional limits are wall time, CPU time, RSS, output bytes,
file size, and an `RLIMIT_NPROC` cap. `network_policy=deny` (the default)
enters an unprivileged user and network namespace before `exec` when the
kernel allows it. If that namespace cannot be created, the run fails closed
and the candidate is not started.

This is not a secure sandbox. The candidate keeps the caller's uid, can read
and write any path that uid can write, and can leave the process group by
creating a new session. Protected-path checks detect those writes afterward
and fail closed. They do not prevent the write.

`isolation=docker` builds the command with
`vare.container_isolation.DockerIsolationRunner` and requires a digest-pinned
image. If Docker cannot start the container, the run fails closed. It does
not fall back to `process_group`.

`backend=mlx` only evaluates the headroom gate. The gate reads available
memory and other processes' command lines for `grpo_mlx`, `mlx_lm`, and
`mlx.core`. It does not signal those processes. A failed gate records
`mlx_refused_headroom` and starts no step. A passing gate still records
`mlx_unavailable`: this process does not launch MLX or Metal.

## Failure classes exercised here

Invalid patches, protected-file modification, path traversal, symlinks,
blocked executables, extra-process caps, denied network, infinite loops,
memory and output limits, corrupt metrics, missing metrics, self-reported
success, worker death, a second execution after restart, and source changes
after enrollment all leave `accepted` false. Raw step bytes that were
committed stay in the store.

## Out of scope

Task packages, independent graders, the research controller, trajectory
export, and GRPO training are not part of this backend.
