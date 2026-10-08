# Task: enforce evaluation command output budgets (protocol v2)

## Context

Repository-editing evaluations run candidate-controlled test and metric commands. Their output must be retained for diagnosis, but the coordinator must not buffer unbounded child output in memory or treat a truncated result as a successful command.

## Goal

Make `ResourceLimits.max_output_bytes` an enforced per-stream capture limit in `run_command`. Read stdout and stderr incrementally, retain at most the configured number of bytes from each stream, and terminate/reap the command process group when either stream exceeds its limit. A command terminated for output overflow must be distinguishable from a normal pass and from a wall-time timeout.

## Acceptance contract

- A command that writes more than the configured limit to stdout and stderr is rejected as output-limited.
- Retained stdout and stderr are each no larger than `max_output_bytes` after UTF-8 decoding and re-encoding.
- Output overflow is reported separately from timeout.
- A normal command below the limit still passes and preserves its output.
- On POSIX, a child process in the command's process group does not survive output-overflow termination.

## Constraints

- Edit only `src/vare/environments/runner.py` for the benchmark task.
- Do not change this brief, descriptor, evaluator, protocol lock, or candidate workspace files outside the declared source list.
- This is a deterministic local CPU resource-control regression. It does not establish a hostile-code sandbox, hard memory/CPU quota, or containment against a process that escapes its process group.
