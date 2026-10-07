# Task: preserve rollout groups during freshness filtering

## Context

Group-relative policy objectives compare responses generated from the same rollout context. A group is usable only while every member required by the comparison is current. Filtering stale members individually can leave a partial group that still looks sampleable.

## Goal

When `sample_current(..., grouped=True)` evaluates freshness, admit or drop each rollout group atomically. If any member in a group is currently inadmissible, return none of that group's members. Keep independent fresh groups sampleable.

## Acceptance contract

- A group of four experiences with one freshness result of `None` contributes zero members to the sampled batch.
- A separate all-fresh group remains eligible and is returned whole when selected.
- The returned grouped batch never contains a partial member set for any group.
- `grouped=False` retains its per-experience freshness behavior.

## Constraints

- Edit only `src/vare/replay.py` for the benchmark task.
- Do not change this brief, descriptor, evaluator, protocol lock, or files under the candidate workspace outside the declared source file.
- This is a deterministic local CPU replay-contract task. It does not establish model learning, verifier accuracy, or throughput under real workloads.
