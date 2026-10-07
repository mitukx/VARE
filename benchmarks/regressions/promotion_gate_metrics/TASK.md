# Task: reject non-finite promotion evidence

## Context

`PromotionGate` compares held-out evaluation reports to decide whether a candidate replaces the incumbent. Python floating-point comparisons with `NaN` are false, so a guard written only as `metric < threshold` can silently accept an invalid metric. The same risk applies to slice regressions, cost limits, verifier disagreement, and paired per-task confidence calculations.

## Goal

Make the promotion decision fail closed when any consumed report metric or paired score is non-finite. Keep finite, valid existing reports behaviorally compatible.

## Acceptance contract

- Reports with non-finite primary scores, slice scores, cost, or verifier-disagreement values are rejected with an explicit invalid-metric reason whether the value appears on the incumbent or candidate.
- When paired confidence evidence is enabled, non-finite per-task scores cannot produce an accepted decision.
- A finite candidate that exceeds the configured primary-gain threshold and satisfies the other gates remains accepted.
- Decision values retained for rejected malformed reports are finite and serializable.

## Constraints

- Edit only `src/vare/promotion.py` for the benchmark task.
- Do not change the test harness, this brief, descriptor, evaluator, or protocol lock to obtain a pass.
- This is a local CPU unit-level task. It does not establish policy learning, model improvement, broad verifier validity, or production deployment safety.
