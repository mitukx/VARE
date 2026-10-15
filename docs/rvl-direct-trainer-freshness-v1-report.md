# RVL direct-trainer freshness boundary v1

## Result

`RVLGRPOHooks.train_candidate()` now recomputes policy and verifier freshness at
the optimizer-use boundary. The direct adapter path fails closed when it has no
current-version source, and the `CapabilityLoop` passes a live assessor so a
version change during model-lock contention is observed before `train_step`.
This is an E0 signal-admission correction. It is not model-learning or task
success evidence.

## Baseline reproduction

The reproducer was run from pinned VARE commit
`e912938125b8a135347961801e84f624c2b833eb`. It loaded signed replay examples,
set their cached policy/verifier lag fields to `96`/`97`, verified their signed
receipts, and called the RVL trainer hook directly. The baseline created
`policy-1` and called the trainer once despite those stale cached fields. Raw
output: [`baseline-reproduction.json`](../results/rvl-direct-trainer-freshness-v1/baseline-reproduction.json).

## Correction and regression

The adapter evaluates each example through `LagController` while holding its
model lock, after restoring the incumbent and immediately before the optimizer
call. It uses the experience's recorded policy and verifier versions plus live
version sources; cached lag fields are updated for diagnostics but do not grant
admission. Missing version context, assessor errors, and invalid assessor
results fail closed. `CapabilityLoop` retains its replay-time freshness filter
and passes the same dynamic assessor into the final training boundary. The
isolated-evaluation wrapper forwards this stronger hook call. Standalone
training scripts now provide their run-owned policy and verifier versions.

The fixed regression starts with signed experiences whose cached lags are zero
and whose source versions are admissible. While the trainer waits for its model
lock, the current policy and verifier advance from 5/4 to 6/5. With a frozen
maximum lag of one, the hook recomputes both lags as two and raises
`stale_training_experience`; the trainer call count remains zero. A separate
regression verifies that direct training without version sources raises
`training_freshness_unverifiable`. Raw output:
[`fixed-reproduction.json`](../results/rvl-direct-trainer-freshness-v1/fixed-reproduction.json).

## Validation

- Full CPU suite: 739 passed, excluding the two optional Math500 grader tests
  documented by the repository quick-start.
- `ruff check src/vare tests`: passed.
- `mypy src/vare`: passed.
- No model training, accelerator run, external reproduction, or MTS evaluation
  was performed.

The command-level record is [`validation.json`](../results/rvl-direct-trainer-freshness-v1/validation.json).

The historical `validate_rvl_grpo_midstep_fault.py` remains byte-pinned to its
original revision by the v1 revision guard; the current direct-call fixtures
are updated in the v2 runner. The baseline and fixed records here use a small
recording trainer, so they establish admission behavior only, not optimizer
quality or production incidence.
