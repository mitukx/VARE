# RVL direct-trainer freshness boundary v1

## Result

`RVLGRPOHooks.train_candidate()` now snapshots its mutable input batch before
waiting for the model lock, then authorizes and checks freshness against that
same detached data before the optimizer call. The direct adapter path fails
closed when it has no current-version source, and the `CapabilityLoop` passes a
live assessor so a version change during model-lock contention is observed.
This is an E0 signal-admission correction. It is not model-learning or task
success evidence.

## Baseline reproduction

The first baseline attempt changed only cached lag values, which are
non-authoritative and therefore did not establish a freshness defect. The
inadequate output is retained at
[`cached-lag-only-inadequate.json`](../results/rvl-direct-trainer-freshness-v1/cached-lag-only-inadequate.json).
The replacement reproducer,
[`reproduce_rvl_direct_trainer_stale_source_v1.py`](../scripts/reproduce_rvl_direct_trainer_stale_source_v1.py),
was run against pinned VARE commit `e912938125b8a135347961801e84f624c2b833eb`.
The signed source versions were policy 4 and verifier 3 while the active
versions were 6 and 5, giving real source lags of 2 and 2. Against the frozen
comparison caps of one, both sources were stale; the receipts still verified,
yet pinned main created `policy-1` and called the trainer once. The runner
asserts the pinned commit and records its SHA-256. Raw output:
[`baseline-reproduction.json`](../results/rvl-direct-trainer-freshness-v1/baseline-reproduction.json).

## Correction and regression

The adapter deep-copies the caller's batch before its first suspension and uses
that detached snapshot for reward authorization, token-exact generation
conversion, group advantages, and freshness. After acquiring the model lock it
authenticates the snapshot, evaluates each example through `LagController`,
then authenticates again immediately before the optimizer call. It uses the
recorded policy and verifier versions plus live version sources; cached lag
fields are diagnostic only. Missing version context, snapshot errors, assessor
errors, and invalid assessor results fail closed. `CapabilityLoop` retains its
replay-time freshness filter and passes the same dynamic assessor into the
final training boundary. The isolated-evaluation wrapper forwards this hook
call. Standalone training scripts provide their run-owned policy and verifier
versions.

The fixed regression starts with signed experiences whose cached lags are zero
and whose source versions are admissible. While the trainer waits for its model
lock, the current policy and verifier advance from 5/4 to 6/5. With a frozen
maximum lag of one, the hook recomputes both lags as two and raises
`stale_training_experience`; the trainer call count remains zero. A second
lock-contention regression mutates the caller's policy version, verifier
version, and shift score after the signed source versions have been captured;
the detached signed versions remain stale and training is rejected. A separate
regression verifies that direct training without version sources raises
`training_freshness_unverifiable`. Raw output:
[`fixed-reproduction.json`](../results/rvl-direct-trainer-freshness-v1/fixed-reproduction.json)
and [`mutable-batch-reproduction.json`](../results/rvl-direct-trainer-freshness-v1/mutable-batch-reproduction.json).

## Validation

- Full CPU suite: 740 passed, excluding the two optional Math500 grader tests
  documented by the repository quick-start.
- `ruff check src/vare tests scripts/reproduce_rvl_direct_trainer_stale_source_v1.py`: passed.
- `mypy src/vare`: passed.
- No model training, accelerator run, external reproduction, or MTS evaluation
  was performed.

The command-level record is [`validation.json`](../results/rvl-direct-trainer-freshness-v1/validation.json).

The historical `validate_rvl_grpo_midstep_fault.py` remains byte-pinned to its
original revision by the v1 revision guard; the current direct-call fixtures
are updated in the v2 runner. The baseline and fixed records here use a small
recording trainer, so they establish admission behavior only, not optimizer
quality or production incidence.
