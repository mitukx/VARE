# RVL GRPO rollout-group boundary audit v1

**Status: correctness defect reproduced and patched; no policy-quality claim.**

## Question and scope

When one VARE training batch contains multiple rollout groups for the same task, does the RVL adapter preserve each declared group when computing GRPO advantages? The frozen estimand is the maximum absolute difference between the adapter's per-sample advantages and population-standardized, clipped rewards computed separately within each declared VARE group, in original sample order.

The audit used deterministic CPU fixtures and the pinned [RVL revision](https://github.com/mitukx/Recursive-Verification-Lag/tree/c7e646b043cb56e5ea3c2623bb8a61e065451f72). It did not run a model, optimizer, GPU, paid service, or task-success evaluation.

## Finding

VARE's curriculum samples tasks with replacement. VARE records separate rollout-group IDs, while the pinned RVL default groups generations by `prompt_id`. The adapter previously discarded the VARE group boundary at the trainer call. Therefore, distinct groups for the same task could be normalized together.

For the frozen witness, two groups have rewards `[0, 0]` and `[1, 1]`, with the same prompt ID. Separate group normalization yields `[0, 0, 0, 0]`; the exact pinned RVL helper yields `[-0.999998000006, -0.999998000006, +0.999998000006, +0.999998000006]`. This is a direct change in the advantages supplied to the policy update, not evidence about the eventual parameter update or task performance.

## Change and validation

The adapter now computes normalized advantages from complete declared VARE groups and passes them through RVL's supported `train_step(..., advantages=...)` override. It rejects incomplete, mixed grouped/ungrouped, malformed, or inconsistent groups before an optimizer step. Inputs without VARE group metadata retain the prior default trainer path.

The frozen protocol's initial test fixture was under-calibrated: an all-zero implementation could have passed its groupwise expectation, and the baseline adapter test observed only that no override was passed. We retain this limitation rather than describing the initial fixture as decisive. Supplemental tests were added after seeing the original baseline result, so these are post-hoc regression controls, not preregistered confirmation: nonconstant rewards with order checks, distinct-prompt positive control, symmetric clipping, and incomplete-group rejection. The exact pinned helper runner separately establishes the merged baseline vector.

| Evidence | Result |
|---|---|
| Exact pinned RVL helper on frozen witness | Prompt-ID merge reproduced; four advantages are approximately `[-1, -1, +1, +1]` |
| VARE baseline, original + supplemental adapter tests | 5 failed, 1 passed; failing cases did not pass groupwise advantages (captured value was `None`) |
| Patched adapter focused and adjacent integration tests | 23 passed |
| Full VARE test suite, with the project virtualenv on `PATH` | Passed; see [`full-pytest-final.log`](../results/rvl-grpo-group-boundary-v1/full-pytest-final.log) |
| Trainer/model execution | Not run; PyTorch is not installed in the project environment |

The added regression tests verify the normalization contract with a fake trainer. They do not verify gradients, optimizer behavior, downstream success, or external reproduction. The fix is not a novel estimator or RL algorithm.

## Decision

**Continue only as a concrete integration correctness repair; stop this as a standalone research contribution.** The more important open question remains whether any affordable real-model update improves independently graded task success. No current evidence from this audit answers it. The partial-audit efficiency study remains stopped: its exact unbiased estimator did not establish a finite-budget MSE advantage over full-group audits.

## Reproduction

From the repository root, with the project environment installed:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest -q \
  tests/test_rvl_grpo_group_boundary.py \
  tests/test_rvl_grpo_group_boundary_calibration.py \
  tests/test_rvl_grpo_hooks.py \
  tests/test_grouped_replay.py \
  tests/test_replay_group_freshness.py
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/reproduce_rvl_group_normalization_reference.py
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest -q
```

The protocol lock is [`rvl_grpo_group_boundary_v1.lock.json`](../protocols/rvl_grpo_group_boundary_v1.lock.json). Raw test logs and exact pinned-helper output are in [`results/rvl-grpo-group-boundary-v1/`](../results/rvl-grpo-group-boundary-v1/).
