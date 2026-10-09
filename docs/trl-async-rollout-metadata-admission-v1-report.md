# TRL AsyncGRPO rollout metadata admission diagnostic — 2026-10-09

## Finding

At pinned TRL revision [`ed8cc2f`](https://github.com/huggingface/trl/commit/ed8cc2f4337fb9b7b1429db31009ad3577cd5b98), `TrainingSequence` carries a conversation `rollout_id`, but `_AsyncRolloutLoop._score_group()` drops it while constructing `RolloutSample`. `RolloutQueueDataset` then checks freshness one sample at a time. The production path therefore cannot identify fork siblings for complete-rollout admission. This is directly relevant to open [issue #7206](https://github.com/huggingface/trl/issues/7206), which asks to preserve rollout identity and avoid partial admission.

This was an exploratory, post-hoc diagnostic, not a preregistered study. The exact pinned-source tests, commands, outcomes, and candidate patch are retained in [`results/trl-async-rollout-metadata-admission-v1`](../results/trl-async-rollout-metadata-admission-v1/).

## Candidate change and result

The local candidate propagates `rollout_id` and `rollout_size` from each `TrainingSequence` through scoring, groups the contiguous queue entries for a rollout, validates their identity/version metadata, and applies the stale-version decision to the group before yielding its rows.

| Checkout | Scorer identity regression | Version transition after first fork row | Overall |
| --- | --- | --- | --- |
| Pinned base `ed8cc2f` | Fails: `RolloutSample` has no `rollout_id` | Fails: queue sample constructor cannot carry rollout metadata | 0/5 pass |
| Local candidate | Passes: one-row and two-row rollouts preserve identity/size | Passes: both fork rows are yielded after one group freshness check | 5/5 pass |

The test runner used the local CPU Python environment, cached dependencies, and offline Hugging Face mode. No model weights, GPU, paid API, or external spending were used. The test logs include the experimental API warning.

An independent internal source review confirmed the built-in producer emits each conversation's rows contiguously from one queue writer. The review also caught that the first candidate version recorded group-level queue metrics; the revised candidate retains each row's dequeue timestamp and queue-size sample, with the five-case comparison rerun afterward. This is same-host review, not external reproduction.

## Batcher-boundary counterexample

The actual `FixedCountBatcher` was then run with one sample per microbatch and `max_staleness=0`. The fixture advances the policy version between microbatches, as an optimizer update would, but does not run an optimizer:

| Checkout | First microbatch at v0 | Next microbatch after version advances to v1 | Fork rows delivered | Stale rows dropped |
| --- | ---: | ---: | ---: | ---: |
| Pinned base | row 10 | fresh sentinel row 99 | 1/2 | 1 |
| Local candidate | row 10 | fork sibling row 11 from v0 | 2/2 | 0 |

The baseline loses a sibling to the per-row queue check. The candidate preserves both at admission, but its second sibling can cross the batch boundary and reach a later update without another freshness check. This does not establish an upstream contract violation because the current source checks freshness at queue dequeue, while the issue asks for complete-rollout admission. It does establish that queue atomicity alone does not guarantee same-update consumption or strict use-time freshness. Raw output and the standard-library runner are in the bundle.

## Claim boundary

This demonstrates a real source-level information-loss and admission defect relative to issue #7206's proposed complete-rollout behavior, and a local regression candidate. It does **not** establish a general optimizer defect, prevalence in production training, a policy-gradient change, model capability improvement, independent reproduction, or maintainer acceptance.

The candidate only makes freshness admission atomic. It still yields rows individually to the fixed-count/token-budget batchers; those batchers can separate siblings across microbatches, and no test here proves every admitted sibling reaches the same optimizer update or survives epoch stopping. The metadata assumes a rollout's queue rows are contiguous; source review confirms this for the built-in single writer, but custom producers are not covered. These are blockers to calling the candidate complete or upstream-review-ready.

The related [PR #7249](https://github.com/huggingface/trl/pull/7249) addresses token normalization across accumulation windows and explicitly leaves queue admission out of scope. Its loss-normalization change is not duplicated here.

## Decision

**STOP the queue-only candidate as a complete trainer-correctness fix.** It reproduces and repairs the narrow complete-admission gap, but the batcher counterexample shows its stronger freshness/update guarantee is unproven and potentially in tension with atomic admission. Do not send an upstream PR yet. The next bounded action is to freeze the intended contract and test whether group-aware batching can keep accepted siblings in one optimizer update while preserving token budgets, collective-safe nonempty ranks, bounded queues, and epoch stopping. If that requires unsafe oversize batches or silent partial loss, stop the line and retain the queue-admission finding with its narrow claim boundary.

This finding is higher leverage than adding more model screens, but the current patch is not yet a complete external contribution.
