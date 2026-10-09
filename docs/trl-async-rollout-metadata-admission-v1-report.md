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

## Claim boundary

This demonstrates a real source-level information-loss and admission defect relative to issue #7206's proposed complete-rollout behavior, and a local regression candidate. It does **not** establish a general optimizer defect, prevalence in production training, a policy-gradient change, model capability improvement, independent reproduction, or maintainer acceptance.

The candidate only makes freshness admission atomic. It still yields rows individually to the fixed-count/token-budget batchers; those batchers can separate siblings across microbatches, and no test here proves every admitted sibling reaches the same optimizer update or survives epoch stopping. The metadata currently assumes a rollout's queue rows are contiguous. These are blockers to calling the candidate complete or upstream-review-ready.

The related [PR #7249](https://github.com/huggingface/trl/pull/7249) addresses token normalization across accumulation windows and explicitly leaves queue admission out of scope. Its loss-normalization change is not duplicated here.

## Decision

**CONTINUE narrowly** on the queue-to-batcher boundary test. Before proposing an upstream patch, freeze a separate protocol and test scorer-to-queue-to-batcher behavior with a policy-version transition, uneven fork sizes, bounded-queue pressure, malformed/noncontiguous metadata, and epoch stopping. Require all admitted rows to be accounted for in one optimizer update or document and enforce the exact weaker contract. Stop if that contract cannot be met without unsafe oversize batches or silent row loss.

This finding is higher leverage than adding more model screens, but the current patch is not yet a complete external contribution.
