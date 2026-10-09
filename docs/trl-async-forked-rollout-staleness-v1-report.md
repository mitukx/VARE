# TRL AsyncGRPO forked-rollout staleness v1

**Result: the pinned TRL production path drops a fork sibling after a policy-version transition; a local candidate that preserves rollout identity and fork count keeps both sibling rows together.**

## Research question

When one generated conversation produces multiple `TrainingSequence` rows through prefix-chain forks, can sample-level staleness filtering admit one row but discard another after the policy version changes? Can staleness be checked once per conversation rollout without grouping separate rollouts that share a GRPO `group_id`?

This tests the expected behavior proposed in [TRL issue #7206](https://github.com/huggingface/trl/issues/7206). Current TRL docs describe `max_staleness` in terms of samples and do not already guarantee per-rollout atomic admission. The finding is therefore a reproduced gap against the issue's proposed contract, not a demonstrated violation of an existing documented promise.

## Frozen comparison

Protocol [`trl_async_forked_rollout_staleness_v1`](../protocols/trl_async_forked_rollout_staleness_v1.lock.json) was committed before confirmation in VARE commit `bf8f4ea`. It pins TRL main `ed8cc2f4337fb9b7b1429db31009ad3577cd5b98`, the base and candidate source-file hashes, one identical VARE-authored regression file, and an end-to-end source-path reproducer.

The deterministic scoring fixture has one conversation (`rollout-a`) with two forked sequences and another conversation (`rollout-b`) with one sequence. They share `group_id=41`, so rollout identity cannot be substituted with GRPO group identity. `max_staleness=0`; the version callback returns 0 for its first admission check, then 1. The real scoring, queue filter, fixed-count batcher, and rollout collator are invoked. A three-row microbatch uses fresh sentinel rows if the baseline drops fork siblings.

The candidate patch adds `rollout_id` and `rollout_size` to scored samples, buffers the declared contiguous fork rows before making a single staleness decision, and fails closed on inconsistent membership. Unmodified/custom workers retain the singleton default. The exact diff is retained at [`candidate-fix.patch`](../results/trl-async-group-staleness-microbatch-v1/candidate-fix.patch).

## Results

| Evidence | Pinned base | Candidate patch |
|---|---:|---:|
| Identical focused regression tests | 1 passed, 2 failed | 3 passed |
| Forked `rollout-a` rows in first packed microbatch | 1/2 | 2/2 |
| Scorer-to-collator active completion tokens from `rollout-a` | 1 | 2 |
| Stale rows dropped before first microbatch | 2 | 0 |
| Scored sample rows preserving rollout IDs | 0/3 | 3/3 |

On base, the first fork row enters the microbatch; the sibling and the other old rollout are dropped as stale, and a fresh sentinel fills the batch. The first regression fails on the missing sibling and the metadata-validation regression fails because base silently accepts inconsistent fork membership. The stale-whole-rollout control passes.

On the candidate, the two `rollout-a` sequences are collected and checked together. The policy-version callback is queried once for this rollout; both rows reach the same two-row microbatch. The distinct `rollout-b` remains separate despite sharing `group_id=41`. All three frozen tests pass.

Raw baseline/candidate stdout, stderr, exit codes, environment, commands, and integrity checks are retained under [run-1](../results/trl-async-forked-rollout-staleness-v1/run-1/). The test fixture and one-command source-path reproducer are [here](../reproducers/trl_async_forked_rollout_staleness/test_async_grpo_group_staleness.py) and [here](../scripts/reproduce_trl_async_forked_rollout_staleness_v1.py).

## Adversarial review and earlier group-level screen

An independent read-only adversarial review caught that the preceding screen used three distinct completions with one shared `group_id`; it demonstrated GRPO prompt-group splitting under VARE's stronger comparison-group invariant, but did **not** reproduce issue #7206's forked-sequence case. That earlier result remains separately recorded in [group staleness atomicity v1](trl-async-group-staleness-atomicity-v1-report.md) and is not used as evidence for this issue-specific result. The corrected protocol uses one rollout with two actual `TrainingSequence` objects sharing `rollout_id`.

An independent same-host review confirmed that the candidate groups by `rollout_id`, not `group_id`, and that the built-in scorer enqueues rows contiguously. It also found material gaps: the three tests construct sample-like rows directly and do not exercise `_score_group`'s metadata assignment; bounded-queue/backpressure behavior is reasoned about but not tested; and freshness is atomic only at admission. A batch boundary or epoch stop can still separate or omit fork siblings after admission. The mismatch path should explicitly handle custom sample-like objects without `rollout_id`. Ruff and the full AsyncGRPO suite were not run. This is code review, not external human review.

## Limits and decision

- This demonstrates a changed training microbatch input, not `compute_loss`, a gradient, optimizer mutation, reward, or downstream task outcome.
- The policy-version change is injected through the production version callback; this does not run the complete trainer's optimizer/weight-sync schedule.
- The patch implements per-conversation atomic admission only. It does not guarantee every fork row reaches the same optimizer update or is consumed before epoch stopping, and it does not make every completion in the broader GRPO prompt group atomic.
- PR #7249 remains a separate accumulation-normalization proposal. No VARE upstream PR has been opened or sent, and no maintainer review or adoption is claimed.

**Decision: continue only with a scorer-to-queue integration regression and a batch/update-boundary test, plus bounded-queue and compatibility checks.** The current result clears the narrow existence gate and measures a 1-of-2 to 2-of-2 admitted-input correction in a hand-built queue fixture. It does not yet establish that scorer metadata works end to end, that siblings participate in one optimizer update, a changed gradient, or the frequency of this event in a real async run. Do not propose an upstream change until those contract questions are settled.
