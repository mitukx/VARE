# TRL AsyncGRPO forked-rollout freshness across microbatches v1

**Result: the atomic-admission candidate emits a stale fork sibling after a policy-version transition between fixed-count microbatches.** The candidate passes its earlier admission test, but it does not satisfy VARE's stricter use-time freshness invariant across optimizer-step boundaries.

## Question and frozen protocol

After a forked rollout is accepted atomically at `max_staleness=0`, can one sibling be packed into a later microbatch after the policy version advances? The protocol was committed before confirmation in VARE commit `a2a9a38`; it pins TRL base `ed8cc2f4337fb9b7b1429db31009ad3577cd5b98`, the candidate patch and source hashes, and the exact CPU script.

The script passes two version-0 fork rows through production `RolloutQueueDataset`, `FixedCountBatcher(microbatch_size=1)`, and `DataCollatorForRollout`, followed by a fresh version-1 sentinel. In the transition arm, the version advances after the first microbatch and before requesting the second. A fixed-version control uses the same sequence without advancing the version.

## Results

| Arm | First microbatch | Second microbatch | Stale rows dropped | Version checks |
|---|---|---|---:|---:|
| Base, fixed version | fork row 10 | fork sibling 11 | 0 | 2 |
| Candidate, fixed version | fork row 10 | fork sibling 11 | 0 | 1 |
| Base, version advances between microbatches | fork row 10 | fresh sentinel 90 | 1 | 3 |
| Candidate, version advances between microbatches | fork row 10 | stale fork sibling 11 | 0 | 1 |

Both source paths complete with exit code 0. The independent same-host review verified the frozen protocol, base/candidate source hashes, patch hash, script hash, raw outputs, and interpretation. Its machine-readable audit is in the run bundle; it is not external human review.

## Interpretation

The candidate makes a freshness decision once for all contiguous rows and then yields them individually. If training crosses a microbatch/optimizer boundary before the sibling is requested, the sibling is emitted without another version check. With `max_staleness=0`, the base drops that now-stale row, while the candidate sends it to the collator. The pinned trainer increments `model_version` at weight sync; `weight_sync_steps` defaults to 1, so the injected boundary corresponds to a supported trainer schedule. This experiment still does not execute the optimizer callback or an optimizer step.

This is a contract tension, not an unqualified upstream defect claim. [TRL issue #7206](https://github.com/huggingface/trl/issues/7206) explicitly proposes complete-rollout admission so a version transition does not remove rollout siblings. The current config documentation describes `max_staleness` as how many policy versions back a sample may still be consumed. Atomic admission preserves every sibling from the admission decision but can exceed that per-consumption freshness cap. The candidate also does not make a rollout indivisible to `FixedCountBatcher` or guarantee every sibling is consumed before epoch stopping.

## Decision

**Reject this candidate as satisfying use-time freshness. Keep the earlier result only as evidence for atomic queue admission under the issue's proposed contract.** Before any upstream patch, define whether a rollout's freshness is fixed at admission or rechecked at each training microbatch, and test the contract through microbatch/optimizer boundaries and epoch stopping. If both all-or-none rollout accounting and a strict freshness cap are required, test a group-aware consumption boundary that keeps the rollout within one optimizer update; fail or explicitly handle rollouts larger than that update's capacity.

No model, gradient, independent task-success, production-prevalence, external-review, or upstream-adoption claim is made. The run is CPU-only with no external spend. Reproduction commands, versions, hashes, exit codes, raw JSON, and same-host audit are retained under [`run-1`](../results/trl-async-forked-rollout-freshness-boundary-v1/run-1/).
