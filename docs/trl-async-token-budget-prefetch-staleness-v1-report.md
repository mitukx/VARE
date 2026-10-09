# TRL AsyncGRPO TokenBudgetBatcher freshness prefetch v1

## Post-run audit correction

An independent review found that the frozen script computes `consumption_staleness_by_microbatch` using the final version after all three collator calls. Its transition-arm vector `[1, 1, 0]` is therefore invalid for the first microbatch. The actual collator order is row 10 at version 0, row 11 at version 1, then sentinel row 90 at version 1; the reconstructed lags are `[0, 1, 0]`. The report's second-microbatch finding remains supported by the call order, but the raw vector must not be used as evidence. Frozen code and output are retained unchanged. The follow-up [optimizer-window study](trl-async-token-budget-optimizer-window-v2-report.md) records version at each consumption point.

**Result: both the pinned base and the local atomic-admission candidate send a version-0 fork sibling to the collator after the current version advances to 1, with `max_staleness=0`.** The base sample passes its staleness check during `TokenBudgetBatcher` look-ahead, before the preceding microbatch causes a policy update; it is held by the batcher and collated afterward.

## Question and frozen protocol

Does production token-budget batching preserve the `max_staleness` limit at the point a row reaches the training microbatch, or can read-ahead check it too early? The protocol was frozen in VARE commit `e49d1ab` before running the confirmation and pins the TRL source, local candidate patch, and reproducer hashes.

The deterministic fixture has two version-0 fork rows, each two input tokens, followed by version-1 sentinels. `TokenBudgetBatcher` has a two-token budget and one data-parallel process, so it must emit the first row when the second row does not fit. The experiment advances the current policy version from 0 to 1 after collating that first microbatch and before requesting the next. Both versions also run a fixed-version control.

## Results

| Source | Version transition | First microbatch | Second microbatch | Lag when collated | Rows dropped |
|---|---:|---|---|---:|---:|
| Pinned base | no | fork row 10 | fork row 11 | 0 | 0 |
| Candidate | no | fork row 10 | fork row 11 | 0 | 0 |
| Pinned base | 0 → 1 | fork row 10 | fork row 11 | 1 | 0 |
| Candidate | 0 → 1 | fork row 10 | fork row 11 | 1 | 0 |

The base checked version twice before returning the first microbatch because the token-budget planner requested the second row to discover the overflow. The candidate checked only once while admitting the complete rollout. In both cases, that second row was already held in the batcher when the version advanced, and both outputs reached the collator afterward. Both process exits were 0. The raw records and environment are retained in [run-1](../results/trl-async-token-budget-prefetch-staleness-v1/run-1/).

## Interpretation and limits

This isolates a freshness timing gap in the dynamic batching path: queue-read freshness and training-microbatch freshness are not equivalent when the planner reads ahead. At the configured boundary, the second row has lag 1 at collation although the maximum is 0. The candidate's rollout-level admission policy does not remove this batching boundary. The pinned config defines `max_staleness` as the maximum number of weight-update steps a sample may lag before discard, and `weight_sync_steps` defaults to 1.

The version transition is injected between collator outputs; the script does not instantiate an optimizer, model forward, or full Hugging Face Trainer loop. Thus it demonstrates a concrete change in trainer input for one supported configuration, not a changed gradient, frequency in production, downstream task outcome, or capability gain. The result does not establish whether maintainers intend freshness to be checked at queue retrieval or at model use; the public issue proposes complete-rollout admission before token-budget packing, so that contract boundary must be resolved explicitly.

**Decision: retain the candidate's earlier atomic-admission measurement, but do not call it a complete fix.** The next useful experiment is to reproduce the actual optimizer-window schedule (including Trainer prefetch, `gradient_accumulation_steps`, weight sync, and epoch stop) and determine which boundary current `max_staleness` is meant to constrain. Do not open an upstream patch until this contract and a narrow correction are tested. This remains a source-level CPU finding only.
