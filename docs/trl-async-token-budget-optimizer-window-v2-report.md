# AsyncGRPO token-budget optimizer-window source-path audit v2

**Result:** the current TRL queue, `TokenBudgetBatcher`, collator, and installed Transformers `Trainer.get_batch_samples` sequence place a version-0 fork sibling in the next optimizer window after a simulated version 0→1 transition at GA=1, but both siblings enter the same pre-update accumulation window at GA=2. Base and local atomic-admission candidate agree.

## Frozen question and method

Protocol `trl_async_token_budget_optimizer_window_v2` and its reproducer were frozen before confirmation in commit `ce5f1c1`. It pins TRL `ed8cc2f4337fb9b7b1429db31009ad3577cd5b98`, Transformers 4.57.3 and the `Trainer.get_batch_samples` implementation hash. A deterministic queue contains two 2-token version-0 fork rows, followed by fresh version-1 sentinels; `token_budget=2`, `max_staleness=0`, and one process. The script calls the installed `Trainer.get_batch_samples` method at accumulation widths 1 and 2, then associates each collated row with the version at that call.

## Results

| Path | GA | Before transition | After transition | Observed lag at collator |
|---|---:|---|---|---:|
| Base | 1 | row 10 | row 11 | 1 |
| Candidate | 1 | row 10 | row 11 | 1 |
| Base | 2 | rows 10, 11 | none in first window | 0, 0 |
| Candidate | 2 | rows 10, 11 | none in first window | 0, 0 |

The full recorded JSON, stderr, commands, environment, and exit codes are in [run-1](../results/trl-async-token-budget-optimizer-window-v2/run-1/); both paths exited 0 and matched. Independent review verified hashes and the ordering. Transformers collects `gradient_accumulation_steps` batches before its training loop; TRL's weight sync is registered on `on_step_end`, after the optimizer step.

## Interpretation and limits

This confirms the v1 script's second-row observation, while correcting its invalid all-batches staleness vector; see the [v1 report correction](trl-async-token-budget-prefetch-staleness-v1-report.md). It shows the outcome depends on where the rollout boundary falls relative to an accumulation window. It is not a novel algorithm, and it does not show a changed gradient, optimizer update, real-run prevalence, or task success. The installed `Trainer.get_batch_samples` call was exercised, but the full DataLoader/Accelerate dispatch and Trainer loop were not; the next report covers that boundary.

`max_staleness` is checked when `RolloutQueueDataset` yields a sample. The config description does not make clear whether its limit applies at queue admission or at the later training-step use after batch prefetch. Issue #7206 proposes complete-rollout admission before token-budget packing, a distinct boundary. The candidate's atomic admission still does not guarantee same-update use.

**Decision:** do not present the atomic-admission candidate as a freshness fix. Continue only to settle the exact check/use contract and test it at the real Trainer step boundary. This result alone does not justify an upstream patch.
