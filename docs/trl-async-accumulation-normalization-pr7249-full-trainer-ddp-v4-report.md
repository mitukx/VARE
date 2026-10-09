# TRL AsyncGRPO normalization through the full Trainer DDP update

**Result: on a fresh, frozen CPU fixture, the pinned PR #7249 candidate matched the pooled-token loss gradient and one-step SGD update through `Trainer.train()`, `Accelerator.backward`, and an actual two-process Gloo DDP reducer. The pinned base failed the predeclared uneven-token cases and passed the equal-token and short-window controls.** This is upstream implementation-validation evidence, not a new algorithm or model-capability result.

## Question

For the token-mean objective described in [TRL issue #7206](https://github.com/huggingface/trl/issues/7206), does the candidate implementation in [TRL PR #7249](https://github.com/huggingface/trl/pull/7249) produce the same gradient and optimizer update as the direct objective over all active completion tokens across ranks and microbatches?

The reference is

\[
L = \frac{\sum_{r,j,t} m_{rjt}\ell_{rjt}}{\sum_{r,j,t}m_{rjt}},
\]

with prompt, context, and padding positions masked. The independent oracle calculates this pooled objective on the exact same sample multiset, differentiates it, and applies one SGD step at learning rate `0.1`.

## Method

Before execution, protocol [v4](../protocols/trl_async_accumulation_normalization_pr7249_full_trainer_ddp_v4.lock.json) froze the base revision `a98fa6a4428f9aae58dfb26d729d7437f662f27a`, PR head `5234eb7c70f4ca7eb92fe8e01611a33eaff17f40`, file hashes, four cases, metrics, thresholds, and static localhost Gloo launch. It uses new token counts and advantages relative to the earlier real-DDP/manual-step run.

The measured path runs production `AsyncGRPOTrainer.compute_loss` and `get_batch_samples`, Transformers `Trainer.train()` and `_inner_training_loop`, `AsyncGRPOTrainer.training_step`, `Accelerator.backward`, DDP `no_sync` accumulation and reducer, and the real SGD optimizer/scheduler step. Captured rank-local `training_step` losses are summed within the window and averaged across the two ranks before comparison with the pooled objective. The `Trainer.train()` returned scalar is retained only as a local diagnostic because the fixture supplies rank-local unprepared DataLoaders.

Runtime: Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, Accelerate 1.12.0; two CPU processes, one thread per rank. Both arms ran in under five seconds. No GPU, paid API, or external spending.

The experiment does not launch the asynchronous rollout worker or use the production queue, batch planner, or Accelerator `DataLoaderDispatcher`. These components are replaced by deterministic rank-local DataLoaders built from the production rollout collator's tensors. This isolates the loss/gradient/optimizer contract while exercising the full Trainer update path.

## Results

Maximum absolute errors against the pooled-token oracle over both ranks:

| Case | Base loss | Base gradient | Base SGD parameters | PR candidate loss | PR candidate gradient | PR candidate parameters |
|---|---:|---:|---:|---:|---:|---:|
| Unequal rank and microbatch tokens | 4.6411e-2 | 2.3206e-2 | 2.3206e-3 | 3.7253e-8 | 0 | 0 |
| Equal-token control | 1.4901e-8 | 5.9605e-8 | 3.7253e-9 | 1.4901e-8 | 5.9605e-8 | 3.7253e-9 |
| One rank has zero active tokens, other rank positive | 1.3333e-1 | 6.6667e-2 | 6.6667e-3 | 2.9802e-8 | 2.9802e-8 | 3.7253e-9 |
| Short final accumulation window | 2.9802e-8 | 0 | 0 | 2.9802e-8 | 0 | 0 |

The candidate passed the frozen `1e-5` loss, gradient, and parameter gates in every case. The base controls passed their `1e-5` gates; both uneven-token cases exceeded the `1e-4` defect gate. Every case completed one Trainer optimizer step, produced finite primary metrics, and ended with identical model-state hashes across ranks.

`Trainer.train()`'s returned `training_loss` differs by rank in this fixture. That field was therefore excluded from the primary metric and retained as a local diagnostic. The primary loss is the explicit mean of both ranks' actual `training_step` return values; the reduced gradients and final parameters are independently checked against the pooled oracle.

## Failed harness attempts retained

- Full-Trainer DDP v1 failed before forward/backward because the harness bypassed the async constructor and omitted its step counters.
- V2 reached the Trainer path, then failed while collecting output: the async log hook expected a worker object, and Trainer had cleared gradients by the time the runner tried to read them.
- V3 completed the base update, but exposed that `TrainerOutput.training_loss` was rank-local with the injected unprepared loaders. The candidate was not run under v3.

These were harness failures or invalid metric aggregation, not scientific non-passes. Their full logs, decisions, and any available base summary are kept alongside the v4 results. V4 was locked before either arm ran and uses a fresh synthetic sample fixture and a globally reduced training-step loss metric.

## Interpretation and limits

This result strengthens the earlier [real-DDP/manual-step audit](trl-async-accumulation-normalization-pr7249-real-ddp-v2-report.md): the base/candidate difference persists when loss scaling and gradient accumulation pass through the actual Trainer optimizer loop and Accelerate backward path. It closes the main remaining local question about whether Trainer's accumulation handling changes the measured update.

It remains a deterministic synthetic token-local model, not a pretrained policy, downstream task, or capability experiment. It does not exercise asynchronous rollout scheduling, production dataloader scatter, distillation, MoE auxiliary losses, vLLM, or accelerator hardware. It does not establish broad TRL correctness, external reproduction, or novelty. The candidate is an existing upstream patch; at the time of inspection, PR #7249 and issue #7206 were still open and the PR listed no reviews. No upstream comment or patch was submitted.

## Decision

**Accept the narrow Trainer/Accelerate/DDP reproduction; do not duplicate PR #7249.** It is a useful technical finding with a pinned one-command reproduction and measurable base-versus-candidate optimizer effect. It is stronger than the earlier manual step but remains far below evidence of real-model improvement. Stop extending this normalization branch unless an upstream reviewer requests a specific additional check. The portfolio's main gap remains independent held-out task-success improvement after a real policy update, plus an outside human reproduction/review.

## Artifacts

- [Frozen v4 protocol](../protocols/trl_async_accumulation_normalization_pr7249_full_trainer_ddp_v4.lock.json), SHA-256 `932bd55fc9cff25a25d46e94f9107f4f12b72f7d3af65d924e4759772d544a55`.
- [Reproduction runner](../scripts/replay_trl_pr7249_full_trainer_ddp_v4.py), SHA-256 `cae0731b9408991a3af700df269ce1ad8400b8d09773bccf1e3c62b32721962b`.
- [Base summary](../results/trl-async-pr7249-full-trainer-ddp-v4/base/summary.json) and [candidate summary](../results/trl-async-pr7249-full-trainer-ddp-v4/candidate/summary.json).
- [Base log](../results/trl-async-pr7249-full-trainer-ddp-v4-base-launcher.log) and [candidate log](../results/trl-async-pr7249-full-trainer-ddp-v4-candidate-launcher.log).
- Earlier harness attempts: [v1 setup failure](../results/trl-async-pr7249-full-trainer-ddp-v1-failure.json), [v2 output-collection failure](../results/trl-async-pr7249-full-trainer-ddp-v2-failure.json), [v3 metric decision](../results/trl-async-pr7249-full-trainer-ddp-v3-metric-invalid.json), and [v3 base summary](../results/trl-async-pr7249-full-trainer-ddp-v3/base/summary.json).

No full repository CI was run for this focused experiment.
