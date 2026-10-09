# TRL Async GRPO accumulation normalization: real DDP audit v2

**Result: the pinned PR #7249 implementation matches the full-window token-mean objective through a real two-process CPU Gloo DDP reducer in four controlled cases. The pinned base does not match in the two predeclared uneven-token cases.** This extends VARE's earlier single-process Trainer test and simulated-rank audit; it is an implementation-correctness finding, not a capability result or a novel algorithm.

## Research question and estimand

Under the token-mean objective stated in [TRL issue #7206](https://github.com/huggingface/trl/issues/7206), does the open [TRL PR #7249](https://github.com/huggingface/trl/pull/7249) compute the same loss gradient and one-step update as pooling every active completion token across all ranks and microbatches in an optimizer window?

For active completion-token losses \(\ell_{rjt}\), the reference objective is

\[
L = \frac{\sum_{r,j,t} m_{rjt}\ell_{rjt}}{\sum_{r,j,t}m_{rjt}},
\]

where \(m\) masks prompt and padding tokens. The independent oracle differentiates this pooled objective directly. The test uses the production `AsyncGRPOTrainer.compute_loss`, Hugging Face Trainer/Accelerator model preparation, actual two-process `DistributedDataParallel` with Gloo, real `no_sync` accumulation, and a fixed manual SGD step. The four cases and tolerances were frozen before execution.

This tests the issue/PR's token-mean contract; it does not claim that token-mean weighting is mandatory for every training objective.

## Prior evidence and novelty

VARE's [v1 audit](trl-async-accumulation-normalization-pr7249-audit-v1-report.md) already reproduced the PR-authored single-process Trainer tests and one optimizer step, but only simulated distributed averaging. This v2 adds an actual two-process reducer and checks both ranks' final gradient and parameter hashes. The production trainer source is pinned at base `a98fa6a4428f9aae58dfb26d729d7437f662f27a` and PR head `5234eb7c70f4ca7eb92fe8e01611a33eaff17f40`; the code delta obtains the accumulation-window token count in `get_batch_samples` and normalizes each microbatch by that count adjusted for DDP's gradient averaging.

Novelty is narrow: this is additional distributed validation of an existing upstream fix. It is not a new normalization method, an independently authored upstream patch, or evidence that the upstream issue is newly discovered.

## Frozen protocol and execution

The v2 lock changes only the launch rendezvous relative to v1. The first v1 attempt never reached worker startup: `torchrun --standalone` connected through reverse-resolved host `1.0.0.127.in-addr.arpa`, retried beyond six minutes, and created no experiment output. This is preserved as a launcher failure, not counted as scientific evidence. V2 pins static rendezvous at `127.0.0.1`; the estimand, data, source revisions, acceptance gates, and analysis remain unchanged. Both arms completed in about five seconds each on CPU.

Runtime: Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, Accelerate 1.12.0; two local processes, Gloo, one CPU thread per process. No GPU, paid API, external spending, or pretrained weights.

## Results

Maximum absolute error against the pooled-token oracle across both ranks:

| Case | Base loss | Base gradient | Base SGD parameters | PR #7249 loss | PR #7249 gradient | PR #7249 SGD parameters |
|---|---:|---:|---:|---:|---:|---:|
| Unequal rank and microbatch token counts | 1.6947e-2 | 8.4735e-3 | 8.4735e-4 | 3.7253e-9 | 5.9605e-8 | 3.7253e-9 |
| Equal-token control | 0 | 0 | 0 | 0 | 0 | 0 |
| One rank has zero active tokens, other rank positive | 1.6667e-1 | 8.3333e-2 | 8.3333e-3 | 2.9802e-8 | 0 | 0 |
| Short final accumulation window | 1.4901e-8 | 2.9802e-8 | 3.7253e-9 | 1.4901e-8 | 2.9802e-8 | 3.7253e-9 |

The locked candidate threshold was `1e-5` for each loss, gradient, and parameter error; all candidate cases pass. The base equal-token and short-window controls pass, while both uneven-token cases exceed their locked `1e-4` defect threshold. In all cases both DDP ranks reported identical gradient and post-step parameter hashes. The candidate's accumulated global token counts were 26, 16, 12, and 10, respectively; the base did not provide this window-level count (`None`).

The zero-token rank can form non-finite per-sequence diagnostic ratios in this synthetic edge case. As frozen, those diagnostics are outside the gate; the measured policy loss, reduced gradient, and update are finite. No claim is made about those diagnostics or every possible zero-token production batch.

## Scope and limitations

- This is a real two-process Gloo reducer, not a full `Trainer.train()` run. The harness invokes production `compute_loss` and Accelerator's DDP model preparation, then calls `backward` and applies fixed SGD directly.
- It does not exercise `Trainer.training_step`, `Accelerator.backward` scaling, production dataloader scatter, asynchronous workers, `AsyncDistillationTrainer`, MoE auxiliary-loss semantics, vLLM, pretrained weights, or downstream task success.
- The deterministic token-local model isolates normalization. The update discrepancy is measured, but no real policy or capability improvement is demonstrated.
- The candidate is an already-open PR's implementation. This audit does not establish maintainers' acceptance, upstream merge, or an external reproduction.
- No full repository CI was run for this focused experiment.

## Decision

**Accept the narrow defect reproduction; do not duplicate PR #7249.** The added evidence removes one clear limitation of VARE's prior audit: the normalization behavior holds (or fails) under an actual distributed gradient reducer, not only simulated rank averaging. The strongest quantitative result is that the base produces gradient error `8.33e-2` in the zero-local-token case while the PR candidate is at floating-point noise (`0`); for unequal positive counts the base gradient error is `8.47e-3` versus `5.96e-8` for the candidate.

This is useful upstream review evidence and a reproducible trainer-correctness result. It does not by itself establish a publication-level contribution or broad research impact. Next, seek outside reproduction or move to a separate high-value question only if a concrete unaddressed issue has a feasible independent evaluation; do not expand this harness into generic distributed infrastructure.

## Reproduction artifacts

- [Frozen v2 protocol](../protocols/trl_async_accumulation_normalization_pr7249_real_ddp_v2.lock.json), SHA-256 `f50e802bbd5d3536ff4f8a7b25a84470eebc43f971ce3b42fb091a27e6f35eec`.
- [Frozen runner](../scripts/replay_trl_pr7249_real_ddp_v1.py), unchanged from the v1 freeze.
- [Base raw summary](../results/trl-async-pr7249-real-ddp-v2/base/summary.json) and [candidate raw summary](../results/trl-async-pr7249-real-ddp-v2/candidate/summary.json).
- [Base launcher log](../results/trl-async-pr7249-real-ddp-v2-base-launcher.log) and [candidate launcher log](../results/trl-async-pr7249-real-ddp-v2-candidate-launcher.log).
- [v1 pre-worker launcher failure record](../results/trl-async-pr7249-real-ddp-v1-launcher-failure.json).

Run each locked command from the VARE repository root. The command includes `PYTHONPATH` for the two exact source checkouts and the shared local dependencies; use a fresh output directory for each arm. The raw summaries record the runner's v1 protocol identifier because its scientific harness is unchanged; this report and the lock identify the v2 launch protocol.
