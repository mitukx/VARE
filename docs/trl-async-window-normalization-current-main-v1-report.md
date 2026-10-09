# TRL AsyncGRPO accumulation-window normalization — current-main audit v1

> **Status update:** the earlier gap noted below—unverified current Trainer window plumbing and DDP—was closed for complete accumulation windows by the outcome-informed two-rank replay in [the v2 report](trl-async-window-normalization-current-trainer-ddp-v2-report.md). That replay matched the candidate update to the full-batch reference on its frozen fixture; it does not resolve live-queue frequency, capability impact, or novelty.

**Decision: the pooled-token normalization defect reproduces in current TRL main's production `compute_loss` method. When supplied the accumulation-window token count, a narrow port of the existing denominator logic removes the discrepancy on frozen CPU fixtures. The existing PR #7249 is stale against that revision.** The current-source candidate does not yet wire the window count through `get_batch_samples`, and has not passed the full Trainer/DDP path; it is not an upstream-ready patch.

## Question

Does current TRL `AsyncGRPOTrainer` produce the pooled active-token gradient across a gradient-accumulation window when the microbatches have unequal completion-token counts and different mean token gradients?

The relevant current source is pinned to TRL commit `f4526e10e25c8618855932528c232e2284c7801c`. `AsyncGRPOTrainer.compute_loss` divides the masked loss sum by that microbatch's `global_n_tokens / world_size`, then divides by `current_gradient_accumulation_steps`. The class declares `loss_is_scaled_for_ga = True`, so the Trainer does not apply another accumulation divisor. Each microbatch therefore contributes its own token-mean gradient divided by the actual accumulation-window length.

For microbatches `i` with active-token count `nᵢ` and summed token gradient `Gᵢ`, the resulting one-rank accumulated gradient is

```text
g_current = (1/K) Σᵢ Gᵢ / max(nᵢ, 1)
```

The pooled-token objective is

```text
g_pooled = Σᵢ Gᵢ / Σᵢ nᵢ
```

These agree when microbatch sizes match or when their mean token gradients happen to match; neither condition is guaranteed by the trainer.

## Frozen protocol and result

The protocol was frozen before evaluating its scalar fixture in [`protocols/trl_async_window_normalization_current_main_v1.lock.json`](../protocols/trl_async_window_normalization_current_main_v1.lock.json). The standalone reproducer verifies source hashes and required formula fragments, then calculates the fixture without importing TRL or Transformers.

| Case | Microbatch `(tokens, summed gradient)` | Current gradient | Pooled gradient | Absolute error |
|---|---|---:|---:|---:|
| Counterexample | `(1, 1), (9, 0)` | 0.5 | 0.1 | 0.4 |
| Negative control | `(1, 1), (9, 9)` | 1.0 | 1.0 | 0.0 |

At learning rate `0.1`, the counterexample's SGD delta is `−0.05` under the current formula and `−0.01` under the pooled objective. This is a 5× update magnitude on the selected fixture, not a claim about typical training runs. The negative control confirms the discrepancy is caused by the mismatch between per-microbatch normalization and the pooled objective, rather than accumulation alone.

Reproduction command:

```bash
python scripts/reproduce_trl_async_window_normalization_current_main_v1.py
```

Raw source snapshots, the result JSON, current PR metadata, and a non-mutating applicability check are retained in [`results/trl-async-window-normalization-current-main-v1/`](../results/trl-async-window-normalization-current-main-v1/). The script has no network, model, GPU, or paid-service dependency after checkout.

## Upstream relationship and novelty

This is not a novel normalization method. It revalidates the issue already described in [TRL issue #7206](https://github.com/huggingface/trl/issues/7206) and the intended fix in [TRL PR #7249](https://github.com/huggingface/trl/pull/7249). At this audit, #7249 is open, not merged, and targets base `a98fa6a`; current main is `f4526e1`. Running `git apply --check` for the PR diff on a clean checkout of current main fails in both the AsyncGRPO and AsyncDistillation trainer files. Therefore the old PR is not a directly applicable fix for the current source tree.

The new evidence is (1) a source-pinned check that the failure expression remains in current main, (2) execution of the exact AST-extracted production `compute_loss` method with PyTorch autograd and a fixed CPU scalar model, (3) a narrow port of the existing PR's denominator logic, and (4) a checked failure of the old diff to apply cleanly. A separately written pure-Python audit independently recomputed 15 checks from the frozen fixture and hashes; all passed.

In the method-level run, base accumulated gradient was `0.5` versus `0.1` for the full-batch reference (absolute error `0.4`). The denominator candidate, when explicitly given the window's total count, was `0.1000000015`, within `7.5e-9` of the reference, and its SGD parameter delta matched exactly at `−0.01`. The same-mean negative control and a one-microbatch short final window matched for base and candidate. The no-window direct-call fallback preserved the old single-batch scale exactly. Because this runner manually supplies the window count, it does not test the needed current-Trainer `get_batch_samples` plumbing. The first runner attempt failed before calculating outcomes due to a fixture-wrapping `TypeError`; it is retained and excluded, and the corrected runner then passed the frozen method-level criteria.

This still does not establish a current-main `Trainer.train()` or `Accelerator/DDP` regression, behavior across the real async queue/planner, impact frequency in normal runs, a pretrained-model effect, downstream task impact, or maintainer acceptance. The scalar model checks the trainer's actual loss method and autograd normalization, but does not represent an LLM's clipped surrogate behavior beyond the local token-gradient form.

## Decision and next test

**CONTINUE narrowly to current Trainer/DDP validation; STOP treating PR #7249's existing patch as ready to merge.** The port passes the method-level autograd test, so the next decisive step is to exercise current `get_batch_samples`, `Trainer.training_step`, and two-rank DDP against the pooled reference, including a short final window. Only a passing regression and clean current-source patch would justify updating or replacing the upstream proposal. This is reproducible trainer-loss correctness evidence, not model capability evidence.

## Provenance

- Current TRL source: [AsyncGRPO trainer at `f4526e1`](https://github.com/huggingface/trl/blob/f4526e10e25c8618855932528c232e2284c7801c/trl/experimental/async_grpo/async_grpo_trainer.py) and [`_BaseTrainer`](https://github.com/huggingface/trl/blob/f4526e10e25c8618855932528c232e2284c7801c/trl/trainer/base_trainer.py).
- Frozen protocol SHA-256: `d8fe2ad9cfff93c12203c931f00d89443709246c147f76472343ad4ca0d850be`.
- Reproducer SHA-256: `c91a49b7c6a6338dfb7102c6cf1a4d3ca3a220e1c1bda403ef953df3f9c2c0fa`.
- Source snapshot SHA-256 values are recorded in the protocol and raw summary.

### Production-method follow-up

The outcome-informed follow-up was separately locked in [`trl_async_window_normalization_current_compute_loss_v1.lock.json`](../protocols/trl_async_window_normalization_current_compute_loss_v1.lock.json). It executes the pinned `compute_loss` AST with PyTorch autograd and saves the candidate port and patch in [`run-2`](../results/trl-async-window-normalization-current-main-v1/run-2/). The independent audit passed `15/15` checks. Reproduce with:

```bash
python scripts/reproduce_trl_async_current_compute_loss_v1.py
```

Protocol SHA-256: `db72b42adff62d54f139acef93c02ffe65de3fe8ebdcfca592d7e7fa83430c50`; reproducer SHA-256: `289d2808232e77970bdad61886b687657e9c10a8cbfeac4485f502280e65d7c9`.
