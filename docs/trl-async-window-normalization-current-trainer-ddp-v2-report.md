# Current TRL AsyncGRPO accumulation-window replay through Trainer and DDP

**Decision: the existing pooled-token normalization discrepancy reaches the current Transformers Trainer and two-rank DDP optimizer update. The candidate window denominator matches the full-batch update on the frozen unequal-token fixture. Stop this as a VARE novelty direction: the underlying issue and intended correction already exist in TRL issue #7206 and open PR #7249.**

## Question and estimand

On TRL `f4526e10e25c8618855932528c232e2284c7801c`, does `AsyncGRPOTrainer`'s actual production loss path produce the same one-step policy-parameter update as a full-batch pooled active-token objective when two accumulated microbatches have unequal token counts and different mean token gradients? Does passing the accumulation-window token count through current `Trainer.get_batch_samples` recover that update under `Accelerator`'s `DataLoaderDispatcher` and two-process DDP/Gloo?

For a window with microbatch active-token counts `nᵢ`, summed token gradients `Gᵢ`, and `K` microbatches, the current loss path accumulates the average of microbatch means:

```text
g_current = (1/K) Σᵢ Gᵢ / nᵢ
```

The pooled objective is:

```text
g_pooled = Σᵢ Gᵢ / Σᵢ nᵢ
```

The candidate sums `global_n_tokens[0]` across the current Trainer accumulation window, then normalizes each local summed loss by `window_tokens / world_size`. DDP's gradient averaging then produces the global pooled-token gradient. This is a correctness comparison for the selected scalar fixture; it does not imply typical-run impact or downstream policy improvement.

## Frozen conditions and execution path

The source, runtime, candidate rule, fixtures, acceptance criteria, and claim boundary are recorded in [`protocols/trl_async_window_normalization_current_dispatcher_ddp_v1.lock.json`](../protocols/trl_async_window_normalization_current_dispatcher_ddp_v1.lock.json). This is an outcome-informed follow-up to the single-process Trainer and method-level experiments, not blind prospective confirmation. The final two-rank replay used TRL `f4526e1`, Transformers `5.19.0`, PyTorch `2.9.1`, Accelerate `1.12.0`, CPU, and Gloo. It executed the production TRL rollout collator and `compute_loss`, Accelerate `DataLoaderDispatcher`, the current Transformers `Trainer.train()`/optimizer path, and actual two-rank DDP gradient reduction.

The deterministic model has one scalar parameter. The candidate uses the exact AST-extracted production `compute_loss` method with only its denominator block replaced, and supplies the window count from the actual Trainer batch-collection hook. The reference packs the same per-rank token rows into one full-batch update. Learning rate is `0.1`; each arm takes one SGD step.

## Results

| Fixture | Arm | Accumulated gradient | Full-batch gradient | Parameter-delta absolute error |
|---|---|---:|---:|---:|
| Unequal-token counterexample: 2 then 18 tokens per rank; second microbatch gradient 0 | Base | 0.5 | 0.1 | 0.04 |
| Same | Candidate | 0.1 | 0.1 | 0.0 |
| Equal-mean negative control | Base | 1.0 | 1.0 | 0.0 |
| Equal-mean negative control | Candidate | 1.0 | 1.0 | 0.0 |

On the counterexample the base update is `−0.05` and the pooled reference/candidate update is `−0.01`. The candidate observed the frozen accumulation-window count `[40.0]`. All ranks completed one optimizer step, rank parameters agreed, and each prepared loader was `DataLoaderDispatcher`. The independent standard-library-only audit reconstructed all 17 checks successfully. Raw Trainer logs and the machine-readable result are in [`run-4`](../results/trl-async-window-normalization-current-main-v1/run-4/); reproduce with:

```bash
GLOO_SOCKET_IFNAME=lo0 TRL_EXPERIMENTAL_SILENCE=1 \
PYTHONPATH=/tmp/trl-current-deps:/tmp/trl-current-pr7249-audit \
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29592 \
  scripts/reproduce_trl_async_current_dispatcher_ddp_v1.py \
  --output results/trl-async-window-normalization-current-main-v1/run-4/summary.json
python scripts/audit_trl_async_current_dispatcher_ddp_v1.py
```

The single-process current-Trainer replay in `run-3` separately passed its shortened final-window fixture: with configured accumulation 2 and one remaining batch, the Trainer supplied count `[1.0]`, and base/candidate each matched the full-batch update. The finite-stream short-window DDP fixture stalled after the only dispatched batch and before an update; it is preserved in `run-4/attempt-1.md` and excluded from the DDP acceptance set. This is a harness/finite-stream limitation, not evidence of a TRL trainer defect. The live AsyncGRPO queue path was not run.

## Prior art and contribution boundary

This is not a novel estimator or defect discovery. TRL issue [#7206](https://github.com/huggingface/trl/issues/7206) and pull request [#7249](https://github.com/huggingface/trl/pull/7249) already describe the accumulation normalization concern and an intended correction. The old PR targets an earlier source revision and does not apply directly to current main; this replay adds concrete coverage of current-main Trainer window plumbing, the current dispatcher, and two-rank DDP under a controlled unequal-token counterexample. It does not establish how frequently real runs encounter material update distortion, maintainer acceptance, or task-success effects.

## Decision

**STOP this branch as a VARE research contribution.** Retain the regression packet as an independently reproducible technical finding and as a validation boundary for any future upstream revision. Do not claim novelty, production-run frequency, policy-quality improvement, or capability gain. The next research investment should target the still-open independent held-out task-success result after a real model update or obtain an outside technical reproduction of a retained result; do not add another normalization variant.

## Provenance

- TRL commit: `f4526e10e25c8618855932528c232e2284c7801c`
- TRL source SHA-256: `2657481258dcc04112761e9ff61a35cb93b2b72c3e197e4524f4602fd6a2dbfe`
- DDP protocol SHA-256: `98af5330dd8198977c741073fae57e5d70a735e81857a3333a8784bc5c06c80c`
- Runner SHA-256: `f55bf570d332f78b3482efc076551236f2e4f4d74ee3fb61fb4629349ee8c32f`; independent auditor SHA-256: `0221e7c75938c1de90e692622a3cb47d0f402545a447131c85463448ca7a12b1`.
- Runner and independent auditor source are retained in `scripts/`; audit result is `17/17`.
- GPU, paid API, and external spend: `0`.
