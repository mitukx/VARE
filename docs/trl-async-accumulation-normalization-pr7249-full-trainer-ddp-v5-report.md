# TRL AsyncGRPO token normalization through Trainer and DDP — v5

**Result:** the corrected, frozen CPU run executed the locked rank/microbatch token layout. In the primary unequal-token case, the pinned TRL base differed from the pooled-token oracle (maximum gradient error `2.8572e-3`), while PR #7249 matched within `5.96e-8`. Across all four cases the candidate stayed below the frozen `1e-5` gate. This is synthetic trainer-correctness evidence for an existing upstream candidate, not a new method, pretrained-model result, or task-success gain.

## Question and estimand

For the token-mean clipped GRPO objective, do the pinned base and PR #7249 candidate produce the same gradient and one-step SGD update as directly pooling all active completion tokens across ranks and gradient-accumulation microbatches?

The oracle is

\[
L = \frac{\sum_{r,j,t} m_{rjt}\ell_{rjt}}{\sum_{r,j,t}m_{rjt}},
\]

where `m` masks prompt, context, and padding positions. The update comparison uses the same initial deterministic token-local model and one SGD step (`lr=0.1`, no weight decay or gradient clipping).

## Frozen protocol and correction history

V4 is retained as a protocol non-pass. Its lock specified global token counts `[13, 10]` for the unequal case, but the runner transposed the rank and microbatch axes and its summaries recorded `[9, 14]`. V5 fixes the construction, uses fresh sample IDs/counts/advantages, and asserts each rank's microbatch counts and the global microbatch totals before constructing the Trainer or taking any optimizer step.

The v5 lock was committed at VARE commit `70751bf6f467648b81c0a948de3025e28dc2b549` before execution. Protocol SHA-256: `2d89c274929c1c30b23e9946127b98db563ed5fac6bea74e34b3e56cbd653613`. It pins base `a98fa6a4428f9aae58dfb26d729d7437f662f27a`, PR head `5234eb7c70f4ca7eb92fe8e01611a33eaff17f40`, source-file hashes, Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, Accelerate 1.12.0, CPU-only Gloo, and a 300-second per-arm cap.

The runner exercises production `AsyncGRPOTrainer.compute_loss` and `get_batch_samples`, Transformers `Trainer.train()` / `_inner_training_loop`, `training_step`, `Accelerator.backward`, DDP `no_sync` and reducer, and the SGD optimizer/scheduler. It uses rank-local deterministic DataLoaders built from the production rollout collator's tensors. The async rollout worker, queue, production batch planner, and Accelerator `DataLoaderDispatcher` remain outside scope.

## Results

Maximum absolute errors against the pooled-token oracle over both ranks:

| Case | Global active tokens per microbatch | Base loss | Base gradient | Base update | PR candidate loss | PR candidate gradient | PR candidate update |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unequal rank and microbatch counts | `[16, 14]` | `5.7142e-3` | `2.8572e-3` | `2.8572e-4` | `5.9605e-8` | `5.9605e-8` | `3.7253e-9` |
| Equal-token control | `[10, 10]` | `2.9802e-8` | `0` | `0` | `2.9802e-8` | `0` | `0` |
| Zero local tokens on one rank | `[8, 5]` | `8.6539e-2` | `4.3269e-2` | `4.3269e-3` | `1.1921e-7` | `5.9605e-8` | `7.4506e-9` |
| Short final accumulation window | `[13]` | `0` | `5.9605e-8` | `3.7253e-9` | `0` | `5.9605e-8` | `3.7253e-9` |

All four cases completed exactly one optimizer step per arm; both DDP ranks ended with matching state hashes. V5 verified the local counts for the primary unequal case as rank 0 `[4, 9]` and rank 1 `[12, 5]`, producing global totals `[16, 14]` as locked. All frozen candidate and base-control gates passed. The base defect gate passed on the unequal and zero-local cases.

The complete wrapper run is `results/trl-async-pr7249-full-trainer-ddp-v5/run-1/`. The base summary SHA-256 is `e3e067a01577e9f02b0201486841b3833aea2159d7f6fe69df7d2efb74dcff66`; candidate summary SHA-256 is `d90a48df67777fa4e433436847233692b6aa89a3ab28fdff47251b20e0ca349e`. The wrapper fetched clean source checkouts and verified both commit and source-file hashes. Reproduce with:

```bash
python scripts/reproduce_trl_pr7249_full_trainer_ddp_v5.py \
  --output results/trl-async-pr7249-full-trainer-ddp-v5/run-N
```

`run-N` must be a new output path. The protocol checks the runner and wrapper hashes and the installed runtime; it does not install dependencies.

## Prior art, novelty, and limits

TRL PR [#7249](https://github.com/huggingface/trl/pull/7249) already proposes token-normalization changes for AsyncGRPO and AsyncDistillation in response to [issue #7206](https://github.com/huggingface/trl/issues/7206). This VARE result is a deeper reproduction of the existing AsyncGRPO candidate through the Trainer/Accelerate/two-process DDP update path. It is not a novel algorithm, an original fix, or a new upstream contribution. At the last check the PR remained open and required review; no acceptance or adoption is claimed.

The fixture is synthetic and deterministic. This experiment does not measure a pretrained policy, independent downstream task success, capability improvement, asynchronous rollout scheduling, production dataloader dispatch, AsyncDistillation, MoE auxiliary loss, vLLM, GPU behavior, or external human reproduction. The v4 mismatch also shows that frozen fixture values must be asserted from constructed tensors, not inferred from their intended nested layout.

**Decision: ACCEPT the v5 reproduction for this narrow Trainer/Accelerate/DDP objective; STOP further variants unless an upstream reviewer identifies a concrete unresolved path.** Preserve v4 as a protocol non-pass. The central model-level gap remains independently graded task-success improvement after a real policy update; the latest ARC GRPO/SFT comparison failed its advancement criterion.

## Verification record

- Frozen v5 protocol and source hashes are retained in [`protocols/trl_async_accumulation_normalization_pr7249_full_trainer_ddp_v5.lock.json`](../protocols/trl_async_accumulation_normalization_pr7249_full_trainer_ddp_v5.lock.json).
- Reproduction wrapper exit code: 0; both source checkouts were clean and pinned; all frozen layout and numeric gates passed.
- A separate read-only audit by another agent in the same task environment verified the protocol/runner/wrapper, source, manifest, and summary hashes; all four fixture layouts; all frozen numeric gates; one optimizer step per rank; and matching DDP post-step state hashes. The audit did not rerun either arm and is not outside human reproduction. See [audit record](../results/trl-async-pr7249-full-trainer-ddp-v5/run-1/independent-audit.json).
- VARE GitHub Actions for freezing commit `70751bf6f467648b81c0a948de3025e28dc2b549` completed successfully (run [37911628655](https://github.com/mitukx/VARE/actions/runs/37911628655)); tests, exact GRPO witnesses, partial-audit recomputation, and demo all passed.
