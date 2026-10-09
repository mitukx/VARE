# TRL AsyncGRPO accumulation normalization through DataLoaderDispatcher — v4

**Result:** the frozen CPU integration run reproduced the pinned base's accumulation-window error after passing real `AsyncGRPOTrainer.DataCollatorForRollout` output through Accelerate's `DataLoaderDispatcher`, Transformers `Trainer.train()`, and two-process Gloo DDP. On the first update, which contained one globally empty microbatch followed by a five-token microbatch with one zero-token rank, base loss error was `6.05` and maximum gradient error was `3.025`. The existing PR #7249 candidate matched the pooled-token oracle: maximum loss/gradient error across both updates was `9.5367e-7`, and final parameter error was zero.

This is an upstream-candidate reproduction with a deterministic synthetic model. It is not a new algorithm, an original VARE fix, a pretrained-model result, or evidence of downstream capability gain.

## Question and estimand

Does PR #7249's shared accumulation-window token denominator preserve the token-mean clipped-GRPO update through the actual AsyncGRPO collator and Accelerate dispatch path, including an empty microbatch, a zero-token local rank, and a short final accumulation window?

For each optimizer update, the reference pools the masked clipped-GRPO token losses over ranks and microbatches and divides by the total active completion-token count. It compares the resulting loss, DDP-synchronized gradient, and SGD-updated parameters. The schedule uses SGD learning rates `0.1` and `0.05` on its two successive updates, with no weight decay or clipping.

## Frozen protocol and source

Protocol v4 was committed before execution. It pins TRL base `a98fa6a4428f9aae58dfb26d729d7437f662f27a` and PR #7249 head `5234eb7c70f4ca7eb92fe8e01611a33eaff17f40`, along with the source-file hashes and Python 3.12.12 / PyTorch 2.9.1 / Transformers 4.57.3 / Accelerate 1.12.0 runtime.

The three global microbatches have completion-token counts `[0, 5, 7]`. The first optimizer window contains counts `[0, 5]`; rank 0 has zero completion tokens in both microbatches, while rank 1 contributes five in the second. The final window contains only the seven-token microbatch. The stream declares length three so Trainer's epoch boundary and short final window are explicit.

The runner uses the pinned production `DataCollatorForRollout`, prepares an `IterableDataset` with `split_batches=True` and `dispatch_batches=True`, asserts that Accelerate returned `DataLoaderDispatcher`, and executes the Trainer optimizer/scheduler loop on two Gloo ranks. Both ranks observed the locked local and replicated global token counts and completed two optimizer updates.

## Results

| Arm | First-window loss error | First-window gradient error | Final-window loss error | Final-window gradient error | Final parameter error | Rank states |
|---|---:|---:|---:|---:|---:|---|
| Pinned base | `6.05` | `3.025` | `9.54e-7` | `0` | `0.3025` | equal within arm |
| PR #7249 candidate | `0` | `0` | `9.54e-7` | `0` | `0` | equal within arm |

The base's first-window mean loss was `6.05` while the pooled oracle was `12.10`; the global empty microbatch diluted the update under per-microbatch scaling. The candidate matched the oracle on that window and on the final short window. The final base parameter discrepancy is the retained effect of the first update; the second update's gradient matched the oracle.

Reproduce with:

```bash
python scripts/reproduce_trl_pr7249_dispatcher_v4.py \
  --output results/trl-pr7249-dispatcher-v4/run-N
```

The [protocol](../protocols/trl_pr7249_dispatcher_v4.lock.json), [runner](../scripts/replay_trl_pr7249_dispatcher_v4.py), [wrapper](../scripts/reproduce_trl_pr7249_dispatcher_v4.py), [manifest](../results/trl-pr7249-dispatcher-v4/run-1/manifest.json), per-arm summaries, and launcher logs retain the raw evidence. V1 and v2 wrapper preflight failures and the v3 unsized-stream non-pass are preserved in their separate run directories; none is counted as a scientific result. A same-task read-only audit matched protocol/source/runner/wrapper/log/summary hashes and independently checked the recorded counts and numeric gates. It noted that the wrapper does not itself enforce every locked finite-value, both-arm step-count, and sample-ID assertion; the saved v4 outputs satisfy the first two, while IDs are verified from the runner definition rather than result telemetry. See the [audit record](../results/trl-pr7249-dispatcher-v4/run-1/independent-audit.json).

## Novelty, limitations, and decision

**Novelty: low.** The denominator correction is already proposed in open [TRL PR #7249](https://github.com/huggingface/trl/pull/7249), which responds to [issue #7206](https://github.com/huggingface/trl/issues/7206). This result extends local evidence from rank-local Trainer inputs to the production rollout collator and Accelerate dispatcher boundary. It does not establish a new method or upstream adoption.

The async queue, rollout worker, live planner/batcher, pretrained model, FSDP, and accelerator hardware are not exercised. The model and examples are deterministic fixtures; loss/gradient correction is trainer-correctness evidence, not a task-success or capability result. The same-task audit is not outside human reproduction, and the wrapper's incomplete gate enforcement limits one-command auditability. At the repository state check on 2026-10-09, PR #7249 remained open without reviews.

**Decision: accept this narrow integration reproduction and stop local variants for PR #7249.** The useful next step is an outside maintainer/researcher review of the frozen packet. Keep real-model learning gated until a materially distinct CPU-feasible task/update pair clears existing feasibility requirements. Do not add a VARE-side duplicate of the already proposed upstream fix.
