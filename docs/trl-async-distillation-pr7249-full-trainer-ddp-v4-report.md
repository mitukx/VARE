# TRL AsyncDistillation token normalization through Trainer/DDP — v4

**Result:** the frozen CPU reproduction found the pinned base diverged from the token-pooled forward-KL objective when completion-token counts differed by accumulation microbatch. In the unequal primary case, maximum gradient error was `4.4950e-3` for base and `1.4901e-8` for the existing PR #7249 candidate. The candidate passed all four cases under the frozen `1e-5` threshold; equal-token and short-final-window controls passed on both revisions.

This extends VARE's v5 AsyncGRPO full-Trainer/DDP reproduction to the distinct AsyncDistillation implementation. It is evidence about a deterministic synthetic Trainer/Accelerate/Gloo path, not a new method or a model-capability result.

## Question and estimand

For beta-zero sparse top-k AsyncDistillation, do the pinned base and PR #7249 candidate produce the same gradient and one-step SGD update as pooling all active completion tokens across ranks and gradient-accumulation microbatches?

The independently coded reference objective is

\[
L(\theta)=\frac{1}{N}\sum_i n_i\,\mathrm{KL}(q_i\Vert p_\theta),\qquad N=\sum_i n_i,
\]

where `n_i` counts active completion positions and `q_i` is a normalized distribution over the model's two-token vocabulary, fully represented by teacher top-k IDs `[0, 1]`. The model is deterministic and token-local, with a shared trainable embedding and linear output head. The comparison uses one SGD step (`lr=0.1`), beta `0`, teacher temperature `1`, no tail bucket, no weight decay, and no gradient clipping.

## Prior art and novelty

TRL [issue #7206](https://github.com/huggingface/trl/issues/7206) and [PR #7249](https://github.com/huggingface/trl/pull/7249) concern token normalization for both AsyncGRPO and AsyncDistillation. The PR's added test file exercises each trainer and `Trainer.train`; its rank comparison simulates averaging two rank gradients in one process. VARE's earlier v5 run exercised actual two-process Gloo DDP for AsyncGRPO but not AsyncDistillation. V4 covers this omitted trainer path with actual Trainer/Accelerate/Gloo execution and an independently coded pooled objective.

**Novelty assessment: low.** This is a deeper reproduction of an existing upstream candidate, not an original correction or algorithm. It may help reviewers assess the candidate's second trainer path. At the last state check, PR #7249 remained open and review-required; no upstream adoption is claimed.

## Frozen protocol and run history

V1's source checkout gate failed before either training arm: the candidate source digest in the lock differed from the pinned file at one character. V2 corrected that digest but its wrapper looked up a differently named wall-time field and exited before starting a runner. V3 started two CPU ranks but exposed a rank/microbatch indexing error during fixture validation, before model construction or an optimizer update. These are retained as protocol/preflight non-passes, not scientific outcomes. V4 freezes the corrected count traversal, source digest, launcher contract, and fresh sample IDs.

The v4 protocol was committed at `1a30a941d7bc3e3ea1b2c202c35af3c82ae7667f` before execution; its VARE GitHub Actions run [37912958656](https://github.com/mitukx/VARE/actions/runs/37912958656) passed. Protocol SHA-256: `edfcf8b3eadc2668c0e20faa74ef6a8bfed7535ac0283ee311605c5c8f65b5cd`. It pins TRL base `a98fa6a4428f9aae58dfb26d729d7437f662f27a`, PR head `5234eb7c70f4ca7eb92fe8e01611a33eaff17f40`, both AsyncDistillation source-file hashes, and Python 3.12.12 / PyTorch 2.9.1 / Transformers 4.57.3 / Accelerate 1.12.0.

Before any Trainer instance or optimizer step, v4 asserts rank-local and globally collated active-token counts against the lock:

| Case | Local tokens per microbatch (rank 0 / rank 1) | Global tokens |
| --- | --- | --- |
| Unequal primary | `[2, 7]` / `[9, 2]` | `[11, 9]` |
| Equal-token control | `[4, 6]` / `[6, 4]` | `[10, 10]` |
| Zero-local-token rank | `[0, 0]` / `[5, 4]` | `[5, 4]` |
| Short final window | `[3]` / `[8]` | `[11]` |

## Results

Maximum absolute errors over both ranks against the pooled two-token forward-KL oracle:

| Case | Base loss | Base gradient | Base update | Candidate loss | Candidate gradient | Candidate update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unequal primary | `3.3815e-3` | `4.4950e-3` | `4.4950e-4` | `2.4796e-8` | `1.4901e-8` | `1.8626e-9` |
| Equal-token control | `1.0943e-8` | `1.4901e-8` | `1.8626e-9` | `1.0943e-8` | `1.4901e-8` | `1.8626e-9` |
| Zero-local-token rank | `8.4037e-4` | `2.7778e-3` | `2.7778e-4` | `2.8871e-8` | `1.4901e-8` | `1.3970e-9` |
| Short final window | `2.0256e-8` | `2.9802e-8` | `2.7940e-9` | `2.0256e-8` | `2.9802e-8` | `2.7940e-9` |

All cases completed one optimizer step in each arm. Rank state hashes matched after every step. The base defect gate (`>=1e-4` gradient or update error) passed in the two uneven cases; both base controls and all candidate cases passed their frozen gates. Both source checkouts were clean and matched their pinned commits and source hashes.

Reproduce with a new output path:

```bash
python scripts/reproduce_trl_pr7249_async_distillation_ddp_v4.py \
  --output results/trl-async-pr7249-distillation-ddp-v1/run-N
```

The [v4 lock](../protocols/trl_async_accumulation_normalization_pr7249_async_distillation_ddp_v4.lock.json), [runner](../scripts/replay_trl_pr7249_async_distillation_ddp_v4.py), [wrapper](../scripts/reproduce_trl_pr7249_async_distillation_ddp_v4.py), [manifest and summaries](../results/trl-async-pr7249-distillation-ddp-v1/run-4/), and earlier [preflight failures](../results/trl-async-pr7249-distillation-ddp-v1/) retain provenance and raw evidence. The run used no GPU, paid API, or external spend. A separate read-only agent audit verified the lock/runner/wrapper, manifest, summary, and log hashes; fixture counts and gates; one update per rank; and matching state hashes. It did not rerun the experiment; the temporary clean checkouts are gone, so source cleanliness is evidenced by the retained manifest and the wrapper's pre-run checks. This is same-task audit, not outside human review. See the [audit record](../results/trl-async-pr7249-distillation-ddp-v1/run-4/independent-audit.json).

## Limits and decision

The experiment bypasses asynchronous rollout and teacher workers, queue/planner, production `DataLoaderDispatcher`, FSDP, and accelerator hardware. It uses a deterministic two-token synthetic model, one update per case, and constructed teacher distributions. It does not measure a pretrained model, real distillation data, downstream task success, generalization, throughput, upstream adoption, or capability improvement. It tests beta-zero full-support forward KL only; other beta values and missing/partial teacher support are outside scope.

**Decision: accept as a narrow AsyncDistillation Trainer/DDP reproduction and stop adding variants for PR #7249.** The PR is the existing fix candidate; this result adds path coverage but no novelty claim. The next useful step is outside review/reproduction or an independently graded model-level task-success experiment that first clears the repository's feasibility gates. Existing negative model studies do not justify another nearby model/task screen.
