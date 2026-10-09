# TRL GRPO KL clip precision patch v1

**Status: frozen CPU source-path fixture passed; patch remains an unreviewed candidate.**

## Question

Can a minimal patch to the open TRL GRPO KL-log-ratio clipping change reject a cap that is not representable in the working dtype and preserve the clipped, bias-corrected KL gradient in low precision, while leaving float32 and unclipped controls unchanged?

The source was pinned to PR #6637 head `0aaea03f2fa449bc7a91f1973e7940da11da65da` and base `2b0d16b7839732f0b652ea7dfd4f492f00e15a48`. The protocol was frozen in commit `4771bda` before the patch was executed: [protocol](../protocols/trl_grpo_kl_clip_precision_patch_v1.lock.json).

## Method and frozen result

The runner directly invokes the production `GRPOTrainer._compute_loss` method with a one-completion-token differentiable fixture (`x=20`, `r=1`, `beta=0.1`, zero policy advantage). It uses CPU and no model, dataset, optimizer, GPU, Liger kernel, or external service. The candidate changes only `trl/trainer/grpo_trainer.py`; patch SHA-256 is `0f6c60de679c3c8b467b2bc8b77b01e06f890486cd567322940e5ce8a2137468`.

| Frozen case | PR head | Candidate | Outcome |
|---|---:|---:|---|
| fp16, clip `1e-8`, bias correction on | returned zero penalty/gradient in the prior audit | raises `ValueError` because cap is not representable | pass |
| fp16, clip `10`, bias correction off | loss 2202.0, gradient −2202.0 | loss 2201.546630859375, gradient −2202.0 | pass within frozen tolerance |
| fp16, clip `10`, bias correction on | prior audit found gradient 0 | loss 2201.546630859375, gradient −1.0 | pass; analytic target −1.0 |
| fp32, clip `10`, bias correction on | loss 2201.546630859375, gradient −0.999755859375 | identical | pass |
| fp32, no clip, bias correction on | loss 48516516.0, gradient −4.0 | identical | pass |

The source-independent checker read only the retained JSON outputs and patch bytes; it did not import TRL. It reported **5/5 protocol gates passed**, no errors, and confirmed the frozen patch hash. The eight raw method executions and checker are in [run-2](../results/trl-grpo-kl-clip-precision-patch-v1/run-2/).

## Interpretation

This establishes a narrow numerical result for the frozen one-token production-loss fixture. The candidate repairs the two observed fp16 behaviors in that fixture and preserves the tested fp32 controls. It is evidence for a reviewable patch proposal, not proof that the issue affects ordinary training runs, not an end-to-end optimizer result, and not a model capability gain.

The first invocation, [run-1](../results/trl-grpo-kl-clip-precision-patch-v1/run-1/), is retained as an invalid CLI attempt: the runner rejected a Boolean flag passed as a string before any case executed. It is not an experiment result and was not silently discarded.

## Limits

- Only token-level importance correction and one completion token were covered; sequence-level importance correction is untested.
- No `bfloat16`, multi-token masking, batched reductions, GPU, fused/Liger implementation, optimizer step, or real model was exercised.
- The numerical effect's prevalence in actual training is unknown; no task outcome or independent external reproduction exists.
- The open PR head is not a released TRL trainer. This patch is not upstreamed, accepted, or maintainer-reviewed.

## Decision

**Continue narrowly; stop broadening the claim.** The patch clears the preregistered local numerical gate and justifies a distinct sequence-level/multi-token audit before proposing an upstream change. Any upstream submission should include the minimal reproduction, dtype-safe fix, and tests against the pinned source; no message or PR has been sent as part of this run.

## Reproduction

From the repository root, with the exact PR-head checkout at `/tmp/trl-grpo-kl-candidate-v1` and clean control at `/tmp/trl-grpo-kl-head-control-v1`:

```bash
python scripts/audit_trl_grpo_kl_clip_precision_patch.py --source-root /tmp/trl-grpo-kl-head-control-v1 --output results/trl-grpo-kl-clip-precision-patch-v1/run-2/pr-head-f16-c10-bias-off.json --dtype float16 --clip 10 --no-bias-correction
```

The full eight-command matrix is preserved in the run bundle. Recheck the retained evidence with:

```bash
python scripts/verify_trl_grpo_kl_clip_precision_patch.py --run-dir results/trl-grpo-kl-clip-precision-patch-v1/run-2 --output results/trl-grpo-kl-clip-precision-patch-v1/run-2/independent-check.json
```
