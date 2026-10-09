# TRL GRPO KL-clip precision patch v5

**Status: frozen CPU source-method protocol passed; this is a narrow numerical finding, not model-improvement evidence.**

## Question

On a fresh, fully specified multi-token token-importance fixture, does a single-file precision candidate prevent the pinned TRL PR #6637 fp16 KL reduction/gradient failure, match an analytic target, mask padding, and preserve the tested fp32 path?

The protocol and scripts were frozen in VARE commit `c48a9b4` before execution. It pins TRL PR head `0aaea03f2fa449bc7a91f1973e7940da11da65da`, base `2b0d16b7839732f0b652ea7dfd4f492f00e15a48`, and candidate patch SHA-256 `0f6c60de679c3c8b467b2bc8b77b01e06f890486cd567322940e5ce8a2137468`.

## Protocol and estimand

The CPU fixture has two sequences of lengths 2 and 3, five active tokens, one masked pad, token importance ratio `exp(0.125)`, positive clipped KL log-ratios of 21, clip 10, `beta=0.1`, zero policy advantages, and bias correction enabled. The production method `_compute_loss` is invoked directly; no model, dataset, optimizer, GPU, or external service is used.

For active token `(b,t)`, the checker derives `r=exp(z−z_old)`, `u=min(x, c)`, `K=exp(u)−u−1`,

`L=(beta/B) sum_b mean_active(K*r)`,

and `dL/dz_bt = −beta*r*u/(B*n_b)`. The aggregate metric is the mean `K*r` over active tokens. The single beta factor and per-sequence masked mean match the pinned method. The padding entry has non-unit policy ratio but zero KL log-ratio and is excluded from reductions; its expected gradient is zero.

## Results

| Measurement | Unmodified PR head, fp16 | Candidate, fp16 | Analytic / control |
|---|---:|---:|---:|
| Loss | 2494.0 | 2493.9395 | 2494.6791 |
| Aggregate KL metric | `Infinity` | 24939.3945 | 24946.7910 |
| Gradients, length 2 | `[-0.5, -0.5]` | `[-0.283203, -0.283203]` | `[-0.283287, -0.283287]` |
| Gradients, length 3 | `[-0.25, -0.25, -0.25]` | `[-0.188843, -0.188843, -0.188843]` | `[-0.188858, -0.188858, -0.188858]` |
| Padding gradient | `0` | `0` | `0` |

The baseline's five finite per-token contributions overflowed when accumulated in fp16 and its active gradients differed from the analytic target. The candidate outputs were finite and within the frozen 0.5% relative / 0.1 absolute tolerance. The fp32 loss, active gradients, and metric were bit-identical between base and candidate. A separate standard-library checker verified source, patch, runner, checker, fixture, and protocol digests and all frozen gates: **pass**. Raw outputs and exact commands are in [run-1](../results/trl-grpo-kl-clip-precision-patch-v5/run-1/); the protocol is [locked here](../protocols/trl_grpo_kl_clip_precision_patch_v5.lock.json).

## Novelty and external relevance

This is not a new RL algorithm or general GRPO result. It is a reproducible finite-precision correctness counterexample against an open upstream implementation, plus evidence that a narrow precision candidate repairs this exact path. The closest upstream context is [TRL PR #6637](https://github.com/huggingface/trl/pull/6637), which adds optional KL log-ratio clipping. Its discussion says the path is disabled by default (`beta=0`) and does not provide a real training configuration reproducing the extreme drift. Therefore this fixture establishes conditional method behavior, not prevalence or practical training impact. The candidate is not upstreamed or maintainer-reviewed.

The earlier v2, v3, and v4 runs are not confirmations: v2 omitted its masked-padding value; v3 executed policy log-probabilities different from its lock; v4's frozen loss equation contained an extra beta factor. Their raw records and corrections remain preserved. V1 is a valid but one-token diagnostic; v5 is the first valid multi-token confirmation under the corrected fixture.

## Limits

- One hand-built fixture and one source-method call; no optimizer update, training prevalence, downstream task success, or capability gain.
- No bf16, GPU, Liger/fused, distributed, or external reproduction.
- The KL path is opt-in in the pinned config and its practical frequency is unknown.
- The candidate is an unreviewed local patch; this result alone does not justify a broad defect claim.

## Decision

**PIVOT away from expanding this patch study.** Preserve this as a reviewable source-method finding. The highest-value next action is an upstream-quality minimal regression/test proposal only if it can be tied to a supported configuration and maintainers confirm the opt-in path merits maintenance; otherwise stop this low-prevalence line and return to a distinct correctness defect with an ordinary reachable configuration or to independent reproduction of existing VARE evidence. Do not claim capability improvement.

## Reproduction

The exact four runner invocations and independent checker command are recorded in `commands.txt` in the run bundle. From the VARE root, run:

```bash
python scripts/verify_trl_grpo_kl_clip_precision_patch_v5.py \
  --run-dir results/trl-grpo-kl-clip-precision-patch-v5/run-1 \
  --output results/trl-grpo-kl-clip-precision-patch-v5/run-1/independent-check.json
```

## Independent audit

A read-only post-run subagent audit recomputed the analytic values and checked the frozen commit, run snapshot, source and patch hashes, fixture digests, and all four records. It found no material mismatch and agrees with the narrow result and pivot decision. This is an internal independent audit, not external human reproduction; the adjudication is retained in [the run bundle](../results/trl-grpo-kl-clip-precision-patch-v5/run-1/review-adjudication.json).
