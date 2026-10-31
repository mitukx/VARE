# GRPO sampling-distribution decision v1 (directive C1): Option A, untruncated sampling

Status: decided before freezing the next training protocol. Applies to the MLX
exact-ratio GRPO path (`src/vare/learning_pipeline/grpo_mlx.py`,
`scripts/grpo_vertical_slice.py`) and to every new protocol that feeds it.

## Problem

The first Qwen3-4B GRPO smoke sampled with temperature 0.7 and top-p 0.95, but
the objective computed behavior and current log-probabilities as
`log_softmax(logits / T)` over the full vocabulary. A top-p sampler draws from
the nucleus-renormalized distribution, so for those samples `logp_behavior` is
not the log-probability of the policy that generated the token, and
`rho = exp(logp_current - logp_behavior)` is not the exact on-policy ratio.

## Decision: Option A

Sample from the untruncated temperature distribution only:
`temperature > 0`, `top_p == 1.0`, `top_k == 0`, `min_p == 0`. With mlx-lm
0.32.2, `make_sampler(temp=T, top_p=1.0)` adds no filter and draws
`categorical(logprobs / T)`, i.e. exactly `softmax(logits / T)`, which is the
distribution `grpo_mlx.token_logprobs(..., temperature=T)` computes. Behavior
and current log-probabilities are therefore both under the sampled
distribution, and the ratio is exact (behavior log-probs are recomputed on the
same differentiated path, so `rho == 1` holds numerically at the update point).

Option B (modeling the truncated distribution, including the nucleus mask's
dependence on the current policy) was not chosen: it changes the objective and
needs its own validated mathematics; Option A removes the mismatch instead.

## Enforcement

* `grpo_mlx.check_exact_ratio_sampling` raises `TruncatedSamplingError` for
  `top_p != 1`, `top_k != 0`, `min_p != 0`, greedy decoding, or non-finite
  settings.
* `scripts/grpo_vertical_slice.py` calls it when a run is constructed (before
  any run directory exists) and again inside `generate`; `--top-p` defaults to
  1.0 and any other value is refused. The run metadata records the sampling
  distribution.
* Tests: `tests/test_grpo_mlx.py` (truncated/greedy settings rejected; the
  slice refuses `--top-p 0.95`; with MLX installed, the top_p=1.0 sampler is
  checked draw-for-draw against `categorical(logprobs / T)`).

## Labels for earlier evidence (artifacts unchanged)

These runs sampled with top_p 0.95. Their frozen artifacts are not edited or
relabelled in place; this document is the label:

| run | what it is now |
|---|---|
| `results/grpo-vertical-slice/smoke-001` | approximate-ratio evidence: real parameter updates, but behavior log-probs ignored the nucleus truncation |
| `results/grpo-group-informativeness-v1` | approximate-ratio evidence: group statistics of the top_p 0.95 sampling distribution, not the exact-ratio training distribution |
| `protocols/grpo_train_pool_probe_v1.json` run | diagnostic only, same top_p 0.95 caveat; rewards are additionally provisional (v1 native grader, pending trusted-grader requalification) |

New probes and training smokes use top_p 1.0. Mixed-group rates measured at
top_p 0.95 are expected to differ somewhat at top_p 1.0 (the untruncated tail
adds lower-probability tokens) and must be re-measured, not carried over.
