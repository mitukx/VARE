# TRL GRPO KL clip precision patch v4

**Status: frozen CPU source-method audit passed; upstream candidate remains unreviewed.**

## Question

With token-level importance sampling and bias-corrected KL enabled, can the pinned TRL production loss produce a non-finite aggregate KL metric and an incorrect fp16 gradient even when every active per-token clipped KL value is finite? Does the single-file precision candidate correct both while preserving fp32 behavior?

## Protocol and estimand

The source is TRL PR #6637 head `0aaea03f2fa449bc7a91f1973e7940da11da65da`. The frozen CPU fixture has two sequences with 2 and 3 active tokens, respectively; every active reference-policy log-ratio is 20, the cap is 10, the policy-old log-ratio is 0.25, and the masked padding token has ratio zero. The bias-correction coefficient is 0.1. The protocol was committed before execution: [lock](../protocols/trl_grpo_kl_clip_precision_patch_v4.lock.json).

For one active token, `K = exp(10) - 10 - 1` and `r = exp(0.25)`. The corrected KL contribution has derivative `-r × 10` with respect to its policy log-probability. Under the production per-sequence masked mean and batch mean, the expected token gradients are `-0.1 × r × 10 / (2 × n)`: `−0.321006` for each token in the length-2 sequence and `−0.214004` for each token in the length-3 sequence.

## Results

The unmodified production method completed, but its fp16 aggregate KL metric was `Infinity`. In fp16, each corrected clipped KL value was finite (`28,272`); summing five active values in the production metric reduction exceeded the maximum fp16 value of `65,504`. The same run's gradients were `−0.5` for the two active tokens in the first sequence and `−0.25` for the three in the second, outside the frozen analytic tolerance.

The candidate returned a finite KL metric (`28,271.8125`), finite loss (`2,827.1814` versus analytic `2,826.8418`), and gradients `−0.321045` and `−0.213989`, within the frozen tolerance. The masked padding gradient was zero. Candidate and unmodified PR head were bit-identical for all tested fp32 controls. The independent checker uses only standard-library arithmetic and retained JSON; it reported **pass, no errors**. All raw runs and hashes are retained in [run-1](../results/trl-grpo-kl-clip-precision-patch-v4/run-1/).

## Interpretation and limits

This is a reproducible, local finding in an actual upstream loss method, with an explicit finite-precision failure and a candidate correction on a fresh multi-token fixture. It is stronger than the prior one-token illustration and resolves the v2/v3 protocol mistakes by freezing the exact active values, padding value, expected baseline behavior, and candidate-specific acceptance checks.

It remains a source-method numerical result, not a full `GRPOTrainer` optimizer step, real-model occurrence-rate estimate, task-success gain, external reproduction, or upstream acceptance. In the pinned TRL config, `beta` defaults to zero, disabling this KL path; the PR discussion also contains no real training configuration that reproduces the extreme ratio. Therefore practical prevalence and impact are unknown. The tested gradient defect requires the opt-in KL path and the specified fp16 regime. No bf16, GPU, fused/Liger, or distributed path was tested.

The v2 and v3 non-passes remain intact. Their reports now disclose, respectively, an incompletely frozen padding input and a runner/protocol log-probability mismatch. Neither is counted as confirmation evidence.

## Decision

**Continue to an upstream-quality review artifact, not a capability claim.** Add a focused regression test against the pinned production method and reproduce it from a clean checkout. Then have an independent reviewer inspect the exact diff and test. Given the disabled-by-default KL path and absent real configuration, do not claim broad impact or post an upstream message until the reproduction and review artifact are complete.

## Independent review correction

The v4 lock wrote the KL loss contribution with an extra factor of `beta`, while the checker, one-beta gradient formula, and production method use a single factor. Thus the checker did not test the literal locked loss estimand: its expected loss was ten times the literal protocol value for `beta=0.1`. The raw baseline overflow, gradient discrepancy, candidate arithmetic, and fp32 parity were independently verified, but **v4 is a protocol non-pass**, not a frozen-protocol pass. The full review adjudication is in `review-adjudication.json` in the run bundle. A fresh protocol must state the production loss exactly and tie fixture constants mechanically to execution.
