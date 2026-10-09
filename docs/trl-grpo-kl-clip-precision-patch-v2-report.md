# TRL GRPO KL clip precision patch v2

**Status: protocol non-pass; candidate arithmetic checks pass on the frozen fixture.**

## Correction after source and protocol audit

The run-1 runner set the masked padding token's `x` to 80, but the v2 lock did not specify a padding value. In this source path, the KL metric reduction multiplies the per-token KL by the mask after computing the exponential, so a non-finite padded intermediate can contaminate a zero-masked reduction. The recorded `kl_metric=Infinity` therefore cannot be attributed solely to active-token behavior from this run. The v2 protocol also had an overbroad all-fixture-finite condition. Treat the run as **protocol invalid/non-pass** and its analytic candidate subchecks as diagnostics only; do not cite the infinite baseline metric as evidence of an active-token overflow. The original command and outputs remain unchanged.

## Question and method

The frozen protocol tested the single-file candidate on TRL PR #6637 head `0aaea03f2fa449bc7a91f1973e7940da11da65da`, with two sequences of active lengths 2 and 3, importance-sampling ratio `exp(0.25)`, clipped and unclipped K3 tokens, fp16/fp32, and token/sequence weighting. It directly called production `GRPOTrainer._compute_loss` on CPU. The analytic sequence objective and gradient are specified in the [locked protocol](../protocols/trl_grpo_kl_clip_precision_patch_v2.lock.json), frozen in commit `2d5aa62` before execution.

## Results

The independently computed fp16 sequence-level target loss was 1651.6446; the candidate returned 1651.8430. Active gradients from the candidate were `[-353.0, 352.25]` for the length-2 sequence and `[-157.25, -157.25, 313.0]` for the length-3 sequence. The analytic values were `[-352.8802, 352.4308]` and `[-156.9070, -156.9070, 313.3003]`; masked padding gradients were zero. Candidate token-level gradients also matched their separate analytic target. Candidate and unmodified PR head were bit-identical in the fp32 sequence-level loss, gradients, and KL metric.

The unmodified PR-head fp16 sequence control recorded `kl_metric=Infinity`; the candidate metric was finite (`16988.5664`). The source-independent checker confirmed the candidate calculations, patch identity, and fp32 control.

## Adjudication

The locked acceptance criterion said **all fixture outputs** must be finite. It did not exempt the intentionally unmodified defect baseline, so that criterion failed exactly as written. Classify the overall protocol as a **non-pass due to a protocol design error**. Do not relabel it as a pass after seeing the outcomes. The candidate-specific analytical checks are retained as diagnostic evidence only. See [run-1](../results/trl-grpo-kl-clip-precision-patch-v2/run-1/).

This is not an external reproduction, real optimizer result, or model outcome. It does not establish a general trainer fix. The reported PR discussion also says the KL regularization path is opt-in (`beta` defaults to zero) and the PR author lacks a real training configuration reproducing the extreme log-ratio. That limits expected prevalence and impact; the numerical source-path issue remains conditional on users enabling the path and matching the tested dtype/regime.

## Decision

**Continue once with a corrected, fresh protocol; do not reuse these consumed values as confirmation.** The next protocol must require finite/analytic outputs from the candidate and explicitly classify non-finite behavior in the unpatched baseline as the predicted defect. Use a new fixture. If the fresh result does not distinguish a real candidate benefit from quantization noise, stop the precision-patch line and prioritize a higher-impact reproducible trainer defect or independently evaluated model update.

## Independent review correction

A separate review found the v2 execution did not match a fully frozen fixture: the runner used masked padding `x=80`, but the lock omitted the padding value. Since non-finite intermediate values on a masked token can contaminate a low-precision reduction, the `Infinity` observation is confounded. The v2 run is now adjudicated **invalid protocol execution**, not merely a broadly gated non-pass. See `review-adjudication.json` in the run bundle.
