# TRL GRPO KL clip precision patch v3

**Status: invalid protocol execution; no confirmation claim.**

The lock specified `z-z_old=0.125`, but the reused runner actually set policy log-probabilities to `0.25`. The lock also predicted that the unmodified head would produce a non-finite KL metric with active `x=16`, clip 8; this was not supported by the source path, which applies the configured cap before the K3 exponential. The run therefore failed the frozen baseline criterion and did not execute the locked input. The independent checker also found the candidate outputs outside the declared analytic expectations, consistent with the fixture mismatch.

An independent protocol review confirms the runner/lock mismatch. The execution is classified as invalid; retain it without treating it as a patch result. See `review-adjudication.json` in the run bundle.

The raw records are retained in [run-1](../results/trl-grpo-kl-clip-precision-patch-v3/run-1/), with the protocol in [the lock](../protocols/trl_grpo_kl_clip_precision_patch_v3.lock.json). Do not treat this run as evidence for or against the patch. The next test, if any, must use a runner whose literal inputs are mechanically tied to its frozen fixture and must avoid requiring an unmodified baseline to fail for an unsupported reason.
