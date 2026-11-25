# Metal numerical path isolation v4

The v3 diagnostic measured four mismatching sampled-vs-differentiated tokens
out of four in each Metal precision arm, while Linux CPU q4 matched exactly.
It did not isolate the responsible operation. This new prospective diagnostic
tests actual Metal without task prompts or generated completions.

The first control probes seeded grouped-query attention directly. The second
uses the pinned 135M checkpoint and its existing evaluation-mode LoRA
configuration, with two fixed token sequences and one fixed target per
sequence. Compare stock fused attention vs an explicit FP32 matmul/softmax
reference, crossed with selected-row vs full vocabulary projection. Each
case materializes the plain path separately from its VJP, and compares
a teacher-forced cached prefill of the same sequence. Nonzero finite gradient
and unchanged full parameter hashes are required. An intervention is scoped
to the diagnostic context and restored even on failure.

The strict 1e-6 score threshold is unchanged. Fixed-target cached scores are
not recorded behavior denominators; even full success does not establish
sampled multi-action agreement or authorize changing the trainer. The
explicit attention implementation may change the numerical policy; the
cross-mode plain-score differences are reported. The raw isolated attention
output differences are not nats. The two arms are FP32/BF16 on registered
Apple Silicon only, with 300-second measurement caps and no GPU fallback.
Neither task data nor training is used. The prior q4 nonpass remains.

The diagnostic is exploratory mechanism isolation, not a claim of
causality for every layer, an upstream kernel bug, or a capability result.
A promising path requires a fresh sampled-cohort diagnostic before any
production score-mode change. The frozen protocol is
[grpo_metal_path_isolation_v4.json](../protocols/grpo_metal_path_isolation_v4.json).

## Measured outcome

Both arms completed at `faad1de0acd7c20c24d187f89d5d93cdd91d3c8b`. BF16 explicit paths pass the two fixed-target score checks; FP32 has no complete passing path at 1e-6. See the [retained exact ZIPs and path table](../results/grpo-metal-path-isolation-v4/README.md). No generated trajectories or optimizer updates were run. The initial pre-measurement test error is preserved and automatic repeat is disabled.