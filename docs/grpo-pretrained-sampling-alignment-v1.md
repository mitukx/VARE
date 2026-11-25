# Pretrained sampling-to-gradient alignment (v1)

## Question and bounded experiment

The strict lag-zero guard rejects a recorded token if its sampling-time
log-probability differs from the differentiated training forward by more
than 1e-6 nat. A toy-model regression is not sufficient evidence that the
guard accepts real pretrained-model traces. This prospective numerical
diagnostic uses the exact collector and differentiated trainer paths from
PR #206, on a small public checkpoint, without any optimizer update or task
success measurement.

The [prospective protocol](../protocols/grpo_pretrained_sampling_alignment_v1.json)
fixes the model revision, weight SHA-256, precision arms (FP32, BF16 and
4-bit affine weight quantization), two neutral context bands, two seeds,
two independently conditioned actions per member, positive temperature,
untruncated sampling, LoRA initialization and 1e-6 threshold. Each arm
generates at most 64 completion tokens; no task dataset is read. The
last-layer rank-4 adapter starts with zero output and dropout zero, and
the pretrained model and adapter bytes are hashed before and after. A
different model or threshold requires a new protocol rather than a changed
result. Both mismatch and agreement are reportable outcomes.

The main audit compares sampled scores with the ordinary and differentiated
trainer scores. A separate arithmetic control recreates native-dtype sampler
normalization on the same uncached trainer projection: it distinguishes
same-logits normalization/temperature differences from differences between
cached decoding and uncached forward (which also include kernel/shape effects).
The latter is not a proof that KV cache alone caused an error.

This is a real-weight numerical experiment on synthetic, task-free prompts;
it is not a model rollout on a coding task, an admissible training cohort,
a held-out comparison, or an evidence of learning. Dummy rewards are all
zero and would not produce a GRPO update. The strict-gate rejection metric
is hypothetical if the sampled tokens were selected into an informative
update. Different precision arms can generate different tokens, so the
arm-level measurements describe within-arm consistency, not a paired
quality or accuracy comparison.

## Reproduce

The dedicated GitHub Actions workflow runs the three CPU arms independently.
It checks out the exact head, uses the existing pinned MLX runtime, downloads
only the public checkpoint and tokenizer files, checks the weight hash,
then executes the diagnostic with offline model access and a wall timeout.
The script records source SHA, protocol hash, snapshot hashes, runtime,
exact neutral prompt/completion tokens, sampled probabilities, arithmetic
controls, and pre/post parameter digests. Numerical mismatch alone does not
fail a job; setup or incomplete execution does. Logs and JSON artifacts are
retained. No repository write permissions, secret model access, GPU or paid
model API is required.

From a clean Linux checkout with the pinned environment:

~~~sh
python -m pip install -e . -r requirements/grpo-trajectory-cpu-test.txt
python scripts/diagnose_grpo_pretrained_sampling.py download \
  --snapshot-path-file /tmp/vare-alignment-snapshot.txt
HF_HUB_OFFLINE=1 python scripts/diagnose_grpo_pretrained_sampling.py run \
  --snapshot "$(cat /tmp/vare-alignment-snapshot.txt)" \
  --arm fp32 --device cpu --output /tmp/vare-alignment-fp32.json
~~~

Use distinct fresh output paths for BF16 and q4. For a compatible Apple
Silicon runtime, use the same version pins excluding the Linux-specific
mlx-cpu package, then select --device gpu (no CPU fallback). GPU results
must be measured and reported separately. The diagnostic never authorizes
training or changes the frozen readiness decision.

## Evidence

The protocol is prospective. See the workflow's exact-head logs and JSON
artifacts for executed or failed arms. No result is asserted by this document
until the corresponding evidence is retained.
