# TRL vLLM truncation support audit v1

**Result: exact counterexample supported; no trainer defect or novelty claim.**
At the pinned TRL source revision, the vLLM path uses processed sampling
log-probabilities and compares them with full-vocabulary training log-probs.
For top-p, top-k, or min-p, the source ratio is exactly `p/q` on sampled
support. That ratio is mathematically valid there. It cannot recover arbitrary
raw-policy expectations when truncation gives `q=0` to actions with `p>0`.
This distinction narrows the claim in open TRL issue #6789; a nonzero
`sampling_logp_difference` alone is not evidence of harmful training.

**Protocol:** [`trl_vllm_truncated_support_audit_v1.lock.json`](../protocols/trl_vllm_truncated_support_audit_v1.lock.json)

**Accepted run:** [`run-4`](../results/trl-vllm-truncated-support-audit-v1/run-4/)

**Prior runner/auditor failures:** [`run-1`–`run-3`](../results/trl-vllm-truncated-support-audit-v1/)

## Question and source

The frozen question was: at zero parameter drift, what ratio does the current
TRL implementation compute between raw policy `p` and processed vLLM behavior
`q`, and can samples from `q` recover `E_p[R]` for every bounded reward after
the sampler removes part of `p`'s support?

This targets the reproduction request in [TRL issue #6789](https://github.com/huggingface/trl/issues/6789),
which remains open as of this source check. The exact source snapshot is TRL
`ed8cc2f4337fb9b7b1429db31009ad3577cd5b98`. SHA-256 locks cover
`grpo_trainer.py`, `grpo_config.py`, and `vllm_generation.py`; the primary
runner rejects any other files. Source inspection confirms that vLLM is
configured for `processed_logprobs` and receives top-p/top-k/min-p settings,
while the training-side log-probability is computed over the full vocabulary.
When vLLM importance-sampling correction is enabled (the pinned config
defaults it on), the source forms `exp(log p - log q)` and applies it to
non-VESPO policy losses. VESPO incorporates the correction in its sequence
weighting path.

## Exact estimand and result

Let `S` be the tokens retained by the sampler, `Z = sum_{a in S} p(a)`, and
`q(a) = p(a)/Z` for `a in S` and zero otherwise. On every sampled token,

```text
p(a) / q(a) = Z
E_q[p(a) / q(a)] = Z
|log p(a) - log q(a)| = -log Z
```

The exact rational calculations and a second integer-only implementation
agreed:

| Raw policy | Transform | Retained mass `Z` | Processed `q` | Source ratio on sampled actions | Absolute log-prob gap |
|---|---|---:|---|---:|---:|
| `(1/2, 3/10, 1/5)` | none | `1` | `(1/2, 3/10, 1/5)` | `1` | `0` |
| `(1/2, 3/10, 1/5)` | top-p `3/4`, top-k `2`, or min-p `1/2` | `4/5` | `(5/8, 3/8, 0)` | `4/5` | `0.22314` |
| `(7/10, 1/5, 1/10)` | top-p `3/4` or top-k `2` | `9/10` | `(7/9, 2/9, 0)` | `9/10` | `0.10536` |
| `(7/10, 1/5, 1/10)` | min-p `1/2` | `7/10` | `(1, 0, 0)` | `7/10` | `0.35667` |

For the first distribution and reward `R=(0,0,1)`, top-p excludes the rewarded
action. The raw-policy expectation is `E_p[R]=1/5`; the exact corrected
expectation from `q` samples is zero. With a softmax parameter `theta` added to
the third logit, the raw-policy gradient at `theta=0` is `p_3(1-p_3)=4/25`,
while the sampled score-gradient estimate is zero because that action is never
sampled. Two identical top-p decisions have sequence-level ratio `(4/5)^2 =
16/25` before clipping. The second independent checker reconstructs every
support, processed probability, ratio, expected reward, gradient, log-prob gap,
and the sequence product from integer weights; it imports no code from the
primary runner.

## Interpretation and claim limits

The non-unit ratio is not itself an error: `p/q` is the correct likelihood
ratio on the overlap between raw target `p` and behavior `q`. The failure is
support coverage. Since `q` assigns zero probability to excluded tokens,
importance weighting cannot recover `E_p[R]` for all rewards without an
additional support assumption or a different sampling policy. If the intended
target is the processed policy `q`, the policy objective and its gradient must
instead be defined consistently for that transformed distribution. The issue's
log-probability-gap metric is expected to move by `-log Z` in this controlled
case, but that metric alone does not show task harm.

This analysis is an exact finite-support counterexample, not an execution of
vLLM or `GRPOTrainer`, a measured real-model update, or a task-success result.
The support-overlap limitation is established prior probability theory and is
already discussed in the upstream issue; no novel algorithm is claimed. An
independent internal source/math review agreed with the narrow interpretation,
but did not constitute outside reproduction.

Two early harness failures are retained rather than erased: run-1 recorded an
incorrect zero score-gradient for the identity control; run-2's independent
checker conflated the raw gradient with the sampled estimate. Run-3 passed the
then-current partial cross-check. Run-4 passed the complete frozen check and is
the accepted result.

**Decision: stop broad claims that the ratio itself is biased.** Keep the
support-mismatch counterexample as a diagnostic for the open issue. Any proposed
source change must first name the intended target distribution and be checked
against a full-support or explicitly support-restricted estimand. Do not call a
metric change or this exact synthetic example a model improvement.

## Reproduction

The calculation uses only Python's standard library. Fetch the three source
files at the pinned revision into a normal TRL checkout, then run from the VARE
root:

```bash
python3 scripts/audit_trl_vllm_truncated_support.py \
  --source-root /path/to/trl \
  --output /tmp/trl-vllm-support-result.json
python3 scripts/verify_trl_vllm_truncated_support.py \
  --result /tmp/trl-vllm-support-result.json \
  --output /tmp/trl-vllm-support-independent-check.json
```

Both outputs should report `status: pass`. The source revision, file hashes,
frozen protocol snapshot, run commit, runner hashes, raw JSON outputs, and
SHA-256 manifests are retained under `results/trl-vllm-truncated-support-audit-v1/`.
