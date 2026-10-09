# TRL GRPO KL clip dtype audit v1

**Outcome:** the open TRL KL-overflow fix PR has two untested CPU float16 failure modes in its clipping path. A tiny positive clip is accepted by config, rounds to zero in float16, and yields zero KL and zero gradient. With the representable clip `10`, the K3 value is finite, but its bias-corrected gradient rounds to zero under float16 arithmetic; the expected gradient is `-1` for the frozen fixture. This is an independent source-path audit of an existing upstream fix, not a new algorithm or an estimate of real-run prevalence.

**Frozen protocol:** [`trl_grpo_kl_clip_dtype_underflow_v1.lock.json`](../protocols/trl_grpo_kl_clip_dtype_underflow_v1.lock.json)
**Accepted audit bundle:** [`run-4`](../results/trl-grpo-kl-clip-dtype-underflow-v1/run-4/)
**Retained attempts:** [`run-1`–`run-3`](../results/trl-grpo-kl-clip-dtype-underflow-v1/)

## Question and source revisions

The question was whether the opt-in K3 overflow guard in the open [TRL PR #6637](https://github.com/huggingface/trl/pull/6637) remains effective after the configured cap is converted to the policy log-probability dtype, and whether the bias-corrected gradient remains nonzero in float16. The PR adds `GRPOConfig.kl_log_ratio_clip` and applies a straight-through upper clamp in `GRPOTrainer._compute_loss`.

The exact base revision was `2b0d16b7839732f0b652ea7dfd4f492f00e15a48`; the exact PR head was `0aaea03f2fa449bc7a91f1973e7940da11da65da`. The runner rejects dirty source checkouts and records hashes for `grpo_config.py` and `grpo_trainer.py`. Both pinned worktrees were clean. The PR remains open at the time of this audit.

## Frozen CPU method fixture

The runner imports and directly invokes the unmodified production `GRPOTrainer._compute_loss` with a one-token float16 tensor fixture. The current token log-probability equals the old log-probability, making the policy importance ratio exactly one; the reference log-probability is 20 nats higher, advantages are zero, and `beta=0.1`. The production K3 term is therefore isolated. Both settings of `use_bias_correction_kl` are recorded. No model or dataset is downloaded.

| Source and cap | Config result | KL metric | Loss | Loss gradient w.r.t. policy token log-prob |
|---|---|---:|---:|---:|
| Base, no cap | n/a | `inf` | `inf` | `-inf` without bias correction; `NaN` with it |
| PR head, `1e-8` | accepted | `0` | `0` | `0` in both settings |
| PR head, `10` | accepted | `22016` | `2202` | `−2202` without bias correction; `0` with it |

The candidate casts the cap to the KL ratio tensor's dtype. `torch.tensor(1e-8, dtype=torch.float16)` is `0.0`; the source accepts this value because config validation only checks that the Python float is positive and finite. Clamping the positive log-ratio to zero produces no KL penalty and no corrective gradient, so this configuration silently disables the intended guard.

For the representable control, the straight-through K3 gradient with bias correction and an importance ratio of one is analytically `−beta * cap = −1`. The real production method returns zero for this float16 fixture. The bias-correction setting defaults to true in `GRPOConfig`. The run without bias correction retains a negative finite gradient, which isolates the cancellation to the multiplication of the K3 term by the importance ratio in low precision. This result does not establish how often model forwards produce float16 log-probabilities in deployed configurations.

The separate rational/JSON checker imports no TRL modules. It verified pinned revisions, both bias-correction arms, config acceptance, the float16 cap conversion, and the recorded method outputs. Run-4 is a complete audit with a mixed non-pass against the candidate's intended gradient behavior. Run-1 omitted the config acceptance check; run-2 omitted the explicit casted-cap evidence; run-3 retained a malformed command attempt. None replaces or hides the later run.

## Interpretation and next decision

This is actionable review evidence for an existing open upstream PR, not a newly discovered failure in a released trainer and not an upstream contribution by VARE. The candidate fix is opt-in and addresses a real overflow path, but the audited PR head accepts a cap that becomes zero in float16 and loses the default bias-corrected gradient for a normal representable cap. A complete upstream fix should reject a cap that becomes non-positive in the working dtype and keep the clipped K3/bias-correction gradient numerically stable in half precision. Those changes have not been made or validated here.

No GPU, paid API, or external compute was used. The experiment does not cover Liger, actual model outputs, optimizer updates, convergence, downstream tasks, or prevalence. No model capability or training-quality gain is claimed.

**Decision: continue only as an upstream review finding.** The next decisive step is to test a narrowly scoped float32 accumulation for the clipped K3 and bias-correction product, plus a fail-fast check after clip conversion, through the same production loss method and the existing controls. Do not duplicate the open PR or treat this fixture as an upstream fix.

## Reproduction

From a checkout at the locked revisions, with PyTorch installed:

```bash
PYTHONPATH=/path/to/trl-base python scripts/audit_trl_grpo_kl_clip_dtype_underflow.py \
  --source-root /path/to/trl-base --revision-name base \
  --output /tmp/trl-kl-base.json
PYTHONPATH=/path/to/trl-pr-head python scripts/audit_trl_grpo_kl_clip_dtype_underflow.py \
  --source-root /path/to/trl-pr-head --revision-name candidate --clip 1e-8 \
  --output /tmp/trl-kl-tiny.json
PYTHONPATH=/path/to/trl-pr-head python scripts/audit_trl_grpo_kl_clip_dtype_underflow.py \
  --source-root /path/to/trl-pr-head --revision-name candidate --clip 10 \
  --output /tmp/trl-kl-control.json
python scripts/verify_trl_grpo_kl_clip_dtype_underflow.py \
  --base /tmp/trl-kl-base.json --tiny /tmp/trl-kl-tiny.json \
  --control /tmp/trl-kl-control.json --output /tmp/trl-kl-audit.json
```
