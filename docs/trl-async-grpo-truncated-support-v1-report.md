# TRL async GRPO truncation support audit v1

**Result:** the pinned production `AsyncGRPOTrainer.compute_loss` reproduces a support-limited policy update on a deterministic CPU fixture. It does not execute vLLM or a complete training run, and it does not establish an upstream defect or a new method.

**Frozen protocol:** [protocol](../protocols/trl_async_grpo_truncated_support_v1.lock.json)
**Accepted run:** [run-4](../results/trl-async-grpo-truncated-support-v1/run-4/)
**Retained attempts:** [run-1](../results/trl-async-grpo-truncated-support-v1/run-1/) and [run-2](../results/trl-async-grpo-truncated-support-v1/run-2/)

## Question and source

The question was whether the async GRPO loss method's ratio and gradient behave as predicted at zero parameter drift when behavior log-probabilities come from a truncated sampler while current model log-probabilities are computed over the full action set.

The source was TRL revision `ed8cc2f4337fb9b7b1429db31009ad3577cd5b98`; the frozen protocol locks four source-file hashes. In that snapshot, the [vLLM serve command](https://github.com/huggingface/trl/blob/ed8cc2f4337fb9b7b1429db31009ad3577cd5b98/trl/scripts/vllm_serve.py) requests processed log-probabilities, the [async rollout worker](https://github.com/huggingface/trl/blob/ed8cc2f4337fb9b7b1429db31009ad3577cd5b98/trl/experimental/async_grpo/async_rollout_worker.py) passes top-p/top-k/min-p settings and stores returned token log-probabilities as `old_log_probs`, and [AsyncGRPOTrainer.compute_loss](https://github.com/huggingface/trl/blob/ed8cc2f4337fb9b7b1429db31009ad3577cd5b98/trl/experimental/async_grpo/async_grpo_trainer.py) forms `exp(log_probs - old_log_probs)` before the PPO clipped objective. Unlike the synchronous path, this async ratio is part of the loss directly and has no optional vLLM correction flag.

These source facts establish the arithmetic path, not the intended target distribution. The report does not label a non-unit ratio a defect by itself.

## Frozen method and result

The primary runner imported and directly invoked the unmodified production `AsyncGRPOTrainer.compute_loss`. It supplied a differentiable three-action causal-LM fixture and enumerated every action with nonzero behavior probability. No extraction or rewrite of the loss code was used. At `theta=0`,

```text
p = (1/2, 3/10, 1/5)
A = (0, 0, 1)
```

For the identity control, `q=p`. For top-p `3/4`, the frozen prefix rule retains the first two actions, giving `Z=4/5` and `q=(5/8,3/8,0)`. The frozen advantage is a method-level PPO input; it is not claimed to be reconstructed from group-normalized GRPO rewards.

| Case | q support | `E_q[p/q]` from production method | q-weighted update direction `−dL/dtheta` | Raw-policy score target |
|---|---|---:|---:|---:|
| Identity | `{0,1,2}` | `1` | `4/25 = 0.16` | `4/25 = 0.16` |
| Top-p `3/4` | `{0,1}` | `4/5 = 0.8` | `0` | `4/25 = 0.16` |

At zero drift, every retained action has the exact pointwise ratio `p/q=Z`. But the rewarded action has `q=0`; no loss computation driven only by q samples can recover its raw-p contribution. The separate rational checker reconstructs support, per-action ratios, score derivatives, q-weighted gradient, and target gradient without importing the primary runner. All frozen checks pass.

The first attempt reached the production method but the runner treated a scalar metric as a tuple and failed during result extraction. That failure is preserved in run-1. Run-2 passed the primary calculation, but its independent checker omitted two production metrics named by the frozen protocol. Run-2 is retained as an incomplete audit. Run-3 passed the completed protocol checks. Run-4 adds a direct independent comparison of the saved `loss_gradient_theta` to the negative expected update direction; its result bytes match run-3 and its expanded checker passes. Run-4 is the current accepted bundle.

## Interpretation and limits

This confirms a finite-support implication in the pinned async loss method: a q-sampled update cannot represent arbitrary raw-p expectations after q removes positive-p actions. This is standard support/absolute-continuity reasoning, not a novel algorithm. Whether this behavior is undesirable depends on the training objective: optimizing raw p from truncated q lacks support, while optimizing a transformed q policy requires a matching policy-gradient definition. The experiment does not settle that contract.

The run uses injected behavior log-probabilities and a tiny controlled model-output fixture. It does **not** run a vLLM server, the async rollout worker, full trainer initialization, an optimizer update, a model or task benchmark, or an outside reproduction. It does not establish prevalence, downstream quality impact, or a defect requiring a code change. The related [TRL issue #6789](https://github.com/huggingface/trl/issues/6789) remains the upstream context; no PR or maintainer response is claimed.

## Reproduction

Use a clean TRL checkout at the locked revision and Python 3.12 with PyTorch, Transformers, Accelerate, Datasets, and `huggingface-hub` installed. The accepted run used PyTorch 2.9.1, Transformers 4.57.3, Accelerate 1.12.0, Datasets 4.8.5, and `huggingface-hub` 0.36.0 on CPU.

```bash
python scripts/audit_trl_async_grpo_truncated_support.py \
  --source-root /path/to/trl \
  --output /tmp/async-grpo-result.json
python scripts/verify_trl_async_grpo_truncated_support.py \
  --result /tmp/async-grpo-result.json \
  --output /tmp/async-grpo-independent-check.json
```

Both outputs should report `status: pass`. The accepted bundle records the protocol snapshot, upstream source revision and hashes, runner and checker hashes, environment, outputs, adjudication, and SHA-256 manifest.
