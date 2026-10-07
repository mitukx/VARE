# Task: keep rollout and learner probabilities aligned

## Context

The rollout backend records token log-probabilities for samples consumed by a GRPO-style learner. A historical run exposed a mismatch: pretrained generation settings could modify scores during sampling, while the learner evaluated the response under the unmodified policy logits. This creates incorrect behavior-policy ratios before any optimizer update.

## Goal

Update the pinned source checkout so that trainable rollouts record probabilities under the same distribution the learner evaluates. The evaluator uses deterministic CPU-only model and tokenizer doubles with deliberately non-neutral pretrained generation settings. It does not require model weights, PyTorch, Transformers, CUDA, or a network connection.

## Acceptance contract

- For positive sampling temperatures, recorded per-token rollout log-probabilities match the neutral policy distribution at that temperature within the locked tolerance.
- A pretrained repetition penalty cannot silently change the scored behavior distribution.
- The learner uses the rollout's sampling temperature when reconstructing token probabilities and rejects samples whose scores do not represent a trainable stochastic policy.
- The sampler restores model training/evaluation state even if generation raises.
- Non-finite or invalid sampling temperatures fail closed.

## Constraints

- The source checkout begins at the revision in `task.json`.
- Keep changes scoped to the rollout/learner probability contract and relevant upstream regression coverage.
- Do not edit this brief, the task descriptor, the protocol lock, or any file under `evaluator/` to obtain a passing score. The grader is invoked from outside the candidate workspace and checks its own locked hash.
- No model download, GPU, paid API, or external compute is part of the task.

## Reproduce

From the VARE checkout:

```bash
python3 scripts/prepare_task.py --workspace /tmp/vare-rvl-parity
# Apply a candidate change in /tmp/vare-rvl-parity.
python3 scripts/grade_task.py --workspace /tmp/vare-rvl-parity
```

The calibration command grades the immutable pre-fix revision and the public fixed revision. The task result is limited to this contract and fixture.
