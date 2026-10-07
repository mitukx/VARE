# Task: keep rollout and learner probabilities aligned

## Context

The rollout backend records token log-probabilities for samples consumed by a GRPO-style learner. A historical run exposed a mismatch: pretrained generation settings could modify scores during sampling, while the learner evaluated the response under the unmodified policy logits. This creates incorrect behavior-policy ratios before any optimizer update.

Hugging Face documents `typical_p` as a truncation setting; the neutral value `1.0` leaves the distribution unchanged ([versioned GenerationConfig reference](https://huggingface.co/docs/transformers/v4.41.2/en/main_classes/text_generation)).

## Goal

Update the pinned source checkout so that trainable rollouts record probabilities under the same distribution the learner evaluates. The evaluator uses deterministic CPU-only model and tokenizer doubles with deliberately non-neutral pretrained generation settings. It does not require model weights, PyTorch, Transformers, CUDA, or a network connection.

## Acceptance contract

- For six distinct prompt/temperature conditions, recorded per-token rollout log-probabilities match the neutral policy distribution at that temperature within the locked tolerance.
- A pretrained repetition penalty cannot silently change the scored behavior distribution.
- A non-neutral pretrained `typical_p` cannot silently change the scored behavior distribution.
- The learner's actual `_sample_objective` path reconstructs the same token probabilities at each temperature; the grader captures the probabilities passed to the objective at runtime rather than matching source text.
- The learner rejects zero, negative, and non-finite temperatures before evaluating the objective.
- The sampler restores either initial model training/evaluation state after successful generation and after generation raises.
- The sampler rejects negative and non-finite temperatures before invoking generation.

## Constraints

- The source checkout begins at the revision in `task.json`.
- Keep changes scoped to the rollout/learner probability contract.
- Do not edit this brief, the task descriptor, the protocol lock, or any file under `evaluator/` to obtain a passing score. The grader is invoked from outside the candidate workspace and checks its own locked hash.
- The grader executes candidate source inside a local subprocess to observe behavior. It is a functional benchmark harness, not an operating-system sandbox for hostile code. Run only on candidate workspaces whose code you are willing to execute as your user.
- No model download, GPU, paid API, or external compute is part of the task.

## Reproduce

From the VARE checkout:

```bash
python3 scripts/prepare_task.py --workspace /tmp/vare-rvl-parity
# Apply a candidate change in /tmp/vare-rvl-parity.
python3 scripts/grade_task.py --workspace /tmp/vare-rvl-parity
```

The calibration command grades the immutable pre-fix revision and the public fixed revision. It saves the patch hash relative to the pre-fix revision. The task result is limited to this contract and fixture.
