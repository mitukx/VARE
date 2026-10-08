# Task: keep inherited bad-word constraints out of the behavior policy

## Context

The rollout backend records token log-probabilities for samples consumed by a GRPO-style learner. A model may carry a `bad_words_ids` list in its pretrained generation configuration. If that constraint reaches sampling, it masks tokens while learner probabilities are computed under the unmodified policy logits.

## Goal

Update the pinned source checkout so that inherited single-token `bad_words_ids` do not change the rollout distribution. The fixture starts with token `2` forbidden, while the deterministic response begins with token `2`. It needs no model weights, PyTorch, Transformers, CUDA, network connection, third-party Python package, or paid compute.

## Acceptance contract

- The six prompt/temperature conditions retain the existing v4 rollout and learner probability-parity checks.
- The effective `bad_words_ids` setting must be neutral (`None`) for every rollout.
- The fixture applies a single-token bad-word constraint to logits. Propagating the inherited constraint therefore changes the probability of the expected response and must fail.
- Existing v4 checks for temperature, repetition penalty, sampling truncation, `no_repeat_ngram_size`, `suppress_tokens`, and model-state restoration remain in force.

This fixture covers single-token `bad_words_ids` inheritance only. It does not establish coverage of multi-token bad-word sequences, all Transformers versions, or every `GenerationConfig` field.

## Constraints

- Start from the immutable revisions in `task.json`.
- Keep changes scoped to the rollout/learner probability contract.
- Do not edit this brief, task descriptor, protocol lock, or evaluator to obtain a passing score. The grader runs outside the candidate workspace and checks its own locked hashes.
- The grader executes candidate source inside a local subprocess. It is a functional harness, not an operating-system sandbox for hostile code. Run only candidate code you are willing to execute as your user.
- No model download, GPU, paid API, or external compute is part of this task.

## Reproduce

```bash
python3 scripts/prepare_task.py --task-root benchmarks/historical/rvl_bad_words_neutrality --workspace /tmp/vare-rvl-bad-words
# Apply a candidate change in /tmp/vare-rvl-bad-words.
python3 scripts/grade_task.py --task-root benchmarks/historical/rvl_bad_words_neutrality --workspace /tmp/vare-rvl-bad-words
```

The lock pins the task inputs and acceptance values. The calibration evidence grades the immutable pre-fix revision and the public fixed revision. A pass covers this fixture contract only; it is not a model-learning or capability result.
