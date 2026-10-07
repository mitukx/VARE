# Task: normalize GRPO token losses per accumulation window

## Context

In the pinned trainer, DAPO/CISPO/VESPO loss paths divide by a token count covering a full generation batch. Optimizer updates accumulate only a configured number of microbatches. When those windows differ, the loss scale can drift. The main trainer and its `gspo_token` experiment have separate implementations of the relevant branch.

The upstream issue and merged change are linked in `task.json`. The task is to correct both pinned branches while preserving default and evaluation behavior.

## Goal

Update the training-time token-loss normalizer so each accumulation window has the intended scale. Keep the main trainer's shared DAPO/CISPO/VESPO path and the experimental DAPO path consistent.

## Acceptance contract

- In training, the effective normalizer accounts for the current accumulation-window size relative to `steps_per_generation`.
- When the counts match, the existing scale is preserved.
- A partial final accumulation window uses its current size.
- Evaluation does not receive a training-only correction.
- The source-derived normalization branch satisfies all locked CPU arithmetic cases in both files.

## Constraints

- Start from `source.base_revision` in `task.json`.
- Keep changes limited to the two listed trainer implementations and directly relevant regression coverage.
- Do not edit this brief, the task descriptor, the protocol lock, or any file under `evaluator/` to obtain a passing score.
- The evaluator parses the production branch and executes its normalizer statements against deterministic scalar doubles. It does not import the full trainer or run a gradient update; do not present it as end-to-end training evidence.
- No model weights, GPU, paid API, or external compute are needed.

## Reproduce

From the VARE checkout:

```bash
python3 scripts/prepare_task.py \
  --task-root benchmarks/historical/trl_grpo_accumulation_scale \
  --workspace /tmp/vare-trl-normalizer
# Edit the isolated candidate checkout.
python3 scripts/grade_task.py \
  --task-root benchmarks/historical/trl_grpo_accumulation_scale \
  --workspace /tmp/vare-trl-normalizer
```

Calibrate the evaluator against the pinned pre-fix and fixed revisions:

```bash
python3 scripts/calibrate_task.py \
  --task-root benchmarks/historical/trl_grpo_accumulation_scale
```
