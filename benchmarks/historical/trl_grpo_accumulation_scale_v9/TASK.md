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
- Within the tested loss branches, no additional write changes `normalizer` after the checked assignment and training gate.
- The loss expression in each selected branch divides directly by the checked `normalizer` value.
- The selected policy-loss numerator remains the pinned masked sum `(per_token_loss * mask).sum()`; zeroing, bypassing, or replacing that numerator is rejected.
- The selected branch must not reassign `loss` after that checked division. Subsequent loss writes through the return are limited to the pinned entropy and auxiliary-loss adjustments; an extra write can change the returned training objective while leaving the denominator check intact. In-place tensor mutators and `out=` writes targeting these locals or a direct local-name alias, including aliases introduced by tuple/list unpacking, are also rejected.
- The selected branch is unique and appears directly in `_compute_loss`'s method body. A preceding return/raise that bypasses it, or a method return that does not return `loss`, is rejected so unreachable source cannot satisfy the extracted arithmetic oracle.

## Constraints

- Start from `source.base_revision` in `task.json`.
- Keep changes limited to the two listed trainer implementations and directly relevant regression coverage.
- Do not edit this brief, the task descriptor, the protocol lock, or any file under `evaluator/` to obtain a passing score.
- Protocol v9 is a separate version of this historical task. It parses the production branch, executes its normalizer statements against deterministic scalar doubles, checks the exact pinned masked loss numerator and direct denominator, requires a reachable unique branch and return of that loss, rejects later branch-local `loss` writes, and checks method-level loss writes against the pinned entropy/auxiliary adjustments, including in-place tensor mutation calls through direct local-name aliases created by ordinary assignment or tuple/list unpacking. It does not execute the full loss expression, import the trainer, or run a gradient update; do not present it as end-to-end training evidence. Protocol v8 remains available in `../trl_grpo_accumulation_scale/` for historical replay.
- Protocol version 3 rejected an unmodeled later overwrite of `normalizer` and a loss denominator that scales or bypasses that value, but a frozen mutation audit later found that it accepted a post-division `loss = loss * 2` in both pinned branches. Protocol version 4 adds a branch-local later-`loss`-write guard. These source-structural checks target demonstrated false accepts and can reject semantically valid refactors; they do not generalize to arbitrary trainer implementations.
- No model weights, GPU, paid API, or external compute are needed.

## Version lineage

Protocol v9 adds direct tuple/list-unpacking aliases to the pinned v8 dataflow checks after a held-out source mutation showed that `loss_alias, = (loss,)` followed by an in-place tensor update could be accepted. The v8 task bundle remains unchanged in `../trl_grpo_accumulation_scale/`.

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
