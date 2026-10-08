# Rollout group integrity v1

## Question and decision

Does `CapabilityLoop.run_round` always provide complete configured rollout groups to group-relative training when the requested rollout target is not divisible by `samples_per_task`?

**Fixed and covered by a regression.** The engine now rounds a positive rollout target up to complete groups. The regression uses a target of six and a group size of four, then checks both the generated rollout set and the replay batch passed to `train_candidate`.

## Defect

Before the fix, the engine computed `ceil(target_rollouts / samples_per_task)` groups, then stopped appending as soon as the exact target count was reached. With `samples_per_task=4` and `rollout_count=6`, it created groups of four and two. Grouped replay preserved those labels faithfully, so the two-item group could be used by a trainer expecting four comparable samples. The existing test only used 16 rollouts and did not expose the case.

## Contract after the fix

- With `samples_per_task > 1`, `rollout_count` is a target budget. The realized count is the smallest multiple of `samples_per_task` greater than or equal to that target.
- With `samples_per_task=4`, a target of six produces eight rollouts in two groups of four.
- A nonpositive explicit rollout target raises `ValueError`.
- Grouped replay continues to select whole groups, even when its item budget is not divisible by the group size.

Rounding up may exceed the target budget by at most `samples_per_task - 1`. Callers with strict budgets should choose a divisible target.

## Verification

`tests/test_grouped_replay.py` contains the nonmultiple regression and checks the groups passed into the candidate-training hook. On Python 3.12.12, the focused suite for grouped replay, replay freshness, and engine concurrency passed (3 tests). The full repository suite passed (149 passed, 12 skipped).

This is a local deterministic correctness check. It does not establish that a particular GRPO/RLOO trainer uses these groups correctly, improve a learned policy, or measure end-to-end throughput.
