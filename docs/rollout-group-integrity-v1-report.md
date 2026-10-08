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
- Group IDs include a deterministic run serial, so calling `run_round` more than once with the same `round_index` does not merge separate cohorts in replay.
- Grouped replay continues to select whole groups, even when its item budget is not divisible by the group size.

Rounding up may exceed the target budget by at most `samples_per_task - 1`. Callers with strict budgets should choose a divisible target.

## Replay-capacity follow-up

A later review found a second way to create a partial group: replay capacity evicts experiences individually. With capacity seven, eight equal-priority experiences in two groups of four left the replay store with group sizes four and three. Freshness checks alone could not identify the missing fourth member.

The engine now stamps each generated group with its declared size and copies VARE's group identity onto each rollout record. Both grouped sampling paths reject a group unless its stored cardinality matches the declared size and, for current sampling, every member is fresh. A partial group can remain in bounded storage after capacity pressure, but it cannot reach `train_candidate`. The added regression reproduces the 4+3 storage case and confirms only the complete four-member group is returned. This favors a valid comparison batch over using an incomplete one; the unused partial experiences are lost to training.

## Verification

`tests/test_grouped_replay.py` covers nonmultiple generation, repeated round indices, and the groups passed into the candidate-training hook. `tests/test_replay_group_freshness.py` covers stale-member removal and the replay-capacity 4+3 case. After the added repeated-round-index and partial-restore regressions, the focused group, freshness, concurrency, and RVL hook suite passed (9 tests), and the full repository suite passed (153 passed, 12 skipped) on Python 3.12.12.

This is a local deterministic correctness check. It does not establish that a particular GRPO/RLOO trainer uses these groups correctly, improve a learned policy, or measure end-to-end throughput.
