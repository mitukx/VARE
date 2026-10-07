# Replay group freshness atomicity

Group-relative objectives such as GRPO compare multiple responses from one rollout group. If current-freshness filtering drops only stale members, the remaining samples no longer form the comparison group the objective expects.

## Frozen task and baseline

The [task](../benchmarks/regressions/replay_group_freshness/TASK.md) and [protocol lock](../benchmarks/regressions/replay_group_freshness/protocol.lock.json) were committed before the baseline evaluation. The fixture has two groups of four; one member in the first group is marked inadmissible by the freshness function. It also checks that independent fresh groups are retained whole and that `grouped=False` keeps per-item filtering.

At baseline revision `cd3129ca52c299dd0e5048d5193ee1d528b0a7ba`, `sample_current(..., grouped=True)` returned seven experiences: three from the group with one stale member and all four from the fresh group. The [baseline record](../results/replay-group-freshness-v1/baseline.grade.json) preserves the partial group.

## Fix and result

The sampler now collects current freshness for every member before admitting grouped experiences. If any member is inadmissible, that group is omitted. The separate all-fresh group remains intact; per-item mode still returns seven eligible experiences.

The fixed source at `798fcfde5fa8c1fd601f08edc349fb50417d2bf3` passes the locked regression. The full local suite passes, including [`test_replay_group_freshness.py`](../tests/test_replay_group_freshness.py). See the [fixed aggregate](../results/replay-group-freshness-v1/summary.json) and [manifest](../results/replay-group-freshness-v1/manifest.json).

## Limits and reproduction

This demonstrates one concrete group-integrity failure and its correction. It is not evidence of model learning, empirical replay quality, throughput, distributed operation, or general correctness beyond the fixture.

```bash
python3.12 benchmarks/regressions/replay_group_freshness/evaluator/grade.py \
  --workspace . \
  --json-out /tmp/replay-group-freshness-grade.json
```

The task uses local CPU code only and needs no model weights, GPU, paid API, or external compute.
