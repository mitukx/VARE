#!/usr/bin/env python3
"""Reproduce per-sample staleness admission splitting one AsyncGRPO group.

Run with the pinned TRL checkout first on PYTHONPATH, for example:

    PYTHONPATH=/tmp/trl-vare-pydeps:/tmp/trl-vare-async-norm \
      python scripts/reproduce_trl_async_group_staleness_v1.py

This deliberately exercises RolloutQueueDataset's production iterator and does
not modify or patch TRL. The fixture models three samples emitted from one
scored group, an optimizer update between reads, and one fresh control sample.
"""

from __future__ import annotations

import json
import queue
from collections import defaultdict
from types import SimpleNamespace

from trl.experimental.async_grpo.async_grpo_trainer import RolloutQueueDataset


def sample(row: int, *, model_version: int, group_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "fixture"}],
        completion=[{"role": "assistant", "content": f"row-{row}"}],
        input_ids=[row, row + 10],
        completion_mask=[0, 1],
        old_log_probs=[0.0, -0.25],
        advantage=float(row),
        model_version=model_version,
        group_id=group_id,
        metrics={"reward": float(row)},
        enqueued_at=None,
    )


def make_dataset(q: queue.Queue, current_version: list[int], metrics: dict) -> RolloutQueueDataset:
    return RolloutQueueDataset(
        rollout_queue=q,
        model_version_fn=lambda: current_version[0],
        check_health_fn=lambda _stale_after_s: None,
        stale_after_s=60.0,
        metrics=metrics,
        max_staleness=0,
        poll_interval_s=0.01,
    )


def main() -> None:
    # Control: a group remains wholly admissible while the policy version is fixed.
    control_q: queue.Queue = queue.Queue()
    for row in range(3):
        control_q.put(sample(row, model_version=0, group_id=41))
    control_metrics: dict = defaultdict(list)
    control_iter = iter(make_dataset(control_q, [0], control_metrics))
    control = [next(control_iter) for _ in range(3)]
    control_ids = [row["input_ids"][0] for row in control]
    assert control_ids == [0, 1, 2]
    control_iter.close()

    # Counterexample: first member is consumed at v0; an optimizer update moves
    # the policy to v1; the other two v0 members are independently dropped.
    q: queue.Queue = queue.Queue()
    for row in range(3):
        q.put(sample(row, model_version=0, group_id=41))
    q.put(sample(99, model_version=1, group_id=99))  # fresh sentinel to end iteration
    metrics: dict = defaultdict(list)
    current_version = [0]
    dataset_iter = iter(make_dataset(q, current_version, metrics))
    first = next(dataset_iter)
    assert first["input_ids"][0] == 0 and first["group_id"] == 41

    current_version[0] = 1  # represents the update between queue reads
    after_update = next(dataset_iter)
    assert after_update["input_ids"][0] == 99 and after_update["group_id"] == 99
    dataset_iter.close()

    dropped = metrics["sample/dropped_stale_total"]
    assert dropped == [1.0, 1.0]
    assert q.empty()

    print(
        json.dumps(
            {
                "result": "counterexample_reproduced",
                "fixed_version_control_yielded_ids": control_ids,
                "same_group_id": 41,
                "accepted_before_update": [first["input_ids"][0]],
                "dropped_after_update_count": len(dropped),
                "fresh_sentinel_yielded": after_update["input_ids"][0],
                "claim_boundary": "The production queue iterator admits individual samples. This demonstrates that a group can be partially admitted across a policy-version transition; it does not measure optimizer or downstream task effects.",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
