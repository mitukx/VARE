#!/usr/bin/env python3
"""Trace a scored AsyncGRPO group through staleness admission and packing.

Run against the pinned TRL checkout:

    PYTHONPATH=/tmp/trl-vare-pydeps:/tmp/trl-vare-async-staleness-main \
      python scripts/reproduce_trl_async_group_staleness_packed_batch_v1.py
"""

from __future__ import annotations

import asyncio
import json
import queue
from collections import defaultdict
from types import SimpleNamespace

from accelerate import PartialState
from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout, RolloutQueueDataset
from trl.experimental.async_grpo.async_rollout_worker import (
    RolloutGroup,
    RolloutSample,
    TrainingSequence,
    _AsyncRolloutLoop,
)


def known_rewards(completions, **_kwargs):
    return [float(i) for i in range(len(completions))]


def score_one_group() -> list[RolloutSample]:
    loop = object.__new__(_AsyncRolloutLoop)
    loop.reward_funcs = [known_rewards]
    loop.reward_func_names = ["known_rewards"]
    loop._env_reward_types = []

    completions = [[{"role": "assistant", "content": f"answer-{i}"}] for i in range(3)]
    sequences = [
        [TrainingSequence([10, 20 + i], [0, 1], [0.0, -0.25], f"rollout-{i}")]
        for i in range(3)
    ]
    group = RolloutGroup(
        prompts=[[{"role": "user", "content": "fixture"}] for _ in range(3)],
        reward_kwargs={},
        completions=completions,
        completions_ids=[[20 + i] for i in range(3)],
        completions_sequences=sequences,
        tool_call_counts=[0, 0, 0],
        tool_failure_counts=[0, 0, 0],
        model_version=0,
        group_id=41,
        env_rewards=[None, None, None],
        rollout_rewards=[None, None, None],
    )
    return asyncio.run(loop._score_group(group))


def as_training_row(item) -> dict:
    return {
        "input_ids": item.input_ids,
        "completion_mask": item.completion_mask,
        "old_log_probs": item.old_log_probs,
        "advantage": item.advantage,
        "group_id": item.group_id,
        "metrics": item.metrics,
    }


def collate(rows: list[dict]) -> dict:
    collator = DataCollatorForRollout(
        pad_token_id=0,
        num_processes=1,
        groups_trained=set(),
        metrics=defaultdict(list),
    )
    return collator.torch_call([[rows]])


def queue_dataset(q, current_version, metrics):
    return RolloutQueueDataset(
        rollout_queue=q,
        model_version_fn=lambda: current_version[0],
        check_health_fn=lambda _stale_after_s: None,
        stale_after_s=60.0,
        metrics=metrics,
        max_staleness=0,
        poll_interval_s=0.01,
    )


def sentinel() -> SimpleNamespace:
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "sentinel"}],
        completion=[{"role": "assistant", "content": "fresh"}],
        input_ids=[90, 99],
        completion_mask=[0, 1],
        old_log_probs=[0.0, -0.1],
        advantage=0.0,
        model_version=1,
        group_id=99,
        metrics={"reward": 0.0},
        enqueued_at=None,
    )


def main() -> None:
    PartialState()
    samples = score_one_group()
    assert len(samples) == 3
    assert {s.group_id for s in samples} == {41}
    full_batch = collate([as_training_row(s) for s in samples])

    control_q: queue.Queue = queue.Queue()
    for item in samples:
        control_q.put(item)
    control_metrics: dict = defaultdict(list)
    control_iter = iter(queue_dataset(control_q, [0], control_metrics))
    fixed_rows = [next(control_iter) for _ in range(3)]
    control_iter.close()
    fixed_batch = collate(fixed_rows)

    q: queue.Queue = queue.Queue()
    for item in samples:
        q.put(item)
    q.put(sentinel())
    current_version = [0]
    queue_metrics: dict = defaultdict(list)
    data_iter = iter(queue_dataset(q, current_version, queue_metrics))
    first_row = next(data_iter)
    current_version[0] = 1
    fresh_row = next(data_iter)
    data_iter.close()
    assert fresh_row["group_id"] == 99

    partial_batch = collate([first_row])
    full_tokens = int(full_batch["global_n_tokens"][0])
    partial_tokens = int(partial_batch["global_n_tokens"][0])
    assert full_tokens == 3
    assert partial_tokens == 1
    assert queue_metrics["sample/dropped_stale_total"] == [1.0, 1.0]

    print(
        json.dumps(
            {
                "result": "partial_scored_group_reaches_packer",
                "scorer_output_rows": len(samples),
                "scorer_output_group_ids": sorted({s.group_id for s in samples}),
                "fixed_version_queue_rows": len(fixed_rows),
                "fixed_version_completion_tokens": int(fixed_batch["global_n_tokens"][0]),
                "full_group_completion_tokens": full_tokens,
                "transition_batch_completion_tokens": partial_tokens,
                "stale_group_rows_dropped": len(queue_metrics["sample/dropped_stale_total"]),
                "retained_completion_token_fraction": partial_tokens / full_tokens,
                "claim_boundary": "Actual TRL scoring and packing functions receive a partial group's rows after the production queue filter. This fixture does not run an optimizer step or measure downstream task success.",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
