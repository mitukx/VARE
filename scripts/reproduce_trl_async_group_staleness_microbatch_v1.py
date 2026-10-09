#!/usr/bin/env python3
"""Trace a scored group through queue admission, microbatch planning, and packing."""

from __future__ import annotations

import asyncio
import json
import queue
from collections import defaultdict
from types import SimpleNamespace

from accelerate import PartialState
from trl.experimental.async_grpo.async_grpo_trainer import (
    DataCollatorForRollout,
    FixedCountBatcher,
    RolloutQueueDataset,
)
from trl.experimental.async_grpo.async_rollout_worker import RolloutGroup, TrainingSequence, _AsyncRolloutLoop


def reward_fixture(completions, **_kwargs):
    return [float(i) for i in range(len(completions))]


def score_group():
    loop = object.__new__(_AsyncRolloutLoop)
    loop.reward_funcs = [reward_fixture]
    loop.reward_func_names = ["reward_fixture"]
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


def row_dict(item):
    return {
        "input_ids": item.input_ids,
        "completion_mask": item.completion_mask,
        "old_log_probs": item.old_log_probs,
        "advantage": item.advantage,
        "group_id": item.group_id,
        "metrics": item.metrics,
    }


def fresh_sentinel(group_id):
    return SimpleNamespace(
        input_ids=[90 + group_id, 99],
        completion_mask=[0, 1],
        old_log_probs=[0.0, -0.1],
        advantage=0.0,
        group_id=group_id,
        metrics={"reward": 0.0},
        model_version=1,
        enqueued_at=None,
    )


def build_batch(items, current_version_fn):
    source_queue: queue.Queue = queue.Queue()
    for item in items:
        source_queue.put(item)
    queue_metrics: dict = defaultdict(list)
    source = RolloutQueueDataset(
        rollout_queue=source_queue,
        model_version_fn=current_version_fn,
        check_health_fn=lambda _timeout: None,
        stale_after_s=60.0,
        metrics=queue_metrics,
        max_staleness=0,
        poll_interval_s=0.01,
    )
    batcher = FixedCountBatcher(source, num_processes=1, microbatch_size=3)
    microbatch = next(iter(batcher))
    groups_trained: set[int] = set()
    metrics: dict = defaultdict(list)
    collator = DataCollatorForRollout(
        pad_token_id=0,
        num_processes=1,
        groups_trained=groups_trained,
        metrics=metrics,
    )
    tensors = collator.torch_call([microbatch])
    return microbatch[0], tensors, queue_metrics, groups_trained


def main():
    PartialState()
    scored = score_group()
    assert len(scored) == 3 and {item.group_id for item in scored} == {41}

    fixed_rows, fixed_tensors, _, fixed_trained = build_batch(scored, lambda: 0)
    assert [item["group_id"] for item in fixed_rows] == [41, 41, 41]
    assert int(fixed_tensors["global_n_tokens"][0]) == 3
    assert fixed_trained == {41}

    reads = [0]

    def transition_version():
        reads[0] += 1
        return 0 if reads[0] == 1 else 1

    transitioned = [*scored, fresh_sentinel(99), fresh_sentinel(100)]
    rows, tensors, queue_metrics, trained = build_batch(transitioned, transition_version)
    group_rows = [item for item in rows if item["group_id"] == 41]
    group_completion_tokens = sum(sum(item["completion_mask"]) for item in group_rows)
    batch_completion_tokens = int(tensors["global_n_tokens"][0])
    assert len(rows) == 3
    assert len(group_rows) == 1
    assert group_completion_tokens == 1
    assert batch_completion_tokens == 3
    assert queue_metrics["sample/dropped_stale_total"] == [1.0, 1.0]
    assert trained == {41, 99, 100}

    print(json.dumps({
        "result": "partial_group_in_optimizer_microbatch_input",
        "full_group_rows_in_fixed_control": len([item for item in fixed_rows if item["group_id"] == 41]),
        "transition_group_rows_in_microbatch": len(group_rows),
        "transition_group_completion_tokens": group_completion_tokens,
        "microbatch_completion_tokens": batch_completion_tokens,
        "dropped_stale_group_rows": len(queue_metrics["sample/dropped_stale_total"]),
        "groups_seen_by_collator": sorted(trained),
        "claim_boundary": "Production scoring, queue filtering, fixed-count planning, and collation are exercised. The batch contains one of three group rows; no compute_loss or optimizer step is run.",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
