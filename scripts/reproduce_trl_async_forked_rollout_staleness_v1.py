#!/usr/bin/env python3
"""Trace one forked conversation through scoring, staleness, batching, and collation."""

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
    completions = [
        [{"role": "assistant", "content": "forked-answer"}],
        [{"role": "assistant", "content": "other-answer"}],
    ]
    forked_rollout = [
        TrainingSequence([10, 20], [0, 1], [0.0, -0.25], "rollout-a"),
        TrainingSequence([10, 30], [0, 1], [0.0, -0.35], "rollout-a"),
    ]
    other_rollout = [TrainingSequence([11, 40], [0, 1], [0.0, -0.45], "rollout-b")]
    group = RolloutGroup(
        prompts=[[{"role": "user", "content": "fixture"}] for _ in range(2)],
        reward_kwargs={},
        completions=completions,
        completions_ids=[[20, 30], [40]],
        completions_sequences=[forked_rollout, other_rollout],
        tool_call_counts=[0, 0],
        tool_failure_counts=[0, 0],
        model_version=0,
        group_id=41,
        env_rewards=[None, None],
        rollout_rewards=[None, None],
    )
    return asyncio.run(loop._score_group(group)), forked_rollout


def fresh_sentinel(group_id):
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "sentinel"}],
        completion=[{"role": "assistant", "content": "fresh"}],
        input_ids=[90 + group_id, 99],
        completion_mask=[0, 1],
        old_log_probs=[0.0, -0.1],
        advantage=0.0,
        model_version=1,
        group_id=group_id,
        metrics={"reward": 0.0},
        enqueued_at=None,
    )


def build_microbatch(samples):
    source_queue: queue.Queue = queue.Queue()
    for item in samples:
        source_queue.put(item)
    metrics: dict = defaultdict(list)
    reads = [0]

    def current_version():
        reads[0] += 1
        return 0 if reads[0] == 1 else 1

    source = RolloutQueueDataset(
        rollout_queue=source_queue,
        model_version_fn=current_version,
        check_health_fn=lambda _timeout: None,
        stale_after_s=60.0,
        metrics=metrics,
        max_staleness=0,
        poll_interval_s=0.01,
    )
    batcher = FixedCountBatcher(source, num_processes=1, microbatch_size=2)
    microbatch = next(iter(batcher))[0]
    collator = DataCollatorForRollout(
        pad_token_id=0,
        num_processes=1,
        groups_trained=set(),
        metrics=defaultdict(list),
    )
    tensors = collator.torch_call([[microbatch]])
    return microbatch, tensors, metrics, reads[0]


def main():
    PartialState()
    scored, forked_sequences = score_group()
    assert len(scored) == 3
    # The first two scorer rows correspond to the same rollout's two forked sequences.
    assert [row.input_ids for row in scored[:2]] == [row.input_ids for row in forked_sequences]
    assert scored[0].group_id == scored[1].group_id == scored[2].group_id == 41

    sentinels = [fresh_sentinel(99), fresh_sentinel(100)]
    microbatch, tensors, metrics, version_checks = build_microbatch([*scored, *sentinels])
    first_rollout_rows = [
        item for item in microbatch if item["input_ids"] in [seq.input_ids for seq in forked_sequences]
    ]
    kept_ids = [item["input_ids"][0] for item in first_rollout_rows]
    result = {
        "result": "forked_rollout_admission_measured",
        "scorer_rows": len(scored),
        "scored_rollout_ids": [getattr(row, "rollout_id", None) for row in scored],
        "forked_rollout_sequence_count": len(forked_sequences),
        "forked_rollout_rows_in_first_microbatch": len(first_rollout_rows),
        "forked_rollout_token_prefixes_kept": kept_ids,
        "microbatch_group_ids": [item["group_id"] for item in microbatch],
        "microbatch_completion_tokens": int(tensors["global_n_tokens"][0]),
        "stale_rows_dropped": len(metrics["sample/dropped_stale_total"]),
        "staleness_checks_for_first_microbatch": version_checks,
        "claim_boundary": "The version callback advances after its first query. Scoring, queue filtering, fixed-count planning, and collation are exercised; no optimizer step or task outcome is measured.",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
