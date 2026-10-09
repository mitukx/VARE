#!/usr/bin/env python3
"""Test whether TokenBudgetBatcher read-ahead can outlive a freshness check."""

from __future__ import annotations

import json
import queue
from collections import defaultdict
from types import SimpleNamespace

from accelerate import PartialState
from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout, RolloutQueueDataset, TokenBudgetBatcher


def sample(row_id: int, *, model_version: int, rollout_id: str, rollout_size: int, group_id: int):
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "token budget prefetch fixture"}],
        completion=[{"role": "assistant", "content": f"row-{row_id}"}],
        input_ids=[row_id, row_id + 10],
        completion_mask=[0, 1],
        old_log_probs=[0.0, -0.25],
        advantage=1.0,
        model_version=model_version,
        group_id=group_id,
        metrics={"reward": 1.0},
        enqueued_at=None,
        rollout_id=rollout_id,
        rollout_size=rollout_size,
    )


def run(version_transition: bool) -> dict:
    PartialState()
    sentinel_version = 1 if version_transition else 0
    items = [
        sample(10, model_version=0, rollout_id="rollout-a", rollout_size=2, group_id=41),
        sample(11, model_version=0, rollout_id="rollout-a", rollout_size=2, group_id=41),
        sample(90, model_version=sentinel_version, rollout_id="sentinel-a", rollout_size=1, group_id=99),
        sample(91, model_version=sentinel_version, rollout_id="sentinel-b", rollout_size=1, group_id=100),
    ]
    source_queue: queue.Queue = queue.Queue()
    for item in items:
        source_queue.put(item)

    version = [0]
    checks = [0]

    def current_version():
        checks[0] += 1
        return version[0]

    metrics = defaultdict(list)
    source = RolloutQueueDataset(
        rollout_queue=source_queue,
        model_version_fn=current_version,
        check_health_fn=lambda _timeout: None,
        stale_after_s=60.0,
        metrics=metrics,
        max_staleness=0,
        poll_interval_s=0.01,
    )
    batcher = iter(
        TokenBudgetBatcher(
            source,
            num_processes=1,
            token_budget=2,
            metrics=defaultdict(list),
        )
    )
    collator = DataCollatorForRollout(
        pad_token_id=0,
        num_processes=1,
        groups_trained=set(),
        metrics=defaultdict(list),
        token_budget=2,
    )

    first = collator.torch_call([next(batcher)])
    checks_before_transition = checks[0]
    if version_transition:
        # Model-version sync happens after the first optimizer update.
        version[0] = 1
    second = collator.torch_call([next(batcher)])
    third = collator.torch_call([next(batcher)])
    batcher.close()

    ids = [
        [int(token) for token in batch["input_ids"][0].tolist()]
        for batch in (first, second, third)
    ]
    expected_versions = {10: 0, 11: 0, 90: sentinel_version, 91: sentinel_version}
    consumed_staleness = [
        version[0] - expected_versions[row[0]] if row[0] in expected_versions else None
        for row in ids
    ]
    return {
        "version_transition_between_collated_microbatches": version_transition,
        "microbatch_input_ids": ids,
        "fork_rows_per_microbatch": [sum(token in (10, 11) for token in row) for row in ids],
        "consumption_staleness_by_microbatch": consumed_staleness,
        "max_staleness": 0,
        "version_checks_before_transition": checks_before_transition,
        "total_version_checks": checks[0],
        "stale_rows_dropped": len(metrics["sample/dropped_stale_total"]),
        "completion_tokens_by_microbatch": [int(batch["global_n_tokens"][0]) for batch in (first, second, third)],
    }


def main() -> None:
    print(
        json.dumps(
            {
                "fixed_version_control": run(version_transition=False),
                "version_transition": run(version_transition=True),
                "claim_boundary": (
                    "Exercises production RolloutQueueDataset, TokenBudgetBatcher, and DataCollatorForRollout. "
                    "The version advances between collated microbatches; no optimizer or model forward is run."
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
