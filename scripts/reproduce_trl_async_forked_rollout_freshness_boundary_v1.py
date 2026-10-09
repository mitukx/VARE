#!/usr/bin/env python3
"""Check freshness at the boundary between microbatches for an admitted fork."""

from __future__ import annotations

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


def sample(row_id: int, *, model_version: int, rollout_id: str, rollout_size: int, group_id: int):
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "freshness fixture"}],
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
    source_queue: queue.Queue = queue.Queue()
    source_queue.put(sample(10, model_version=0, rollout_id="rollout-a", rollout_size=2, group_id=41))
    source_queue.put(sample(11, model_version=0, rollout_id="rollout-a", rollout_size=2, group_id=41))
    source_queue.put(sample(90, model_version=1, rollout_id="sentinel", rollout_size=1, group_id=99))

    version = [0]
    checks = [0]

    def current_version():
        checks[0] += 1
        return version[0]

    metrics = defaultdict(list)
    dataset = RolloutQueueDataset(
        rollout_queue=source_queue,
        model_version_fn=current_version,
        check_health_fn=lambda _timeout: None,
        stale_after_s=60.0,
        metrics=metrics,
        max_staleness=0,
        poll_interval_s=0.01,
    )
    batches = iter(FixedCountBatcher(dataset, num_processes=1, microbatch_size=1))
    collator = DataCollatorForRollout(
        pad_token_id=0,
        num_processes=1,
        groups_trained=set(),
        metrics=defaultdict(list),
    )

    first = next(batches)
    first_tensors = collator.torch_call([first])
    first_ids = first_tensors["input_ids"][0].tolist()
    if version_transition:
        # Represents a policy-version advance after the first microbatch is consumed.
        version[0] = 1
    second = next(batches)
    second_tensors = collator.torch_call([second])
    second_ids = second_tensors["input_ids"][0].tolist()
    batches.close()

    return {
        "version_transition_between_microbatches": version_transition,
        "first_microbatch_input_ids": first_ids,
        "second_microbatch_input_ids": second_ids,
        "first_is_fork_row": first_ids[0] == 10,
        "second_is_fork_sibling": second_ids[0] == 11,
        "second_is_fresh_sentinel": second_ids[0] == 90,
        "completion_tokens_per_microbatch": [
            int(first_tensors["global_n_tokens"][0]),
            int(second_tensors["global_n_tokens"][0]),
        ],
        "stale_rows_dropped": len(metrics["sample/dropped_stale_total"]),
        "version_checks": checks[0],
        "max_staleness": 0,
    }


def main() -> None:
    results = {
        "fixed_version_control": run(version_transition=False),
        "version_transition": run(version_transition=True),
        "claim_boundary": (
            "Production RolloutQueueDataset, FixedCountBatcher, and DataCollatorForRollout are exercised. "
            "The version transition is injected between microbatches; no optimizer, model, or task outcome is run."
        ),
    }
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
