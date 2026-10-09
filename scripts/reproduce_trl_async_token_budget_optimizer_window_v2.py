#!/usr/bin/env python3
"""Follow token-budget read-ahead through Transformers' accumulation fetch window."""

from __future__ import annotations

import hashlib
import inspect
import json
import queue
from collections import defaultdict
from types import SimpleNamespace

import torch
import transformers
from accelerate import PartialState
from transformers import Trainer
from trl.experimental.async_grpo.async_grpo_trainer import (
    DataCollatorForRollout,
    RolloutQueueDataset,
    TokenBudgetBatcher,
)


def sample(row_id: int, model_version: int, rollout_id: str, rollout_size: int, group_id: int):
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "optimizer window fixture"}],
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


def run(gradient_accumulation_steps: int) -> dict:
    PartialState()
    source_items = [
        sample(10, 0, "rollout-a", 2, 41),
        sample(11, 0, "rollout-a", 2, 41),
        sample(90, 1, "sentinel-a", 1, 99),
        sample(91, 1, "sentinel-b", 1, 100),
    ]
    source_queue: queue.Queue = queue.Queue()
    for item in source_items:
        source_queue.put(item)

    version = [0]
    queue_metrics = defaultdict(list)
    source = RolloutQueueDataset(
        rollout_queue=source_queue,
        model_version_fn=lambda: version[0],
        check_health_fn=lambda _timeout: None,
        stale_after_s=60.0,
        metrics=queue_metrics,
        max_staleness=0,
        poll_interval_s=0.01,
    )
    batcher = TokenBudgetBatcher(source, num_processes=1, token_budget=2, metrics=defaultdict(list))
    batch_iterator = iter(batcher)
    collator = DataCollatorForRollout(
        pad_token_id=0,
        num_processes=1,
        groups_trained=set(),
        metrics=defaultdict(list),
        token_budget=2,
    )

    # Call the installed Transformers implementation. Only item counting is stubbed;
    # batch collection itself is Trainer.get_batch_samples.
    trainer_probe = SimpleNamespace(_get_num_items_in_batch=lambda _samples, _device: None)
    fetched, _ = Trainer.get_batch_samples(
        trainer_probe, batch_iterator, gradient_accumulation_steps, torch.device("cpu")
    )
    collation_versions = []
    fetched_ids = []
    for planner_batch in fetched:
        collation_versions.append(version[0])
        tensors = collator.torch_call([planner_batch])
        fetched_ids.append([int(token) for token in tensors["input_ids"][0].tolist()])

    # Match AsyncGRPO's StepIntervalCallback: the policy version advances at the
    # optimizer-step boundary, after the accumulated batch window has been fetched.
    version[0] = 1
    if not fetched_ids:
        raise AssertionError("Trainer fetched no batches")

    # For GA=1, retrieve the next window after the callback; for GA>1 this fixture
    # only needs the first complete window to determine whether the sibling crossed.
    next_ids = None
    next_collation_version = None
    if gradient_accumulation_steps == 1:
        next_window, _ = Trainer.get_batch_samples(trainer_probe, batch_iterator, 1, torch.device("cpu"))
        next_collation_version = version[0]
        next_tensors = collator.torch_call([next_window[0]])
        next_ids = [int(token) for token in next_tensors["input_ids"][0].tolist()]

    batch_iterator.close()
    expected_versions = {10: 0, 11: 0, 90: 1, 91: 1}
    consumed = []
    for row, collate_version in zip(fetched_ids, collation_versions):
        consumed.append({"ids": row, "version": collate_version, "lag": collate_version - expected_versions[row[0]]})
    if next_ids is not None:
        consumed.append(
            {"ids": next_ids, "version": next_collation_version, "lag": next_collation_version - expected_versions[next_ids[0]]}
        )
    return {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "batches_fetched_before_optimizer_boundary": len(fetched_ids),
        "batches_collated_before_optimizer_boundary": fetched_ids,
        "collation_versions_before_optimizer_boundary": collation_versions,
        "post_boundary_batch_ids": next_ids,
        "post_boundary_collation_version": next_collation_version,
        "consumption_records": consumed,
        "max_staleness": 0,
        "stale_rows_dropped": len(queue_metrics["sample/dropped_stale_total"]),
    }


def main() -> None:
    source = inspect.getsource(Trainer.get_batch_samples).encode()
    print(
        json.dumps(
            {
                "transformers_version": transformers.__version__,
                "trainer_get_batch_samples_source_sha256": hashlib.sha256(source).hexdigest(),
                "ga_1": run(1),
                "ga_2": run(2),
                "claim_boundary": (
                    "Uses pinned TRL production queue, token-budget batcher, and collator plus the installed "
                    "Transformers Trainer.get_batch_samples method. Optimizer math and model forward are omitted; "
                    "the optimizer boundary is represented by the pinned AsyncGRPO weight-sync callback order."
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
