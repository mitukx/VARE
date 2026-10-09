"""Source-level counterexample: queue admission is not same-update consumption."""

from collections import defaultdict
from dataclasses import fields
import json
import queue

from accelerate import PartialState
from trl.experimental.async_grpo.async_grpo_trainer import FixedCountBatcher, RolloutQueueDataset
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample


PartialState()
current_version = [0]
rollout_fields = {field.name for field in fields(RolloutSample)}
data_queue = queue.Queue()


def put(row_id, version, rollout_id=None, rollout_size=1):
    kwargs = dict(
        prompt=[], completion=[], input_ids=[row_id], completion_mask=[1], old_log_probs=[0.0],
        advantage=1.0, model_version=version, group_id=row_id, metrics={},
    )
    if "rollout_id" in rollout_fields:
        kwargs.update(rollout_id=rollout_id, rollout_size=rollout_size)
    data_queue.put(RolloutSample(**kwargs))


put(10, 0, "forked", 2)
put(11, 0, "forked", 2)
put(99, 1)
metrics = defaultdict(list)
dataset = RolloutQueueDataset(
    rollout_queue=data_queue,
    model_version_fn=lambda: current_version[0],
    check_health_fn=lambda _timeout: None,
    stale_after_s=60.0,
    metrics=metrics,
    max_staleness=0,
    poll_interval_s=0.01,
)
microbatches = iter(FixedCountBatcher(dataset, num_processes=1, microbatch_size=1))
first = next(microbatches)
first_row = first[0][0]
current_version[0] = 1  # represents the policy-version transition after the first optimizer microbatch
second = next(microbatches)
second_row = second[0][0]

print(json.dumps({
    "first_microbatch": first_row["input_ids"][0],
    "version_before_second_microbatch": current_version[0],
    "second_microbatch": second_row["input_ids"][0],
    "second_row_rollout_id": second_row.get("rollout_id"),
    "stale_rows_dropped": int(sum(metrics["sample/dropped_stale_total"])),
    "claim_boundary": "The version transition is simulated between real fixed-count microbatches; no optimizer or model is run.",
}, sort_keys=True))
