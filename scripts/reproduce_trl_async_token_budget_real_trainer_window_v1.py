#!/usr/bin/env python3
"""Run the pinned rollout batching path through a real CPU Transformers optimizer loop."""

from __future__ import annotations

import json
import queue
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from accelerate import PartialState
from transformers import Trainer, TrainerCallback, TrainingArguments
from trl.experimental.async_grpo.async_grpo_trainer import (
    DataCollatorForRollout,
    RolloutQueueDataset,
    TokenBudgetBatcher,
)


def sample(row_id: int, model_version: int, rollout_id: str, rollout_size: int, group_id: int):
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "real trainer optimizer-window fixture"}],
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


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, input_ids=None, **kwargs):
        return {"loss": self.weight * input_ids.float().mean()}


class VersionCallback(TrainerCallback):
    def __init__(self, current_version: list[int], weights: list[float]):
        self.current_version = current_version
        self.weights = weights

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.current_version[0] += 1
        if model is not None:
            self.weights.append(float(model.weight.detach().cpu()))
        return control


class RecordingTrainer(Trainer):
    def __init__(self, *args, trace, **kwargs):
        self.trace = trace
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        output = model(input_ids=inputs["input_ids"])
        loss = output["loss"]
        return (loss, output) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.trace["training_batches"].append([int(x) for x in inputs["input_ids"][0].tolist()])
        self.trace["training_versions"].append(self.trace["version"][0])
        return super().training_step(model, inputs, num_items_in_batch)


def run(gradient_accumulation_steps: int) -> dict:
    PartialState()
    current_version = [0]
    items = [
        sample(10, 0, "rollout-a", 2, 41),
        sample(11, 0, "rollout-a", 2, 41),
        sample(90, 1, "sentinel-a", 1, 99),
        sample(91, 1, "sentinel-b", 1, 100),
    ]
    source_queue: queue.Queue = queue.Queue()
    for item in items:
        source_queue.put(item)

    queue_metrics = defaultdict(list)
    source = RolloutQueueDataset(
        rollout_queue=source_queue,
        model_version_fn=lambda: current_version[0],
        check_health_fn=lambda _timeout: None,
        stale_after_s=60.0,
        metrics=queue_metrics,
        max_staleness=0,
        poll_interval_s=0.01,
    )
    planner = TokenBudgetBatcher(source, num_processes=1, token_budget=2, metrics=defaultdict(list))
    production_collator = DataCollatorForRollout(
        pad_token_id=0,
        num_processes=1,
        groups_trained=set(),
        metrics=defaultdict(list),
        token_budget=2,
    )
    trace = {"version": current_version, "collation": [], "training_batches": [], "training_versions": []}

    def recording_collator(batch):
        planner_batch = batch[0]
        ids = [[int(token) for token in row[0].input_ids] for row in planner_batch]
        trace["collation"].append(
            {
                "ids": ids,
                "current_version": current_version[0],
                "sample_versions": [[sample.model_version for sample in row] for row in planner_batch],
            }
        )
        return production_collator(batch)

    steps = 2 if gradient_accumulation_steps == 1 else 1
    with TemporaryDirectory(prefix="vare-trainer-window-") as output_dir:
        args = TrainingArguments(
            output_dir=output_dir,
            max_steps=steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=0.001,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        model = ScalarModel()
        weights = [float(model.weight.detach().cpu())]
        trainer = RecordingTrainer(
            model=model,
            args=args,
            train_dataset=planner,
            data_collator=recording_collator,
            trace=trace,
            callbacks=[VersionCallback(current_version, weights)],
        )
        trainer.train()
        final_weight = float(model.weight.detach().cpu())
        trace["weights_after_optimizer_steps"] = weights
        trace["final_weight"] = final_weight
        trace["gradient_accumulation_steps"] = gradient_accumulation_steps
        trace["max_staleness"] = 0
        trace["stale_rows_dropped"] = len(queue_metrics["sample/dropped_stale_total"])
        trace["global_steps"] = trainer.state.global_step
        planner_iter = getattr(trainer, "_vare_planner_iterator", None)
        if planner_iter is not None:
            planner_iter.close()
        return trace


def main() -> None:
    print(json.dumps({"ga_1": run(1), "ga_2": run(2)}, sort_keys=True))


if __name__ == "__main__":
    main()
