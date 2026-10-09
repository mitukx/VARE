# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

import queue
from collections import defaultdict
from types import SimpleNamespace

from accelerate import PartialState

from trl.experimental.async_grpo.async_grpo_trainer import RolloutQueueDataset


def _sample(row_id, *, model_version, group_id, rollout_id=None, rollout_size=1):
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "fixture"}],
        completion=[{"role": "assistant", "content": f"row-{row_id}"}],
        input_ids=[row_id, row_id + 10],
        completion_mask=[0, 1],
        old_log_probs=[0.0, -0.25],
        advantage=float(row_id),
        model_version=model_version,
        group_id=group_id,
        metrics={"reward": float(row_id)},
        enqueued_at=None,
        rollout_id=rollout_id,
        rollout_size=rollout_size,
    )


def _dataset(items, version, metrics):
    data_queue = queue.Queue()
    for item in items:
        data_queue.put(item)
    dataset = RolloutQueueDataset(
        rollout_queue=data_queue,
        model_version_fn=lambda: version[0],
        check_health_fn=lambda _timeout: None,
        stale_after_s=60.0,
        metrics=metrics,
        max_staleness=0,
        poll_interval_s=0.01,
    )
    return iter(dataset)


def test_policy_update_between_reads_preserves_every_fork_of_admitted_rollout():
    PartialState()
    version = [0]
    metrics = defaultdict(list)
    fork_rows = [
        _sample(0, model_version=0, group_id=41, rollout_id="rollout-a", rollout_size=2),
        _sample(1, model_version=0, group_id=41, rollout_id="rollout-a", rollout_size=2),
    ]
    other_rollout = _sample(2, model_version=0, group_id=41, rollout_id="rollout-b")
    sentinel = _sample(99, model_version=1, group_id=99)
    sentinel_2 = _sample(100, model_version=1, group_id=100)
    dataset_iter = _dataset([*fork_rows, other_rollout, sentinel, sentinel_2], version, metrics)

    first = next(dataset_iter)
    version[0] = 1
    sibling = next(dataset_iter)
    after_drops = next(dataset_iter)
    dataset_iter.close()

    assert first["input_ids"][0] == 0
    assert sibling["input_ids"][0] == 1
    assert first["group_id"] == sibling["group_id"] == 41
    assert after_drops["group_id"] == 99
    assert metrics["sample/dropped_stale_total"] == [1.0]


def test_stale_forked_rollout_is_dropped_as_one_admission_unit():
    PartialState()
    metrics = defaultdict(list)
    fork_rows = [
        _sample(0, model_version=0, group_id=41, rollout_id="rollout-a", rollout_size=2),
        _sample(1, model_version=0, group_id=41, rollout_id="rollout-a", rollout_size=2),
    ]
    sentinel = _sample(99, model_version=1, group_id=99)
    dataset_iter = _dataset([*fork_rows, sentinel], [1], metrics)

    accepted = next(dataset_iter)
    dataset_iter.close()

    assert accepted["group_id"] == 99
    assert metrics["sample/dropped_stale_total"] == [1.0, 1.0]


def test_fork_metadata_mismatch_fails_closed():
    PartialState()
    first = _sample(0, model_version=0, group_id=41, rollout_id="rollout-a", rollout_size=2)
    second = _sample(1, model_version=0, group_id=41, rollout_id="rollout-b", rollout_size=2)
    dataset_iter = _dataset([first, second], [0], defaultdict(list))

    try:
        next(dataset_iter)
    except RuntimeError as error:
        assert "membership is not contiguous or has inconsistent metadata" in str(error)
    else:
        raise AssertionError("inconsistent rollout membership was accepted")
    finally:
        dataset_iter.close()
