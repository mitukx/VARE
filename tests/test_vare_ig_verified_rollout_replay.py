from __future__ import annotations

import json
from pathlib import Path

import pytest

from vare.grpo_mlx import Sample
from vare.learning_pipeline.vare_ig_tasks import TRAIN_TASKS, FAMILY_BY_TASK
from vare.learning_pipeline.verified_rollout_replay import (
    VerifiedGroupReplay,
    acknowledge_durable_decision,
)
from vare.trajectory_store import audit_trajectory_store


def _group_rows(task, verifier: str):
    samples = []
    rows = []
    for index, reward in enumerate((1.0, 0.0, 1.0, 0.0)):
        samples.append(Sample(
            instance_id=task.task_id,
            behavior_checkpoint_id="qwen3-4b:test-policy",
            behavior_step=0,
            prompt_ids=[11, 12, 13],
            completion_ids=[21 + index, 31],
            reward=reward,
            verifier=verifier,
            sample_index=index,
            meta={"task_sha256": task.sha256(), "phase": "train-step-0"},
        ))
        rows.append({
            "instance_id": task.task_id,
            "sample_index": index,
            "seed": 100 + index,
            "phase": "train-step-0",
            "response_text": f"def solve(value): return {index}",
            "sampler_logprobs": [-0.25, -0.5],
            "generation_seconds": 0.1,
            "temperature": 1.0,
            "sampling_distribution": "untruncated softmax(logits / temperature)",
            "finish": "stop",
            "candidate_source_sha256": f"{index:064x}",
            "reward": reward,
            "reward_detail": {
                "trusted_contract": True,
                "passed": reward == 1.0,
                "grade_status": "ok",
                "cases": [{"case_id": "case-1", "passed": reward == 1.0}],
                "case_set_sha256": "a" * 64,
            },
        })
    return samples, rows


def test_real_rollout_bridge_persists_verifies_replays_and_acknowledges(tmp_path: Path):
    task = TRAIN_TASKS[0]
    verifier = "b" * 64
    checkpoint_sha = "c" * 64
    samples, rows = _group_rows(task, verifier)
    bridge = VerifiedGroupReplay(
        tmp_path,
        run_id="vare-ig-test-run",
        group_budget=1,
        group_size=4,
        replay_seed=9,
    )
    record = bridge.add_group(
        task=task,
        family=FAMILY_BY_TASK[task.task_id],
        prompt_messages=[{"role": "user", "content": task.prompt}],
        samples=samples,
        rows=rows,
        policy_id="qwen3-4b:test-policy",
        policy_version=0,
        policy_checkpoint_sha256=checkpoint_sha,
        group_id="vareig:test:0:train-step-0:0:task",
    )
    replayed, replay_rows = bridge.sample_current(
        count=4,
        policy_id="qwen3-4b:test-policy",
        policy_version=0,
        policy_checkpoint_sha256=checkpoint_sha,
    )

    assert record["replay_admitted"] is True
    assert [sample.reward for sample in replayed].count(1.0) == 2
    assert {sample.sample_index for sample in replayed} == {0, 1, 2, 3}
    assert all(row["sampler_logprobs"] == [-0.25, -0.5] for row in replay_rows)
    assert all(row["reward_detail"]["trusted_contract"] for row in replay_rows)
    assert {row["state"] for row in bridge.store.group(record["group_id"])} == {"verified"}

    with pytest.raises(RuntimeError, match="freshness filtering"):
        bridge.sample_current(
            count=4,
            policy_id="stale-checkpoint",
            policy_version=0,
            policy_checkpoint_sha256=checkpoint_sha,
        )

    selection_path = tmp_path / "scheduler-selection.json"
    selection_path.write_text(json.dumps({"trajectory_groups": [record]}), encoding="utf-8")
    ack_path = tmp_path / "learner-events.json"
    ack_path.write_text(json.dumps([{"event": "optimizer_update_committed", "update_step": 1}]),
                        encoding="utf-8")
    decision = {"update_applied": True, "effective_update_count": 1}
    first = acknowledge_durable_decision(
        tmp_path, update_step=1, acknowledgment_path=ack_path, decision=decision,
    )
    second = acknowledge_durable_decision(
        tmp_path, update_step=1, acknowledgment_path=ack_path, decision=decision,
    )

    assert first == second
    assert {row["state"] for row in bridge.store.group(record["group_id"])} == {"consumed"}
    assert audit_trajectory_store(tmp_path / "trajectory-store.sqlite3")["pass"] is True
