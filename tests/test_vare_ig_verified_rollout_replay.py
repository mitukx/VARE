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


@pytest.mark.parametrize("invalid_kind", [
    "mixed_prompt_ids",
    "logprob_length",
    "nonfinite_logprob",
    "untrusted_reward",
    "row_reward_disagreement",
    "non_json_metadata",
    "nonfinite_nested_metadata",
])
def test_invalid_group_preflight_is_failure_atomic_and_retryable(tmp_path: Path, invalid_kind: str):
    """Malformed generation evidence must not enroll any durable trajectory."""
    task = TRAIN_TASKS[0]
    verifier = "b" * 64
    samples, rows = _group_rows(task, verifier)
    group_id = f"vareig:negative:{invalid_kind}:train-step-0:0:task"
    bridge = VerifiedGroupReplay(
        tmp_path, run_id=f"negative-{invalid_kind}", group_budget=1,
        group_size=4, replay_seed=0,
    )
    if invalid_kind == "mixed_prompt_ids":
        samples[2].prompt_ids = [999, 998]
    elif invalid_kind == "logprob_length":
        rows[2]["sampler_logprobs"] = [-0.2]
    elif invalid_kind == "nonfinite_logprob":
        rows[2]["sampler_logprobs"] = [float("nan"), -0.5]
    elif invalid_kind == "untrusted_reward":
        rows[2]["reward_detail"]["trusted_contract"] = False
    elif invalid_kind == "row_reward_disagreement":
        rows[2]["reward"] = 0.0
    elif invalid_kind == "non_json_metadata":
        rows[2]["sampling_distribution"] = object()
    elif invalid_kind == "nonfinite_nested_metadata":
        rows[2]["inference_profile"] = {"prefill_seconds": float("nan")}

    def submit(current_samples, current_rows):
        return bridge.add_group(
            task=task,
            family=FAMILY_BY_TASK[task.task_id],
            prompt_messages=[{"role": "user", "content": task.prompt}],
            samples=current_samples,
            rows=current_rows,
            policy_id="qwen3-4b:test-policy",
            policy_version=0,
            policy_checkpoint_sha256="c" * 64,
            group_id=group_id,
        )

    with pytest.raises(ValueError):
        submit(samples, rows)
    assert bridge.store.group(group_id) == []
    assert len(bridge.replay) == 0

    clean_samples, clean_rows = _group_rows(task, verifier)
    record = submit(clean_samples, clean_rows)
    assert record["replay_admitted"] is True
    assert len(bridge.store.group(group_id)) == 4
    assert audit_trajectory_store(tmp_path / "trajectory-store.sqlite3")["pass"] is True


def test_row_reward_mismatch_cannot_fabricate_an_informative_group(tmp_path: Path):
    """A forged persisted reward must not create a nonzero GRPO advantage."""
    from vare.grpo_mlx import group_advantages

    task = TRAIN_TASKS[0]
    samples, rows = _group_rows(task, "b" * 64)
    for sample, row in zip(samples, rows, strict=True):
        sample.reward = 0.0
        row["reward"] = 0.0
        row["reward_detail"]["passed"] = False
    assert group_advantages([sample.reward for sample in samples])[1] is False
    rows[3]["reward"] = 1.0
    assert group_advantages([float(row["reward"]) for row in rows])[1] is True

    bridge = VerifiedGroupReplay(
        tmp_path, run_id="forged-informative-group",
        group_budget=1, group_size=4, replay_seed=0,
    )
    group_id = "vareig:forged-reward:train-step-0:0:task"
    with pytest.raises(ValueError, match="persisted reward and sample reward disagree"):
        bridge.add_group(
            task=task,
            family=FAMILY_BY_TASK[task.task_id],
            prompt_messages=[{"role": "user", "content": task.prompt}],
            samples=samples, rows=rows,
            policy_id="qwen3-4b:test-policy", policy_version=0,
            policy_checkpoint_sha256="c" * 64,
            group_id=group_id,
        )
    assert bridge.store.group(group_id) == []
    assert len(bridge.replay) == 0
