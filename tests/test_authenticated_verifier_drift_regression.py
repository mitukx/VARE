"""Deterministic replay regression; generated members are fixtures, not on-policy data."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from scripts.benchmark_verifier_drift_scenarios import authoritative_score, scorer_for
from scripts.benchmark_verifier_consistent_async_grpo import GROUP_SIZE
from test_rvl_grpo_group_boundary_calibration import CapturingTrainer, _hooks
from vare.evidence import sha256_json
from vare.trajectory_admission import TrajectoryRewardRoot, admit_terminal_grpo_group
from vare.trajectory_store import AgentStep, PolicyCheckpoint, TrajectoryStore, audit_trajectory_store
from vare.types import Attempt, Experience, Task, Verification


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "benchmarks/verifier_consistent_async_grpo_v7/protocol.json").read_text())
SCENARIO = next(row for row in PROTOCOL["scenarios"] if row["id"] == "false_accept")
TASK = Task("drift-fixture", "A bird landed on the", "landing",
            {"learning_rule": "terminal_action_on_policy_grpo_v1"})
POLICY = PolicyCheckpoint("policy-0", 0, "a" * 64)
OUTPUTS = ["grass.", "ground", "rock", "floor.", "branch", "lake", "ground.", "trees"]
RUN_ID = "authenticated-verifier-drift-fixture-v1"


class CountingTrainer(CapturingTrainer):
    def __init__(self):
        super().__init__()
        self.train_step_calls = 0

    def train_step(self, samples, *, advantages=None):
        self.train_step_calls += 1
        super().train_step(samples, advantages=advantages)


def _identity(version: int) -> str:
    key = "v0" if version == 0 else "v1"
    return sha256_json({"scenario": "false_accept", "version": version,
                        "rule": SCENARIO["verifiers"][key]["rule"]})


def _write_group(store, root, verifier_shas, group_id: str, versions: list[int]) -> None:
    ids = [f"{group_id}-{index:02d}" for index in range(GROUP_SIZE)]
    for member, trajectory_id in enumerate(ids):
        store.submit(trajectory_id, TASK, expected_steps=1, group_id=group_id,
                     member=member, group_size=GROUP_SIZE)
    for member, trajectory_id in enumerate(ids):
        lease = store.claim("execute", "fixture-generator")
        output = OUTPUTS[member]
        generation = {
            "prompt_id": TASK.id, "prompt": TASK.prompt, "response": output,
            "logprob": -0.1, "token_count": 1, "latency_s": 0.001,
            "metadata": {
                "model": "fixture-policy-0", "backend": "fixture",
                "sampling_temperature": 0.8,
                "logprob_distribution": "neutral_temperature_scaled_policy",
                "prompt_token_ids": [101, 102], "response_token_ids": [member + 201],
                "response_token_logprobs": [-0.1],
            },
        }
        store.append_step(lease, AgentStep(
            index=0, event_id=f"{trajectory_id}-terminal", policy=POLICY,
            action={"kind": "inference"}, observation={}, state={}, generation=generation,
        ))
        store.finish(lease)
    for member, trajectory_id in enumerate(ids):
        lease = store.claim("verify", "fixture-verifier")
        row = store.trajectory(trajectory_id)
        version = versions[member]
        score = scorer_for(SCENARIO, version, OUTPUTS[member])
        value = Verification(
            score, score >= 0.5, 1.0, version, "landing", trusted=True,
            metadata={"trajectory_sha256": row["head"],
                      "verifier_config_sha256": verifier_shas[version]},
        )
        store.record_verification(lease, root.attest(trajectory_id, row["head"], value))


def _signed_experiences(store, root, group_id: str) -> list[Experience]:
    result = []
    for row in store.group(group_id):
        step = list(store.iter_steps(row["id"]))[-1]
        verification = Verification(**json.loads(row["verification"]))
        assert root.verifies(row["id"], row["head"], verification)
        task_spec = json.loads(row["task"])["task"]
        task = Task(**task_spec)
        generation = step["generation"]
        attempt = Attempt(
            task, generation["response"], POLICY.policy_id, POLICY.version, 0,
            logprob=generation["logprob"], metadata={
                "vare_rollout_group": group_id,
                "vare_rollout_group_size": GROUP_SIZE,
                "rvl_generation": generation,
            },
        )
        result.append(Experience(attempt, verification, 0, 0, 0.0))
    return result


def test_authenticated_verifier_drift_replay_fails_closed_and_retains_fresh_group(tmp_path):
    store = TrajectoryStore.create(tmp_path / "drift.sqlite", run_id=RUN_ID,
                                   max_inflight=GROUP_SIZE * 3)
    root = TrajectoryRewardRoot(
        b"k" * 32, RUN_ID,
        hashlib.sha256((ROOT / "benchmarks/verifier_consistent_async_grpo_v7/protocol.json")
                       .read_bytes()).hexdigest(),
    )
    store.register_policy(POLICY)
    verifier_shas = {version: _identity(version) for version in (0, 1)}
    for version, digest in verifier_shas.items():
        store.register_verifier("landing", version, digest)

    _write_group(store, root, verifier_shas, "stale-v0", [0] * GROUP_SIZE)
    _write_group(store, root, verifier_shas, "fresh-v1", [1] * GROUP_SIZE)
    _write_group(store, root, verifier_shas, "mixed-v0-v1",
                 [0] * (GROUP_SIZE // 2) + [1] * (GROUP_SIZE // 2))

    false_accepts = 0
    for group_id in ("stale-v0", "fresh-v1", "mixed-v0-v1"):
        for row in store.group(group_id):
            output = list(store.iter_steps(row["id"]))[-1]["generation"]["response"]
            signed = Verification(**json.loads(row["verification"]))
            authoritative = authoritative_score(SCENARIO, output)
            false_accepts += int(signed.score > 0 and authoritative == 0)
            if signed.verifier_version == 1:
                assert signed.score == authoritative
            assert root.verifies(row["id"], row["head"], signed)
    assert false_accepts > 0

    admitted = {
        group_id: admit_terminal_grpo_group(
            store, group_id, active_policy=POLICY,
            verifier_name="landing", verifier_version=1,
            verifier_config_sha256=verifier_shas[1], trust_root=root,
        )
        for group_id in ("stale-v0", "fresh-v1", "mixed-v0-v1")
    }
    assert not admitted["stale-v0"].admitted
    assert "verifier_stale_or_mismatched" in admitted["stale-v0"].reasons
    assert admitted["fresh-v1"].admitted and len(admitted["fresh-v1"].experiences) == GROUP_SIZE
    assert not admitted["mixed-v0-v1"].admitted
    assert "verifier_stale_or_mismatched" in admitted["mixed-v0-v1"].reasons

    mixed_experiences = _signed_experiences(store, root, "mixed-v0-v1")
    trainer = CountingTrainer()
    hooks = _hooks(trainer, current_verifier_version=1)
    with pytest.raises(ValueError, match="VARE rollout group mixes"):
        asyncio.run(hooks.train_candidate("policy-0", mixed_experiences))
    assert trainer.train_step_calls == 0

    fresh = admitted["fresh-v1"]
    store.close_group("stale-v0", admitted["stale-v0"].heads, reason="quarantined",
                      decision={"reasons": list(admitted["stale-v0"].reasons)})
    store.close_group("mixed-v0-v1", admitted["mixed-v0-v1"].heads,
                      reason="quarantined",
                      decision={"reasons": list(admitted["mixed-v0-v1"].reasons)})
    asyncio.run(hooks.train_candidate("policy-0", list(fresh.experiences)))
    assert trainer.train_step_calls == 1
    assert store.counts() == {"quarantined": GROUP_SIZE * 2, "verified": GROUP_SIZE}
    assert all(row["state"] == "verified" for row in store.group("fresh-v1"))
    assert audit_trajectory_store(store.path)["pass"] is True
