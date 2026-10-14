import hashlib
import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vare.integrations.rvl import RVLTokenReplayReader
from vare.integrations.rvl_grpo import RVLGRPOHooks
from vare.integrations.rvl_trajectory import rvl_experiences_to_trajectory_group
from vare.trajectory_reference import TrajectoryContractError


def _make_replay(path):
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE groups(
          id TEXT PRIMARY KEY, policy_version INTEGER NOT NULL, payload TEXT NOT NULL,
          hash TEXT NOT NULL, status TEXT NOT NULL, verifier_version INTEGER NOT NULL,
          generated_at REAL NOT NULL, verified_at REAL, verification_token TEXT,
          verification_lease_until REAL, verification_attempts INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE verification_configuration(
          id INTEGER PRIMARY KEY CHECK(id=1), hash TEXT NOT NULL
        );
        """
    )
    db.execute(
        "INSERT INTO verification_configuration VALUES(1,?)", ("e" * 64,)
    )
    groups = {"group-a": [0.0, 0.0], "group-b": [1.0, 1.0]}
    for group_id, rewards in groups.items():
        items = []
        for index, reward in enumerate(rewards):
            token_logprobs = [-0.2 - 0.05 * index, -0.3]
            generation = {
                "prompt_id": "shared-task",
                "prompt": "Choose the correct answer.",
                "response": f"answer-{group_id}-{index}",
                "logprob": sum(token_logprobs),
                "token_count": 2,
                "latency_s": 0.02,
                "metadata": {
                    "family": "fixture",
                    "backend": "hf-local",
                    "model": "fixture/Qwen-revision-1",
                    "precision": "fp32",
                    "sampling_temperature": 0.8,
                    "logprob_distribution": "neutral_temperature_scaled_policy",
                    "prompt_token_ids": [1, 2, 3],
                    "response_token_ids": [101 + index, 201 + index],
                    "response_token_logprobs": token_logprobs,
                },
            }
            items.append(
                {
                    "generation": generation,
                    "reward": reward,
                    "verifier_version": 3,
                    "metadata": {"verifier_name": "fixture-exact", "trusted": True},
                }
            )
        behavior = []
        for item in items:
            generation = dict(item["generation"])
            generation.pop("latency_s", None)
            behavior.append(generation)
        behavior_hash = hashlib.sha256(
            json.dumps(
                behavior, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        db.execute(
            "INSERT INTO groups VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                group_id,
                7,
                json.dumps(items),
                behavior_hash,
                "ready",
                3,
                1.0,
                2.0,
                None,
                None,
                1,
            ),
        )
    db.commit()
    db.close()


def test_reader_to_rvl_hooks_and_trajectory_bridge_preserve_groups_and_tokens(tmp_path):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )
    assert len(experiences) == 4

    class Trainer:
        def snapshot_training_state(self):
            return {}

    hooks = RVLGRPOHooks(
        backend=object(),
        trainer=Trainer(),
        eval_tasks=[],
        score_fn=lambda _task, _response: 0.0,
        generation_factory=lambda **fields: SimpleNamespace(**fields),
        verified_generation_factory=lambda **fields: SimpleNamespace(**fields),
    )
    training_samples = [hooks._to_verified_generation(item) for item in experiences]
    assert training_samples[0].generation.metadata["response_token_ids"] == [101, 201]
    assert training_samples[0].generation.metadata["prompt_token_ids"] == [1, 2, 3]
    assert training_samples[0].reward == experiences[0].verification.score

    advantages = RVLGRPOHooks._group_relative_advantages(
        experiences,
        training_samples,
        eps=1e-6,
        clip=5.0,
    )
    # Same prompt ID appears in two distinct replay groups. Each group has
    # constant rewards, so both must normalize independently to zero.
    assert advantages == [0.0, 0.0, 0.0, 0.0]

    group_a = rvl_experiences_to_trajectory_group(experiences[:2])
    group_b = rvl_experiences_to_trajectory_group(experiences[2:])
    targets_a = group_a.materialize()
    targets_b = group_b.materialize()
    assert len(targets_a) == len(targets_b) == 4
    assert sum(target.weight for target in targets_a) == pytest.approx(1.0)
    assert {target.outcome_score for target in targets_a} == {0.0}
    assert {target.outcome_score for target in targets_b} == {1.0}
    assert {target.policy_id for target in targets_a} == {"rvl-policy-v7"}
    assert {target.model_id for target in targets_a} == {"fixture/Qwen-revision-1"}
    assert {target.sampling_temperature for target in targets_a} == {0.8}
    assert {target.behavior_logprob for target in targets_a} == {-0.2, -0.25, -0.3}


def test_rvl_trajectory_bridge_fails_closed_on_partial_group(tmp_path):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )
    with pytest.raises(TrajectoryContractError, match="incomplete relative"):
        rvl_experiences_to_trajectory_group(experiences[:1])


def test_rvl_trajectory_bridge_requires_behavior_distribution_and_verifier_config(tmp_path):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )

    experiences[0].attempt.metadata["rvl_generation"]["metadata"][
        "logprob_distribution"
    ] = "greedy_scores_not_trainable"
    with pytest.raises(TrajectoryContractError, match="not identified"):
        rvl_experiences_to_trajectory_group(experiences[:2])

    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )
    del experiences[0].verification.metadata["rvl_verification_config_sha256"]
    with pytest.raises(TrajectoryContractError, match="lacks verifier configuration"):
        rvl_experiences_to_trajectory_group(experiences[:2])


def test_rvl_trajectory_bridge_rejects_mixed_prompt_identities_inside_group(tmp_path):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )
    generation = experiences[1].attempt.metadata["rvl_generation"]
    generation["prompt_id"] = "different-task"
    generation["prompt"] = "A different prompt."
    experiences[1].attempt.task = replace(
        experiences[1].attempt.task,
        id="different-task",
        prompt="A different prompt.",
    )

    with pytest.raises(TrajectoryContractError, match="more than one prompt identity"):
        rvl_experiences_to_trajectory_group(experiences[:2])


def test_rvl_trajectory_bridge_rejects_mismatched_aggregate_logprob(tmp_path):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )
    experiences[0].attempt.metadata["rvl_generation"]["logprob"] = -9.0

    with pytest.raises(TrajectoryContractError, match="aggregate log-probability differs"):
        rvl_experiences_to_trajectory_group(experiences[:2])


def test_rvl_trajectory_bridge_requires_every_declared_generation_index(tmp_path):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )
    experiences[0].attempt.metadata["rvl_generation_index"] = 9

    with pytest.raises(TrajectoryContractError, match="indices do not cover every"):
        rvl_experiences_to_trajectory_group(experiences[:2])


@pytest.mark.parametrize("mutation", ["token_id", "generation_text"])
def test_rvl_trajectory_bridge_rejects_generation_edits_after_read(tmp_path, mutation):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )
    generation = experiences[0].attempt.metadata["rvl_generation"]
    if mutation == "token_id":
        generation["metadata"]["response_token_ids"][0] += 1000
    else:
        generation["response"] = "a different scored answer"
        experiences[0].attempt.output = generation["response"]

    with pytest.raises(TrajectoryContractError, match="changed after replay reading"):
        rvl_experiences_to_trajectory_group(experiences[:2])


def test_rvl_trajectory_bridge_rejects_reward_edits_after_read(tmp_path):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )
    experiences[0].verification.score = 0.75

    with pytest.raises(TrajectoryContractError, match="score changed after replay reading"):
        rvl_experiences_to_trajectory_group(experiences[:2])


def test_rvl_trajectory_bridge_rejects_group_hash_mismatch_from_sqlite(tmp_path):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    db = sqlite3.connect(path)
    row = db.execute(
        "SELECT rowid,payload FROM groups WHERE id='group-a'"
    ).fetchone()
    payload = json.loads(row[1])
    payload[0]["generation"]["metadata"]["response_token_ids"][0] += 1000
    db.execute("UPDATE groups SET payload=? WHERE rowid=?", (json.dumps(payload), row[0]))
    db.commit()
    db.close()

    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )
    with pytest.raises(TrajectoryContractError, match="does not match a supported replay encoding"):
        rvl_experiences_to_trajectory_group(experiences[:2])


def test_rvl_bridge_accepts_pinned_direct_verified_record_hash_encoding(tmp_path):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    db = sqlite3.connect(path)
    for group_id in ("group-a", "group-b"):
        raw = db.execute("SELECT payload FROM groups WHERE id=?", (group_id,)).fetchone()[0]
        items = json.loads(raw)
        verified_records = []
        for item in items:
            verified = json.loads(
                json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
            verified["generation"].pop("latency_s", None)
            verified.pop("verifier_latency_s", None)
            verified_records.append(verified)
        digest = hashlib.sha256(
            json.dumps(
                verified_records,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        db.execute("UPDATE groups SET hash=? WHERE id=?", (digest, group_id))
    db.commit()
    db.close()

    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )
    assert experiences[0].attempt.metadata["rvl_group_behavior_hash_status"] == (
        "verified_record_match"
    )
    assert rvl_experiences_to_trajectory_group(experiences[:2]).group_id == "group-a"


@pytest.mark.parametrize("field", ["attempt", "verification"])
def test_rvl_trajectory_bridge_cross_checks_duplicate_group_lineage(tmp_path, field):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )
    if field == "attempt":
        experiences[0].attempt.metadata["rvl_group_id"] = "foreign-group"
    else:
        experiences[0].verification.metadata["rvl_group_id"] = "foreign-group"

    expected = "attempt lineage" if field == "attempt" else "verifier lineage"
    with pytest.raises(TrajectoryContractError, match=expected):
        rvl_experiences_to_trajectory_group(experiences[:2])


def test_rvl_trajectory_bridge_rejects_stale_verifier_config_fingerprint(tmp_path):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=4,
    )

    with pytest.raises(TrajectoryContractError, match="not version-bound for stale groups"):
        rvl_experiences_to_trajectory_group(experiences[:2])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("policy_version", "mixes policy or behavior-distribution"),
        ("model", "mixes policy or behavior-distribution"),
        ("temperature", "mixes policy or behavior-distribution"),
        ("verifier_version", "mixes evaluator identities"),
        ("verifier_config", "mixes evaluator identities"),
    ],
)
def test_rvl_trajectory_bridge_rejects_mixed_group_provenance(
    tmp_path, mutation, message
):
    path = tmp_path / "token-replay.sqlite"
    _make_replay(path)
    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=7,
        current_verifier_version=3,
    )
    second = experiences[1]
    if mutation == "policy_version":
        second.attempt.policy_version += 1
    elif mutation == "model":
        second.attempt.metadata["rvl_generation"]["metadata"]["model"] = "other-model"
    elif mutation == "temperature":
        second.attempt.metadata["rvl_generation"]["metadata"][
            "sampling_temperature"
        ] = 0.7
    elif mutation == "verifier_version":
        second.verification.verifier_version += 1
    else:
        second.verification.metadata["rvl_verification_config_sha256"] = "d" * 64

    with pytest.raises(TrajectoryContractError, match=message):
        rvl_experiences_to_trajectory_group(experiences[:2])
