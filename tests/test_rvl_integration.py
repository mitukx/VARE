import json
import sqlite3
from pathlib import Path

import pytest

from vare.integrations.rvl import RVLTokenReplayReader


def _make_replay(path: Path) -> None:
    db = sqlite3.connect(path)
    db.execute(
        """CREATE TABLE groups(
        id TEXT PRIMARY KEY, policy_version INTEGER NOT NULL, payload TEXT NOT NULL,
        hash TEXT NOT NULL, status TEXT NOT NULL, verifier_version INTEGER NOT NULL,
        generated_at REAL NOT NULL, verified_at REAL, verification_token TEXT,
        verification_lease_until REAL, verification_attempts INTEGER NOT NULL DEFAULT 0)"""
    )
    ready = [
        {
            "generation": {
                "prompt_id": "p1",
                "prompt": "2+2?",
                "response": "#### 4",
                "logprob": -0.2,
                "token_count": 1,
                "latency_s": 0.05,
                "metadata": {
                    "family": "math",
                    "model": "fixture-model-revision",
                    "sampling_temperature": 0.8,
                    "logprob_distribution": "neutral_temperature_scaled_policy",
                    "prompt_token_ids": [1, 2],
                    "response_token_ids": [10],
                    "response_token_logprobs": [-0.2],
                },
            },
            "reward": 1.0,
            "verifier_latency_s": 0.01,
            "verifier_version": 3,
            "metadata": {"trusted": True, "verifier_name": "exact"},
        },
        {
            "generation": {
                "prompt_id": "p2",
                "prompt": "3+5?",
                "response": "#### 7",
                "logprob": -1.2,
                "token_count": 1,
                "latency_s": 0.07,
                "metadata": {
                    "family": "math",
                    "model": "fixture-model-revision",
                    "sampling_temperature": 0.8,
                    "logprob_distribution": "neutral_temperature_scaled_policy",
                    "prompt_token_ids": [3, 4],
                    "response_token_ids": [11],
                    "response_token_logprobs": [-1.2],
                },
            },
            "reward": 0.0,
            "verifier_latency_s": 0.01,
            "verifier_version": 3,
            "metadata": {"trusted": True, "verifier_name": "exact"},
        },
    ]
    db.execute(
        "INSERT INTO groups VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        ("g1", 4, json.dumps(ready), "x", "ready", 3, 1.0, 2.0, None, None, 0),
    )
    db.execute(
        "CREATE TABLE verification_configuration(id INTEGER PRIMARY KEY, hash TEXT NOT NULL)"
    )
    db.execute(
        "INSERT INTO verification_configuration VALUES(1,?)",
        ("c" * 64,),
    )
    pending = [
        {
            "prompt_id": "p3",
            "prompt": "1+1?",
            "response": "#### 2",
            "logprob": -0.1,
            "token_count": 3,
            "latency_s": 0.03,
            "metadata": {"family": "math"},
        }
    ]
    db.execute(
        "INSERT INTO groups VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        ("g2", 5, json.dumps(pending), "y", "pending_verification", -1, 3.0, None, None, None, 0),
    )
    db.commit()
    db.close()


def test_rvl_reader_preserves_versions_and_behavior(tmp_path):
    path = tmp_path / "replay.sqlite"
    _make_replay(path)
    reader = RVLTokenReplayReader(path)
    snap = reader.snapshot(current_policy_version=6, current_verifier_version=5)
    assert snap.groups == 2
    assert snap.generations == 3
    assert snap.statuses == {"ready": 1, "pending_verification": 1}
    assert snap.max_policy_lag == 2
    assert snap.max_verifier_lag == 2
    assert snap.mean_reward == 0.5
    exps = reader.ready_experiences(current_policy_version=6, current_verifier_version=5)
    assert len(exps) == 2
    assert exps[0].policy_lag == 2
    assert exps[0].verifier_lag == 2
    assert exps[0].verification.score == 1.0
    assert exps[0].attempt.logprob == -0.2
    assert exps[0].attempt.task.family == "math"
    assert exps[0].verification.trusted
    assert exps[1].verification.score == 0.0
    assert not exps[1].verification.passed
    assert exps[0].attempt.metadata["vare_rollout_group"] == "g1"
    assert exps[0].attempt.metadata["vare_rollout_group_size"] == 2
    assert exps[0].attempt.metadata["rvl_generation"]["metadata"]["prompt_token_ids"] == [1, 2]
    assert exps[0].verification.metadata["rvl_replay_status"] == "ready"
    assert exps[0].verification.metadata["rvl_verification_config_sha256"] == "c" * 64


def test_rvl_reader_accepts_legacy_database_without_verification_config_table(tmp_path):
    path = tmp_path / "legacy-replay.sqlite"
    _make_replay(path)
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE verification_configuration")

    experiences = RVLTokenReplayReader(path).ready_experiences(
        current_policy_version=6,
        current_verifier_version=5,
    )
    assert len(experiences) == 2
    assert "rvl_verification_config_sha256" not in experiences[0].verification.metadata


def test_rvl_reader_rejects_malformed_verification_config_digest(tmp_path):
    path = tmp_path / "malformed-replay.sqlite"
    _make_replay(path)
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE verification_configuration SET hash=? WHERE id=1",
            ("A" * 64,),
        )

    with pytest.raises(ValueError, match="lowercase SHA-256 digest"):
        RVLTokenReplayReader(path).ready_experiences(
            current_policy_version=6,
            current_verifier_version=5,
        )


@pytest.mark.parametrize(
    "reward",
    [None, True, False, float("nan"), float("inf"), -0.01, 1.01, "0.5"],
)
def test_rvl_reader_rejects_invalid_ready_rewards(tmp_path, reward):
    path = tmp_path / "replay.sqlite"
    _make_replay(path)
    with sqlite3.connect(path) as db:
        raw = db.execute("SELECT payload FROM groups WHERE id='g1'").fetchone()[0]
        payload = json.loads(raw)
        payload[0]["reward"] = reward
        db.execute(
            "UPDATE groups SET payload=? WHERE id='g1'",
            (json.dumps(payload),),
        )

    reader = RVLTokenReplayReader(path)
    with pytest.raises(ValueError, match="ready RVL experience reward"):
        reader.snapshot(current_policy_version=6, current_verifier_version=5)
    with pytest.raises(ValueError, match="ready RVL experience reward"):
        reader.ready_experiences(current_policy_version=6, current_verifier_version=5)


def test_rvl_reader_does_not_default_missing_ready_reward_to_zero(tmp_path):
    path = tmp_path / "replay.sqlite"
    _make_replay(path)
    with sqlite3.connect(path) as db:
        raw = db.execute("SELECT payload FROM groups WHERE id='g1'").fetchone()[0]
        payload = json.loads(raw)
        del payload[0]["reward"]
        db.execute(
            "UPDATE groups SET payload=? WHERE id='g1'",
            (json.dumps(payload),),
        )

    reader = RVLTokenReplayReader(path)
    with pytest.raises(ValueError, match="ready RVL experience reward"):
        reader.snapshot(current_policy_version=6, current_verifier_version=5)
    with pytest.raises(ValueError, match="ready RVL experience reward"):
        reader.ready_experiences(current_policy_version=6, current_verifier_version=5)


def test_rvl_reader_rejects_empty_group_marked_ready(tmp_path):
    path = tmp_path / "replay.sqlite"
    _make_replay(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE groups SET payload='[]' WHERE id='g1'")

    reader = RVLTokenReplayReader(path)
    with pytest.raises(ValueError, match="ready RVL group g1 has no experiences"):
        reader.snapshot(current_policy_version=6, current_verifier_version=5)
    with pytest.raises(ValueError, match="ready RVL group g1 has no experiences"):
        reader.ready_experiences(current_policy_version=6, current_verifier_version=5)


@pytest.mark.parametrize(
    ("column", "value", "expected_message"),
    [
        ("policy_version", 7, "stored policy version 7 is ahead of current version 6"),
        ("verifier_version", 6, "stored verifier version 6 is ahead of current version 5"),
    ],
)
def test_rvl_reader_rejects_future_ready_group_versions(
    tmp_path, column, value, expected_message
):
    path = tmp_path / "replay.sqlite"
    _make_replay(path)
    with sqlite3.connect(path) as db:
        db.execute(f"UPDATE groups SET {column}=? WHERE id='g1'", (value,))
    reader = RVLTokenReplayReader(path)
    with pytest.raises(ValueError, match=expected_message):
        reader.snapshot(current_policy_version=6, current_verifier_version=5)
    with pytest.raises(ValueError, match=expected_message):
        reader.ready_experiences(current_policy_version=6, current_verifier_version=5)


def test_rvl_reader_rejects_future_per_experience_verifier_version(tmp_path):
    path = tmp_path / "replay.sqlite"
    _make_replay(path)
    with sqlite3.connect(path) as db:
        raw = db.execute("SELECT payload FROM groups WHERE id='g1'").fetchone()[0]
        payload = json.loads(raw)
        payload[0]["verifier_version"] = 6
        db.execute(
            "UPDATE groups SET payload=? WHERE id='g1'",
            (json.dumps(payload),),
        )
    reader = RVLTokenReplayReader(path)
    with pytest.raises(ValueError, match="stored experience verifier version 6 is ahead"):
        reader.snapshot(current_policy_version=6, current_verifier_version=5)
    with pytest.raises(ValueError, match="stored experience verifier version 6 is ahead"):
        reader.ready_experiences(current_policy_version=6, current_verifier_version=5)


def test_rvl_reader_rejects_malformed_ready_experience_version(tmp_path):
    path = tmp_path / "replay.sqlite"
    _make_replay(path)
    with sqlite3.connect(path) as db:
        raw = db.execute("SELECT payload FROM groups WHERE id='g1'").fetchone()[0]
        payload = json.loads(raw)
        payload[0]["verifier_version"] = True
        db.execute(
            "UPDATE groups SET payload=? WHERE id='g1'",
            (json.dumps(payload),),
        )
    reader = RVLTokenReplayReader(path)
    with pytest.raises(ValueError, match="stored experience verifier must be a non-negative integer"):
        reader.ready_experiences(current_policy_version=6, current_verifier_version=5)
