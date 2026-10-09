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
                "token_count": 3,
                "latency_s": 0.05,
                "metadata": {"family": "math"},
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
                "token_count": 3,
                "latency_s": 0.07,
                "metadata": {"family": "math"},
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
    exps = reader.ready_experiences(current_policy_version=6, current_verifier_version=5)
    assert len(exps) == 2
    assert exps[0].policy_lag == 2
    assert exps[0].verifier_lag == 2
    assert exps[0].attempt.logprob == -0.2
    assert exps[0].attempt.task.family == "math"
    assert exps[0].verification.trusted
    assert not exps[1].verification.passed


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
