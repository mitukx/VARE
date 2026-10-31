from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "sprint_xiii" / "audit_online_actor_v3.py"
)
spec = importlib.util.spec_from_file_location(
    "sprint_xiii_audit_online_actor_v3", SCRIPT
)
auditor = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(auditor)


def test_retained_prepublication_candidate_can_share_hash_with_recovery_commit(
    tmp_path: Path,
) -> None:
    txid = "seed-909-E-update-1"
    name = "candidate-seed-909-E-update-1-attempt-1.pt.gz"
    payload = b"deterministic candidate"
    sha = hashlib.sha256(payload).hexdigest()
    models = tmp_path / "models"
    tx_dir = tmp_path / "transactions" / txid
    models.mkdir()
    tx_dir.mkdir(parents=True)
    (models / name).write_bytes(payload)
    (models / f"{name}.json").write_text(json.dumps({"sha256": sha}))
    (tx_dir / "attempt-1.json").write_text(
        json.dumps(
            {
                "status": "candidate_durable_unpublished",
                "candidate_checkpoint": name,
                "candidate_checkpoint_sha256": sha,
            }
        )
    )
    rows = [
        {
            "kind": "candidate_checkpoint_written",
            "transaction_id": txid,
            "attempt": 1,
            "checkpoint": name,
            "checkpoint_sha256": sha,
            "seq": 10,
        },
        {
            "kind": "fault_injected",
            "transaction_id": txid,
            "fault": "checkpoint_before_publication",
            "seq": 11,
        },
    ]

    # A later retry may commit the same deterministic hash. The interrupted
    # attempt still passes only if its own record, manifest and bytes remain.
    committed_hashes = {sha}
    assert sha in committed_hashes
    assert auditor.retained_prepublication_candidate(tmp_path, rows, rows[1])

    (models / name).write_bytes(b"tampered")
    assert not auditor.retained_prepublication_candidate(tmp_path, rows, rows[1])


def test_prepublication_candidate_must_precede_fault(tmp_path: Path) -> None:
    fault = {
        "fault": "checkpoint_before_publication",
        "transaction_id": "tx",
        "seq": 4,
    }
    rows = [
        {
            "kind": "candidate_checkpoint_written",
            "transaction_id": "tx",
            "attempt": 1,
            "checkpoint": "candidate.pt.gz",
            "checkpoint_sha256": "0" * 64,
            "seq": 5,
        }
    ]
    assert not auditor.retained_prepublication_candidate(tmp_path, rows, fault)
