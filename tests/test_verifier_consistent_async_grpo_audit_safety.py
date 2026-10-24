from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from scripts.audit_verifier_consistent_async_grpo import (
    audit_durable_promotion,
    audit_recovered_consumption,
    max_error,
    require_behavior_alignment,
    require_crash_correction_manifest,
    validate_cutpoint_inventory,
    verify_manifest,
)
from scripts.benchmark_verifier_consistent_async_grpo import max_tensor_error


def _cutpoints() -> list[dict[str, str]]:
    return [{"cutpoint": name} for name in (
        "rollout_complete_before_admission",
        "admission_before_optimizer",
        "optimizer_computed_before_commit",
        "commit_before_acknowledgment",
        "evaluation_before_promotion",
    )]


def test_cutpoint_inventory_requires_every_declared_phase_exactly_once():
    expected = validate_cutpoint_inventory(_cutpoints(), process_kills=5)

    assert len(expected) == 5


def test_cutpoint_inventory_rejects_duplicate_that_omits_a_phase():
    repeated = _cutpoints()
    repeated[-1] = {"cutpoint": "rollout_complete_before_admission"}

    with pytest.raises(RuntimeError, match="repeats"):
        validate_cutpoint_inventory(repeated, process_kills=5)


def test_cutpoint_inventory_rejects_malformed_names():
    malformed = _cutpoints()
    malformed[-1] = {"cutpoint": []}

    with pytest.raises(RuntimeError, match="malformed cutpoint"):
        validate_cutpoint_inventory(malformed, process_kills=5)


def test_behavior_alignment_enforces_frozen_limit_and_finiteness():
    require_behavior_alignment(9.9e-5)

    for error in (1e-4, float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="behavior log-probability mismatch"):
            require_behavior_alignment(error)


def _consumption_snapshot(tmp_path, states=None, ack_heads=None):
    states = states or ["consumed"] * 8
    heads = {f"crash-group-{index:02d}": f"head-{index}" for index in range(8)}
    db_path = tmp_path / "trajectory-snapshot.sqlite"
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE trajectories (id TEXT,group_id TEXT,member INTEGER,"
                   "group_size INTEGER,state TEXT,head TEXT)")
        db.execute("CREATE TABLE events (sequence INTEGER,event TEXT)")
        for index, state in enumerate(states):
            db.execute("INSERT INTO trajectories VALUES (?,?,?,?,?,?)",
                       (f"crash-group-{index:02d}", "crash-group", index, 8,
                        state, heads[f"crash-group-{index:02d}"]))
        event = {"kind": "group_acknowledged", "job_id": "crash-group",
                 "details": {"state": "consumed", "heads": ack_heads or heads}}
        db.execute("INSERT INTO events VALUES (?,?)", (1, json.dumps(event)))
    worker = {"store_counts": {"consumed": 8, "quarantined": 0}}
    return db_path, worker, heads


def test_recovered_consumption_requires_snapshot_rows_and_acknowledgment(tmp_path):
    snapshot, worker, _ = _consumption_snapshot(tmp_path)

    result = audit_recovered_consumption(snapshot, worker)

    assert result["state_counts"] == {"consumed": 8}
    assert result["acknowledgment_events"] == 1


def test_recovered_consumption_rejects_stale_pre_ack_snapshot(tmp_path):
    snapshot, worker, _ = _consumption_snapshot(tmp_path, states=["verified"] * 8)

    with pytest.raises(RuntimeError, match="snapshot terminal states disagree"):
        audit_recovered_consumption(snapshot, worker)


def test_recovered_consumption_rejects_acknowledgment_for_other_heads(tmp_path):
    other_heads = {f"crash-group-{index:02d}": f"wrong-{index}" for index in range(8)}
    snapshot, worker, _ = _consumption_snapshot(tmp_path, ack_heads=other_heads)

    with pytest.raises(RuntimeError, match="does not cover the exact consumed snapshot"):
        audit_recovered_consumption(snapshot, worker)


def _promotion_fixture(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "policy-1.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"recovered-policy")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with sqlite3.connect(tmp_path / "promotion.sqlite") as db:
        db.execute("CREATE TABLE active (singleton INTEGER,policy_id TEXT,"
                   "checkpoint_sha256 TEXT)")
        db.execute("CREATE TABLE transactions (candidate_id TEXT,candidate_sha256 TEXT,"
                   "state TEXT,tx_id TEXT)")
        db.execute("INSERT INTO active VALUES (1,'policy-1',?)", (digest,))
        db.execute("INSERT INTO transactions VALUES ('policy-1',?,'committed','tx-1')",
                   (digest,))
    (tmp_path / "active.json").write_text(json.dumps({
        "policy_id": "policy-1", "checkpoint_sha256": digest,
    }))
    journal = {"active_policy_id": "policy-1", "active_checkpoint_sha256": digest,
               "pending_tx_id": None}
    return journal, digest


def test_durable_promotion_requires_database_json_journal_and_checkpoint_agreement(tmp_path):
    journal, digest = _promotion_fixture(tmp_path)

    result = audit_durable_promotion(tmp_path, journal)

    assert result == {"policy_id": "policy-1", "checkpoint_sha256": digest,
                      "committed_transactions": 1}


def test_durable_promotion_rejects_stale_database_active_pointer(tmp_path):
    journal, _ = _promotion_fixture(tmp_path)
    with sqlite3.connect(tmp_path / "promotion.sqlite") as db:
        db.execute("UPDATE active SET policy_id='policy-0',checkpoint_sha256='stale'")

    with pytest.raises(RuntimeError, match="durable active pointer differs"):
        audit_durable_promotion(tmp_path, journal)


def test_corrected_crash_runner_manifest_is_valid_for_seed_only_pilot(tmp_path):
    manifest = tmp_path / "crash-correction.json"
    manifest.write_text("{}")

    require_crash_correction_manifest("corrected", "frozen", manifest)


def test_corrected_crash_runner_requires_manifest_even_without_crash_bundle():
    with pytest.raises(RuntimeError, match="provide its correction manifest"):
        require_crash_correction_manifest("corrected", "frozen", None)


def test_frozen_crash_runner_rejects_unneeded_correction_manifest(tmp_path):
    manifest = tmp_path / "crash-correction.json"
    manifest.write_text("{}")

    with pytest.raises(RuntimeError, match="frozen runner is active"):
        require_crash_correction_manifest("frozen", "frozen", manifest)


def test_hash_map_manifest_verifies_and_rejects_unlisted_files(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"retained")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "SHA256.json").write_text(json.dumps({"payload.bin": digest}))

    verify_manifest(tmp_path)

    (tmp_path / "extra.bin").write_bytes(b"unlisted")
    with pytest.raises(RuntimeError, match="non-transient retained files"):
        verify_manifest(tmp_path)


def test_structured_inventory_verifies_rows_and_rejects_duplicate_paths(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"retained")
    row = {
        "path": "payload.bin",
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "bytes": payload.stat().st_size,
    }
    (tmp_path / "SHA256.json").write_text(json.dumps({
        "schema": "vare.artifact-inventory.v1", "files": [row],
    }))

    verify_manifest(tmp_path)

    (tmp_path / "SHA256.json").write_text(json.dumps({
        "schema": "vare.artifact-inventory.v1", "files": [row, row],
    }))
    with pytest.raises(RuntimeError, match="duplicate artifact"):
        verify_manifest(tmp_path)


def test_structured_inventory_rejects_path_traversal(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"outside")
    row = {
        "path": "../outside.bin",
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        "bytes": outside.stat().st_size,
    }
    (tmp_path / "SHA256.json").write_text(json.dumps({
        "schema": "vare.artifact-inventory.v1", "files": [row],
    }))

    with pytest.raises(RuntimeError, match="unsafe artifact inventory path"):
        verify_manifest(tmp_path)


def test_structured_inventory_rejects_noncanonical_path_alias(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"retained")
    row = {
        "path": "./payload.bin",
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "bytes": payload.stat().st_size,
    }
    (tmp_path / "SHA256.json").write_text(json.dumps({
        "schema": "vare.artifact-inventory.v1", "files": [row],
    }))

    with pytest.raises(RuntimeError, match="unsafe artifact inventory path"):
        verify_manifest(tmp_path)


def test_structured_inventory_allows_only_missing_disposable_sqlite_sidecars(tmp_path):
    row = {
        "path": "snapshot.sqlite-shm",
        "sha256": hashlib.sha256(b"shared-memory-index").hexdigest(),
        "bytes": len(b"shared-memory-index"),
    }
    empty_wal = {
        "path": "snapshot.sqlite-wal",
        "sha256": hashlib.sha256(b"").hexdigest(),
        "bytes": 0,
    }
    (tmp_path / "SHA256.json").write_text(json.dumps({
        "schema": "vare.artifact-inventory.v1", "files": [row, empty_wal],
    }))

    verify_manifest(tmp_path)


@pytest.mark.parametrize("comparison", [max_error, max_tensor_error])
def test_tensor_comparison_rejects_later_nan_after_finite_zero(comparison):
    torch = pytest.importorskip("torch")
    actual = {
        "finite": torch.tensor([0.0]),
        "non_finite": torch.tensor([float("nan")]),
    }
    expected = {
        "finite": torch.tensor([0.0]),
        "non_finite": torch.tensor([0.0]),
    }

    with pytest.raises(RuntimeError, match="non-finite"):
        comparison(actual, expected)
