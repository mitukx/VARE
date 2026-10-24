from __future__ import annotations

import hashlib
import json

import pytest

from scripts.audit_verifier_consistent_async_grpo import (
    max_error,
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
