from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_regression_evidence import audit_manifest
from scripts.calibrate_regressions import protocol_tasks


def _write_manifest(root: Path, entries: dict[str, str]) -> None:
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": entries}), encoding="utf-8"
    )


def test_regression_protocol_allowlist_excludes_invalidated_versions():
    protocol, entries = protocol_tasks()

    assert protocol["protocol_id"] == "vare-current-regression-calibration-v1"
    assert {entry["task_id"] for entry in entries} == {
        "vare-promotion-gate-nonfinite-metrics-v1",
        "vare-replay-group-freshness-atomicity-v1",
        "vare-environment-command-output-budget-v4",
    }
    assert any("v1 is invalidated" in value for value in protocol["exclusions"])
    assert any("v3 is invalidated" in value for value in protocol["exclusions"])


def test_manifest_audit_checks_hash_and_exact_file_inventory(tmp_path: Path):
    payload = b"frozen result\n"
    (tmp_path / "result.json").write_bytes(payload)
    _write_manifest(tmp_path, {"result.json": hashlib.sha256(payload).hexdigest()})

    assert audit_manifest(tmp_path) == 1

    (tmp_path / "result.json").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_manifest(tmp_path)


def test_manifest_audit_rejects_unlisted_files_and_path_traversal(tmp_path: Path):
    (tmp_path / "listed.json").write_text("{}", encoding="utf-8")
    _write_manifest(tmp_path, {"listed.json": hashlib.sha256(b"{}").hexdigest()})
    (tmp_path / "unlisted.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory mismatch"):
        audit_manifest(tmp_path)

    _write_manifest(tmp_path, {"../outside": "0" * 64})
    with pytest.raises(ValueError, match="unsafe path"):
        audit_manifest(tmp_path)
