from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.audit_mlxlm_1971_historical_implementation_hashes import (
    audit_all,
    audit_frozen_pair,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_all_retained_protocol_implementation_hashes_match_recoverable_git() -> None:
    report = audit_all(Path(__file__).resolve().parents[1])
    assert [entry["protocol_id"].rsplit("v", 1)[-1] for entry in report["versions"]] == [
        "4", "5", "6", "7", "8", "9"
    ]
    assert report["implementation_files_verified"] == 85


def test_coordinated_worktree_tampering_cannot_reseal_frozen_protocol(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    implementation = root / "impl.py"
    implementation.write_bytes(b"trusted implementation\n")
    implementation_hash = hashlib.sha256(implementation.read_bytes()).hexdigest()
    protocol_path = "protocol.json"
    lock_path = "protocol.lock.json"
    protocol = {
        "protocol_id": "test-v1",
        "implementation_sha256": {"impl.py": implementation_hash},
    }
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode() + b"\n"
    lock = {
        "protocol_id": "test-v1",
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "implementation_sha256": protocol["implementation_sha256"],
    }
    (root / protocol_path).write_bytes(protocol_bytes)
    (root / lock_path).write_text(json.dumps(lock, sort_keys=True) + "\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "freeze source")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    audit_frozen_pair(root, protocol_path, lock_path, source_commit)

    # Tamper with implementation, protocol and lock coherently in the worktree.
    implementation.write_bytes(b"attacker implementation\n")
    protocol["implementation_sha256"]["impl.py"] = hashlib.sha256(
        implementation.read_bytes()
    ).hexdigest()
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode() + b"\n"
    (root / protocol_path).write_bytes(protocol_bytes)
    lock["protocol_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    lock["implementation_sha256"] = protocol["implementation_sha256"]
    (root / lock_path).write_text(json.dumps(lock, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="frozen protocol differs from Git source"):
        audit_frozen_pair(root, protocol_path, lock_path, source_commit)


def test_git_replace_cannot_redirect_historical_blob_reads(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    protocol_path = "protocol.json"
    lock_path = "protocol.lock.json"

    def write_snapshot(source: bytes) -> None:
        (root / "impl.py").write_bytes(source)
        protocol = {
            "protocol_id": "test-v1",
            "implementation_sha256": {"impl.py": hashlib.sha256(source).hexdigest()},
        }
        protocol_bytes = json.dumps(protocol, sort_keys=True).encode() + b"\n"
        lock = {
            "protocol_id": "test-v1",
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "implementation_sha256": protocol["implementation_sha256"],
        }
        (root / protocol_path).write_bytes(protocol_bytes)
        (root / lock_path).write_text(json.dumps(lock, sort_keys=True) + "\n")

    write_snapshot(b"trusted implementation\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "trusted source")
    trusted = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

    write_snapshot(b"replacement implementation\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "replacement source")
    replacement = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    _git(root, "replace", trusted, replacement)

    with pytest.raises(ValueError, match="frozen protocol differs from Git source"):
        audit_frozen_pair(root, protocol_path, lock_path, trusted)
