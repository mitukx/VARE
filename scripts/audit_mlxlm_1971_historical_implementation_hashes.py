#!/usr/bin/env python3
"""Verify frozen MLX-LM #1971 protocol hashes against recoverable Git objects."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale"
GIT_ENV = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}
SOURCE_COMMITS = {
    4: "23f71a0c6923f6e677ed8abc0d4ed186ae047e89",
    5: "9f0993ac2c71a44bcfbabd2a1399fc12e5d6d777",
    6: "9f0993ac2c71a44bcfbabd2a1399fc12e5d6d777",
    7: "9f0993ac2c71a44bcfbabd2a1399fc12e5d6d777",
    8: "37ae70f4e93d6ebf9c5a35efff3149c0f5f86e4a",
    9: "3f12eefd71d1d2213b37e3a174b446cb5155e979",
}


def _git_bytes(root: Path, commit: str, relative: str) -> bytes:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe Git path: {relative}")
    return subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{relative}"],
        env=GIT_ENV,
        check=True,
        capture_output=True,
        timeout=20,
    ).stdout


def audit_frozen_pair(
    root: Path, protocol_path: str, lock_path: str, source_commit: str
) -> dict[str, Any]:
    """Check protocol, lock and every declared implementation against Git blobs."""
    subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{source_commit}^{{commit}}"],
        env=GIT_ENV,
        check=True,
        capture_output=True,
        timeout=10,
    )
    protocol_file = root / protocol_path
    lock_file = root / lock_path
    protocol_bytes = protocol_file.read_bytes()
    lock_bytes = lock_file.read_bytes()
    if protocol_bytes != _git_bytes(root, source_commit, protocol_path):
        raise ValueError(f"frozen protocol differs from Git source: {protocol_path}")
    if lock_bytes != _git_bytes(root, source_commit, lock_path):
        raise ValueError(f"frozen lock differs from Git source: {lock_path}")
    protocol = json.loads(protocol_bytes)
    lock = json.loads(lock_bytes)
    if lock.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError(f"protocol identity mismatch: {protocol_path}")
    if lock.get("protocol_sha256") != hashlib.sha256(protocol_bytes).hexdigest():
        raise ValueError(f"protocol bytes do not match lock: {protocol_path}")
    implementation_hashes = protocol.get("implementation_sha256")
    if (
        not isinstance(implementation_hashes, dict)
        or not implementation_hashes
        or implementation_hashes != lock.get("implementation_sha256")
    ):
        raise ValueError(f"implementation manifest mismatch: {protocol_path}")

    verified: dict[str, str] = {}
    for relative, expected in sorted(implementation_hashes.items()):
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
        ):
            raise ValueError(f"malformed implementation hash entry: {relative!r}")
        actual = hashlib.sha256(_git_bytes(root, source_commit, relative)).hexdigest()
        if actual != expected:
            raise ValueError(f"historical implementation hash mismatch: {relative}")
        verified[relative] = actual
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_path": protocol_path,
        "lock_path": lock_path,
        "source_commit": source_commit,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "implementation_sha256": verified,
    }


def audit_all(root: Path) -> dict[str, Any]:
    versions = []
    for version, source_commit in sorted(SOURCE_COMMITS.items()):
        versions.append(
            audit_frozen_pair(
                root,
                f"{TASK_ROOT}/protocol-v{version}.json",
                f"{TASK_ROOT}/protocol-v{version}.lock.json",
                source_commit,
            )
        )
    return {"schema_version": 1, "versions": versions, "implementation_files_verified": sum(len(v["implementation_sha256"]) for v in versions)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="optional append-only JSON evidence path")
    args = parser.parse_args()
    report = audit_all(ROOT)
    report["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["source_commit"] = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"refusing to overwrite audit evidence: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
