#!/usr/bin/env python3
"""Read-only, Git-object-anchored preflight for the historical v9 MLX-LM grader.

This verifies a v9 checkout before it is used by an operator. It does not
modify, re-freeze, or replace the historical v9 calibration protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from typing import Any

from scripts.audit_mlxlm_1971_historical_implementation_hashes import (
    GIT_ENV,
    SOURCE_COMMITS,
    TASK_ROOT,
    _git_bytes,
    audit_frozen_pair,
)


def attest_current_v9(root: Path) -> dict[str, Any]:
    """Fail closed if checkout bytes differ from the immutable v9 Git objects."""
    root = root.resolve(strict=True)
    frozen_commit = SOURCE_COMMITS[9]
    protocol_path = f"{TASK_ROOT}/protocol-v9.json"
    lock_path = f"{TASK_ROOT}/protocol-v9.lock.json"
    audit = audit_frozen_pair(root, protocol_path, lock_path, frozen_commit)
    paths = sorted({protocol_path, lock_path, *audit["implementation_sha256"]})
    for relative in paths:
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"v9 input is not a regular file: {relative}")
        if not stat.S_ISREG(candidate.lstat().st_mode):
            raise ValueError(f"v9 input is not a regular file: {relative}")
        expected = _git_bytes(root, frozen_commit, relative)
        if candidate.read_bytes() != expected:
            raise ValueError(f"v9 checkout bytes differ from immutable Git: {relative}")
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-v", "--", *paths],
            check=True,
            env=GIT_ENV,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("cannot verify v9 index flags") from exc
    found: set[str] = set()
    for line in result.stdout.splitlines():
        if len(line) < 3 or line[1] != " ":
            raise ValueError("malformed v9 Git index flag")
        tag, name = line[0], line[2:]
        if name not in paths or name in found:
            raise ValueError("unexpected or duplicate v9 index entry")
        if tag == "S" or tag.islower():
            raise ValueError(f"hidden v9 Git index flag: {name}")
        found.add(name)
    if found != set(paths):
        raise ValueError("v9 Git index inventory incomplete")
    return {
        "schema_version": 1,
        "protocol_id": audit["protocol_id"],
        "frozen_source_commit": frozen_commit,
        "protocol_sha256": audit["protocol_sha256"],
        "implementation_files_verified": len(audit["implementation_sha256"]),
        "current_source_matches_pinned_git": True,
        "replacement_objects_disabled": True,
    }


def export_candidate_attested_v9(
    source_root: Path, workspace: Path, patch_path: Path | None = None
) -> dict[str, Any]:
    """Call the historical public exporter only after immutable source attestation."""
    attest_current_v9(Path(__file__).resolve().parents[1])
    from vare.mlxlm_1971_repo_task import export_candidate

    return export_candidate(source_root, workspace, patch_path)


def grade_workspace_attested_v9(
    workspace: Path,
) -> dict[str, Any]:
    """Call the historical public grader only after immutable source attestation."""
    attest_current_v9(Path(__file__).resolve().parents[1])
    from vare.mlxlm_1971_repo_task import grade_workspace

    return grade_workspace(workspace)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(attest_current_v9(args.root), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
