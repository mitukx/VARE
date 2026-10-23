#!/usr/bin/env python3
"""Prospective v12 operator with an isolated snapshot of untrusted source Git.

This version delegates to the pinned v11 operator only after copying the
requested baseline commit into fresh bare Git metadata. It is a source
provenance check, not model or task-success evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_mlxlm_1971_v11_operator as v11  # noqa: E402

TASK = ROOT / "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale"
PROTOCOL = TASK / "protocol-v12.json"
LOCK = TASK / "protocol-v12.lock.json"
PROTOCOL_SHA256 = "9bd5ef255748c41344f641b3c3383324337a07fad06fa189d4ce4f376bf19332"
LOCK_SHA256 = "698087aa0c66a3f5fd4f0ef7753d76cf6f244ebc9ee193c30af9919a2f9f31b5"
PROTOCOL_ID = "vare.mlsys-repository-task.mlxlm-1971.v12.operator"
BASE_OPERATOR_SHA256 = (
    "604986569eee4ebe33d24c5009292786501d5956fb5024050a814c14659f98de"
)
BASELINE_COMMIT = "5f3c79d6bb1864a85d5343818192bd061b161e3a"
SOURCE_PATH = "scripts/run_mlxlm_1971_v12_operator.py"
BASE_OPERATOR_PATH = "scripts/run_mlxlm_1971_v11_operator.py"
BASE_PROTOCOL_PATH = (
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/protocol-v11.json"
)
BASE_LOCK_PATH = (
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/protocol-v11.lock.json"
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git(repository: Path, *args: str, timeout: int = 120) -> str:
    result = subprocess.run(
        v11._git_command(repository, *args),
        env=v11.GIT_ENV,
        capture_output=True,
        check=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def verify_v12_protocol() -> dict[str, Any]:
    """Verify v12 identity and the exact v11 operator it delegates to."""
    protocol_bytes = v11._read_regular_file(
        PROTOCOL, max_bytes=256_000, label="v12 protocol"
    )
    lock_bytes = v11._read_regular_file(LOCK, max_bytes=256_000, label="v12 lock")
    if _sha256(protocol_bytes) != PROTOCOL_SHA256:
        raise ValueError("v12 protocol differs from its pinned bytes")
    if _sha256(lock_bytes) != LOCK_SHA256:
        raise ValueError("v12 lock differs from its pinned bytes")
    protocol = json.loads(protocol_bytes)
    lock = json.loads(lock_bytes)
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("supersedes_protocol_id")
        != "vare.mlsys-repository-task.mlxlm-1971.v11.operator"
        or protocol.get("status") != "PROSPECTIVE_OPERATOR_INTERFACE_NOT_EXECUTED"
        or protocol.get("baseline_commit") != BASELINE_COMMIT
        or protocol.get("source_snapshot_method")
        != "fresh-depth-one-bare-fetch-no-checkout"
        or protocol.get("source_repository_checked_out_directly") is not False
        or protocol.get("candidate_controlled_filters_executed") is not False
        or protocol.get("candidate_controlled_hooks_executed") is not False
        or protocol.get("allowed_actions") != ["export", "grade"]
        or protocol.get("hidden_evaluation_output_to_candidate") is not False
        or protocol.get("base_operator_sha256") != BASE_OPERATOR_SHA256
        or protocol.get("base_protocol_sha256") != v11.PROTOCOL_SHA256
        or lock.get("protocol_id") != PROTOCOL_ID
        or lock.get("protocol_sha256") != PROTOCOL_SHA256
        or lock.get("implementation_sha256", {}).get(BASE_OPERATOR_PATH)
        != BASE_OPERATOR_SHA256
    ):
        raise ValueError("v12 operator protocol identity mismatch")

    dependencies = {
        BASE_OPERATOR_PATH: BASE_OPERATOR_SHA256,
        BASE_PROTOCOL_PATH: v11.PROTOCOL_SHA256,
        BASE_LOCK_PATH: v11.LOCK_SHA256,
    }
    if lock.get("implementation_sha256") != dependencies:
        raise ValueError("v12 pinned implementation inventory differs")
    for relative, expected in dependencies.items():
        actual = _sha256(
            v11._read_regular_file(ROOT / relative, max_bytes=2_000_000, label=relative)
        )
        if actual != expected:
            raise ValueError(f"pinned v11 dependency changed: {relative}")
    v11.verify_v11_protocol()
    return protocol


def _snapshot_source_repository(
    source_repository: Path, destination: Path, commit: str
) -> Path:
    """Fetch one commit without checking out source-controlled attributes/filters."""
    source = v11._require_real_directory(source_repository)
    if destination.exists() or destination.is_symlink():
        raise ValueError("source snapshot destination already exists")
    destination.mkdir(mode=0o700)
    subprocess.run(
        ["git", "init", "--bare", "--quiet", str(destination)],
        env=v11.GIT_ENV,
        capture_output=True,
        check=True,
        timeout=30,
    )
    _git(
        destination,
        "fetch",
        "--depth=1",
        "--no-tags",
        "--no-recurse-submodules",
        str(source),
        commit,
    )
    fetched = _git(destination, "rev-parse", "FETCH_HEAD")
    if fetched != commit:
        raise ValueError("source repository did not provide the pinned baseline commit")
    _git(destination, "cat-file", "-e", f"{commit}^{{commit}}")
    _git(destination, "update-ref", "refs/heads/vare-pinned-baseline", commit)
    _git(destination, "symbolic-ref", "HEAD", "refs/heads/vare-pinned-baseline")
    refs = _git(destination, "for-each-ref", "--format=%(refname)").splitlines()
    if refs != ["refs/heads/vare-pinned-baseline"]:
        raise ValueError("isolated source repository contains unexpected refs")
    if (destination / "info/attributes").exists():
        raise ValueError("isolated source repository unexpectedly has local attributes")
    return destination


def _materialize_source_worktree(
    snapshot_repository: Path, destination: Path, commit: str
) -> Path:
    """Populate a worktree from isolated metadata without running source hooks."""
    if destination.exists() or destination.is_symlink():
        raise ValueError("source worktree destination already exists")
    subprocess.run(
        v11._git_command(
            snapshot_repository,
            "worktree",
            "add",
            "--detach",
            "--no-checkout",
            str(destination),
            commit,
        ),
        env=v11.GIT_ENV,
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )
    subprocess.run(
        v11._git_command(destination, "read-tree", "-mu", commit),
        env=v11.GIT_ENV,
        capture_output=True,
        check=True,
        text=True,
        timeout=120,
    )
    if _git(destination, "rev-parse", "HEAD") != commit:
        raise ValueError("isolated source worktree is not at the pinned commit")
    if _git(destination, "rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError("isolated source snapshot did not produce a worktree")
    return destination


def run_v12_operation(
    action: str,
    workspace: Path,
    *,
    source_repository: Path | None = None,
    patch_path: Path | None = None,
) -> dict[str, Any]:
    """Run a v11 action, snapshotting untrusted Git before export."""
    if action not in {"grade", "export"}:
        raise ValueError("unsupported v12 operation")
    if action == "grade" and (source_repository is not None or patch_path is not None):
        raise ValueError("grade cannot receive export-only inputs")
    if action == "export" and source_repository is None:
        raise ValueError("export requires source repository")
    verify_v12_protocol()
    if action == "grade":
        report = v11.run_v11_operation("grade", workspace)
        return {
            **report,
            "operator_protocol_id": PROTOCOL_ID,
            "operator_protocol_sha256": PROTOCOL_SHA256,
            "source_snapshot_commit": None,
        }

    assert source_repository is not None
    trusted_v9 = v11._trusted_v9_protocol()
    baseline_commit = trusted_v9.get("baseline_commit")
    if baseline_commit != BASELINE_COMMIT:
        raise ValueError("trusted v9 baseline commit differs from v12 protocol")
    with tempfile.TemporaryDirectory(prefix=".vare-v12-source-", dir=ROOT) as temp:
        snapshot = _snapshot_source_repository(
            source_repository, Path(temp) / "source-objects.git", baseline_commit
        )
        source_worktree = _materialize_source_worktree(
            snapshot, Path(temp) / "source-worktree", baseline_commit
        )
        report = v11.run_v11_operation(
            "export", workspace, source_repo=source_worktree, patch_path=patch_path
        )
        return {
            **report,
            "operator_protocol_id": PROTOCOL_ID,
            "operator_protocol_sha256": PROTOCOL_SHA256,
            "source_snapshot_commit": baseline_commit,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("export", "grade"), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args(argv)
    report = run_v12_operation(
        args.action,
        args.workspace,
        source_repository=args.source_repo,
        patch_path=args.patch,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if args.action == "export" or report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
