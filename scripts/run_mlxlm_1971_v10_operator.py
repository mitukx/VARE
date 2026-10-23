#!/usr/bin/env python3
"""Prospective v10 operator: source-bound MLX-LM #1971 export/grading.

This is a new operator API, not a rewrite of frozen v9. Its trusted parent
process owns grader outputs; no model inference or qualification is implied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale"
PROTOCOL = TASK / "protocol-v10.json"
LOCK = TASK / "protocol-v10.lock.json"
PROTOCOL_SHA256 = "bc65db13f1a0d1d5dea571cc9ea632b25bb9d0d60b96361b5183656481976dc3"
LOCK_SHA256 = "09c69a6b87d1265a37772fb5d53bb77c873b02ad45496f343c6968c0442df1f0"
OPERATOR_COMMIT = "45fd0da592955205e6518c12e9e9b315797541e2"
V9_COMMIT = "3f12eefd71d1d2213b37e3a174b446cb5155e979"
V9_PROTOCOL_SHA256 = "aa8fec4c3a3fbba119b83009c7f3c164b762758a5c1593faf3a78f3464678fb2"
SOURCE_PATH = "scripts/run_attested_mlxlm_1971_v9_calibration.py"
PINNED_OPERATOR_SOURCES = (
    SOURCE_PATH,
    "scripts/attest_mlxlm_1971_v9_current_git.py",
    "scripts/audit_mlxlm_1971_historical_implementation_hashes.py",
)
V9_PROTOCOL_PATH = (
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/protocol-v9.json"
)
GIT_ENV = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}

_PINNED_CHILD = """
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from scripts.run_attested_mlxlm_1971_v9_calibration import run_pinned_v9_task

request = json.loads(sys.stdin.read())
if request["action"] == "grade":
    report = run_pinned_v9_task("grade", Path(request["workspace"]))
elif request["action"] == "export":
    patch = request.get("patch_path")
    report = run_pinned_v9_task(
        "export", Path(request["workspace"]),
        source_repo=Path(request["source_repo"]),
        patch_path=Path(patch) if patch is not None else None,
    )
else:
    raise ValueError("unsupported operator action")
print(json.dumps(report, sort_keys=True))
"""


def _git_bytes(revision: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{revision}:{relative}"],
        env=GIT_ENV, capture_output=True, check=True, timeout=30,
    )
    return result.stdout


def verify_v10_protocol() -> dict[str, Any]:
    """Independently verify the v10 manifest and the two historical trust roots."""
    for path in (PROTOCOL, LOCK):
        if path.is_symlink() or not path.is_file():
            raise ValueError("v10 protocol or lock is missing or symlinked")
    source = PROTOCOL.read_bytes()
    if hashlib.sha256(source).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("v10 protocol differs from independently pinned bytes")
    lock_bytes = LOCK.read_bytes()
    if hashlib.sha256(lock_bytes).hexdigest() != LOCK_SHA256:
        raise ValueError("v10 lock differs from independently pinned bytes")
    protocol = json.loads(source)
    lock = json.loads(lock_bytes)
    if (
        lock.get("protocol_id") != protocol.get("protocol_id")
        or lock.get("protocol_sha256") != PROTOCOL_SHA256
        or protocol.get("protocol_id")
        != "vare.mlsys-repository-task.mlxlm-1971.v10.operator"
        or protocol.get("status") != "PROSPECTIVE_OPERATOR_INTERFACE_NOT_EXECUTED"
        or protocol.get("operator_source_commit") != OPERATOR_COMMIT
        or protocol.get("operator_source_path") != SOURCE_PATH
        or protocol.get("reference_v9_source_commit") != V9_COMMIT
        or protocol.get("reference_v9_protocol_sha256") != V9_PROTOCOL_SHA256
        or protocol.get("allowed_actions") != ["export", "grade"]
        or protocol.get("hidden_evaluation_output_to_candidate") is not False
        or protocol.get("requires_disable_git_replacement_refs") is not True
        or protocol.get("requires_immutable_pinned_operator_worktree") is not True
        or protocol.get("requires_nested_v9_source_attestation") is not True
    ):
        raise ValueError("v10 operator protocol identity mismatch")
    historical = _git_bytes(V9_COMMIT, V9_PROTOCOL_PATH)
    if hashlib.sha256(historical).hexdigest() != V9_PROTOCOL_SHA256:
        raise ValueError("pinned v9 Git protocol digest mismatch")
    for relative in PINNED_OPERATOR_SOURCES:
        if not _git_bytes(OPERATOR_COMMIT, relative):
            raise ValueError(f"pinned v10 operator source missing from Git: {relative}")
    return protocol


def run_v10_operation(
    action: str,
    workspace: Path,
    *,
    source_repo: Path | None = None,
    patch_path: Path | None = None,
) -> dict[str, Any]:
    """Use the historical trusted grader only through pinned source commits."""
    if action not in {"grade", "export"}:
        raise ValueError("unsupported v10 operation")
    if action == "grade" and (source_repo is not None or patch_path is not None):
        raise ValueError("grade cannot receive export-only arguments")
    if action == "export" and source_repo is None:
        raise ValueError("export requires source repository")
    if workspace.is_symlink():
        raise ValueError("refusing symlinked candidate workspace")
    verify_v10_protocol()
    request: dict[str, str] = {
        "action": action, "workspace": str(workspace.absolute()),
    }
    if source_repo is not None:
        request["source_repo"] = str(source_repo.resolve(strict=True))
    if patch_path is not None:
        request["patch_path"] = str(patch_path.resolve(strict=True))

    with tempfile.TemporaryDirectory(prefix="vare-v10-operator-") as temporary:
        checkout = Path(temporary) / "trusted-operator"
        subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "add", "--detach",
             str(checkout), OPERATOR_COMMIT],
            env=GIT_ENV, check=True, capture_output=True, timeout=120,
        )
        try:
            for relative in PINNED_OPERATOR_SOURCES:
                source_file = checkout / relative
                if source_file.is_symlink() or not source_file.is_file():
                    raise ValueError(f"untrusted operator source path: {relative}")
                if source_file.read_bytes() != _git_bytes(OPERATOR_COMMIT, relative):
                    raise ValueError(
                        f"operator worktree differs from pinned Git bytes: {relative}"
                    )
            result = subprocess.run(
                [sys.executable, "-I", "-c", _PINNED_CHILD],
                cwd=checkout, env=GIT_ENV,
                input=json.dumps(request), text=True, capture_output=True,
                check=False, timeout=1800,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"pinned v10 operator exited nonzero: {result.returncode}"
                )
            try:
                report = json.loads(result.stdout)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("v10 operator did not return one JSON object") from exc
            if not isinstance(report, dict):
                raise RuntimeError("v10 operator returned non-object evidence")
            return report
        finally:
            subprocess.run(
                ["git", "-C", str(ROOT), "worktree", "remove", "--force",
                 str(checkout)],
                env=GIT_ENV, check=True, capture_output=True, timeout=120,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("grade", "export"), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_v10_operation(
        args.action, args.workspace,
        source_repo=args.source_repo, patch_path=args.patch,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
