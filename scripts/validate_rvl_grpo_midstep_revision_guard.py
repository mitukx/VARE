#!/usr/bin/env python3
"""Require exact clean VARE/RVL revisions before replaying the frozen v1 check."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


VARE_COMMIT = "63ba181ae54a67738b6d40730ce10a9e8bdd7ba0"
RVL_COMMIT = "c7e646b043cb56e5ea3c2623bb8a61e065451f72"
V1_PROTOCOL_SHA256 = "2939204c394b07618b6ed69fa36a0caacf34d08ab215d4f991b3bf059a12f4f6"
V1_VALIDATOR_SHA256 = "971a4f19498be7fe440b78290b1d5768f8bb306eb9c64127537f70f76e316823"


class RevisionGuardError(ValueError):
    """Raised when a requested source checkout is not the locked clean revision."""


def _git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RevisionGuardError(result.stderr.strip() or "git query failed")
    return result.stdout.strip()


def require_clean_revision(source: Path, expected_commit: str) -> tuple[Path, str]:
    """Return checkout root and HEAD, requiring exact HEAD and no working changes."""
    source = source.resolve()
    root = Path(_git(source, "rev-parse", "--show-toplevel")).resolve()
    head = _git(source, "rev-parse", "HEAD")
    if head != expected_commit:
        raise RevisionGuardError(
            f"revision mismatch: expected {expected_commit}, got {head}"
        )
    status = _git(source, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RevisionGuardError("checkout has tracked or untracked changes")
    return root, head


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(vare_source: Path, rvl_source: Path) -> dict:
    vare_root, vare_head = require_clean_revision(vare_source, VARE_COMMIT)
    rvl_root, rvl_head = require_clean_revision(rvl_source, RVL_COMMIT)
    if vare_root != vare_source.resolve():
        raise RevisionGuardError("--vare-source must be the VARE checkout root")
    if rvl_source.resolve() != (rvl_root / "src/rvl_systems").resolve():
        raise RevisionGuardError(
            "--rvl-source must be the src/rvl_systems directory of its Git checkout"
        )

    validator = vare_root / "scripts/validate_rvl_grpo_midstep_fault.py"
    protocol = vare_root / "protocols/rvl_grpo_midstep_fault_v1.json"
    if _sha256(validator) != V1_VALIDATOR_SHA256:
        raise RevisionGuardError("frozen v1 validator hash mismatch")
    if _sha256(protocol) != V1_PROTOCOL_SHA256:
        raise RevisionGuardError("frozen v1 protocol hash mismatch")

    result = subprocess.run(
        [sys.executable, str(validator), "--rvl-source", str(rvl_source.resolve())],
        cwd=vare_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        if result.stdout:
            sys.stdout.write(result.stdout)
        raise RuntimeError(f"frozen v1 validator exited {result.returncode}")
    summary = json.loads(result.stdout)
    if summary.get("status") != "pass" or not all(summary.get("checks", {}).values()):
        raise RuntimeError("frozen v1 validator did not pass every check")
    summary["revision_guard"] = {
        "vare_git_commit": vare_head,
        "rvl_git_commit": rvl_head,
        "both_checkouts_clean": True,
        "v1_protocol_and_validator_hashes_verified": True,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vare-source", type=Path, required=True,
        help="clean VARE checkout at the frozen validator commit",
    )
    parser.add_argument(
        "--rvl-source", type=Path, required=True,
        help="src/rvl_systems from the exact clean pinned RVL checkout",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.vare_source, args.rvl_source), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RevisionGuardError, RuntimeError) as exc:
        print(f"revision guard failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
