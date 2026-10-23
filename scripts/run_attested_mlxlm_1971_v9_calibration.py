#!/usr/bin/env python3
"""Execute frozen MLX-LM #1971 v9 calibration from an immutable Git worktree.

This is a prospective operator wrapper, not a new calibration result or a
replacement for the frozen v9 protocol. Direct historical v9 entrypoints
remain unsupported for provenance-sensitive new evaluations.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.attest_mlxlm_1971_v9_current_git import (  # noqa: E402
    SOURCE_COMMITS,
    attest_current_v9,
)

PINNED_V9_COMMIT = "3f12eefd71d1d2213b37e3a174b446cb5155e979"


def run_frozen_calibration(source_repo: Path, output: Path) -> int:
    """Run only source whose committed bytes match the independent v9 anchor."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing calibration output: {output}")
    source_repo = source_repo.resolve(strict=True)
    if not source_repo.is_dir():
        raise ValueError("source repo must be a directory")
    if SOURCE_COMMITS[9] != PINNED_V9_COMMIT:
        raise ValueError("historical v9 source anchor changed")
    env = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}
    with tempfile.TemporaryDirectory(prefix="vare-attested-mlxlm1971-v9-") as temp:
        worktree = Path(temp) / "frozen-v9"
        subprocess.run(
            [
                "git", "-C", str(ROOT), "worktree", "add", "--detach",
                str(worktree), PINNED_V9_COMMIT,
            ],
            check=True, capture_output=True, env=env, timeout=120,
        )
        try:
            report = attest_current_v9(worktree)
            if (
                report.get("frozen_source_commit") != PINNED_V9_COMMIT
                or report.get("current_source_matches_pinned_git") is not True
            ):
                raise ValueError("historical v9 checkout failed immutable attestation")
            command = [
                sys.executable,
                "-I",
                str(worktree / "scripts/run_mlxlm_1971_calibration_v9.py"),
                "--source-repo", str(source_repo),
                "--output", str(output.absolute()),
            ]
            result = subprocess.run(
                command, cwd=worktree, env=env,
                check=False, timeout=1800,
            )
            return result.returncode
        finally:
            subprocess.run(
                ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(worktree)],
                check=True, capture_output=True, env=env, timeout=120,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run_frozen_calibration(args.source_repo, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
