#!/usr/bin/env python3
"""Execute frozen MLX-LM #1971 v9 calibration from an immutable Git worktree.

This is a prospective operator wrapper, not a new calibration result or a
replacement for the frozen v9 protocol. Direct historical v9 entrypoints
remain unsupported for provenance-sensitive new evaluations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
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


def _operator_path(path: Path) -> Path:
    path = path.absolute()
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError("operator paths must not contain symlinks")
    return path


def _reserve_output(output: Path) -> tuple[Path, tuple[int, int]]:
    output = _operator_path(output)
    parent = output.parent.stat()
    if parent.st_uid != os.getuid() or parent.st_mode & 0o022:
        raise ValueError("calibration output requires an operator-owned private parent")
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing existing calibration output: {output}") from exc
    info = output.lstat()
    return output, (info.st_dev, info.st_ino)


def _verify_output_identity(output: Path, identity: tuple[int, int]) -> None:
    _operator_path(output)
    info = output.lstat()
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        raise ValueError("reserved calibration output identity changed")


def run_frozen_calibration(source_repo: Path, output: Path) -> int:
    """Run only source whose committed bytes match the independent v9 anchor."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing calibration output: {output}")
    source_repo = _operator_path(source_repo)
    if not source_repo.is_dir():
        raise ValueError("source repo must be a directory")
    if SOURCE_COMMITS[9] != PINNED_V9_COMMIT:
        raise ValueError("historical v9 source anchor changed")
    output, output_identity = _reserve_output(output)
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
            _verify_output_identity(output, output_identity)
            command = [
                sys.executable,
                "-I",
                str(worktree / "scripts/run_mlxlm_1971_calibration_v9.py"),
                "--source-repo", str(source_repo),
                "--output", str(output / "records"),
            ]
            result = subprocess.run(
                command, cwd=worktree, env=env,
                check=False, timeout=1800,
            )
            _verify_output_identity(output, output_identity)
            return result.returncode
        finally:
            subprocess.run(
                ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(worktree)],
                check=True, capture_output=True, env=env, timeout=120,
            )



_PINNED_TASK_CHILD = """
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))
from vare.mlxlm_1971_repo_task import export_candidate, grade_workspace

request = json.loads(sys.stdin.read())
if request["action"] == "grade":
    report = grade_workspace(Path(request["workspace"]))
elif request["action"] == "export":
    patch = request.get("patch_path")
    report = export_candidate(
        Path(request["source_repo"]),
        Path(request["workspace"]),
        Path(patch) if patch is not None else None,
    )
else:
    raise ValueError("unsupported pinned operation")
print(json.dumps(report, sort_keys=True))
"""


def run_pinned_v9_task(
    action: str,
    workspace: Path,
    *,
    source_repo: Path | None = None,
    patch_path: Path | None = None,
) -> dict[str, object]:
    """Dispatch grade/export from immutable v9 source in a separate process.

    This does not turn the legacy unguarded API into a new certified protocol.
    The trusted caller must keep grading output separate from the candidate.
    """
    if action not in {"grade", "export"}:
        raise ValueError("unsupported pinned operation")
    if action == "grade" and (source_repo is not None or patch_path is not None):
        raise ValueError("grade cannot receive exporter inputs")
    if action == "export" and source_repo is None:
        raise ValueError("export requires source repo")
    if SOURCE_COMMITS[9] != PINNED_V9_COMMIT:
        raise ValueError("historical v9 source anchor changed")

    workspace = _operator_path(workspace)
    request = {
        "action": action,
        "workspace": str(workspace.absolute()),
    }
    if source_repo is not None:
        source_repo = _operator_path(source_repo)
        if not source_repo.is_dir():
            raise ValueError("source repo must be a directory")
        request["source_repo"] = str(source_repo)
    if patch_path is not None:
        patch_path = _operator_path(patch_path)

    env = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}
    with tempfile.TemporaryDirectory(prefix="vare-attested-mlxlm1971-task-") as temp:
        if patch_path is not None:
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            with os.fdopen(os.open(patch_path, flags), "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("patch must be a regular file")
                patch_bytes = stream.read(256_001)
            if len(patch_bytes) > 256_000:
                raise ValueError("patch exceeds historical byte limit")
            snapshot = Path(temp) / "candidate.patch"
            snapshot.write_bytes(patch_bytes)
            snapshot.chmod(0o400)
            request["patch_path"] = str(snapshot)
        worktree = Path(temp) / "frozen-v9"
        subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "add", "--detach",
             str(worktree), PINNED_V9_COMMIT],
            check=True, capture_output=True, env=env, timeout=120,
        )
        try:
            attestation = attest_current_v9(worktree)
            if (
                attestation.get("frozen_source_commit") != PINNED_V9_COMMIT
                or attestation.get("current_source_matches_pinned_git") is not True
            ):
                raise ValueError("historical v9 checkout failed immutable attestation")
            result = subprocess.run(
                [sys.executable, "-I", "-c", _PINNED_TASK_CHILD],
                cwd=worktree,
                env=env,
                input=json.dumps(request),
                text=True,
                capture_output=True,
                check=False,
                timeout=1800,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"pinned v9 {action} failed with exit {result.returncode}"
                )
            try:
                report = json.loads(result.stdout)
            except (ValueError, TypeError) as exc:
                raise RuntimeError("pinned v9 task produced invalid JSON") from exc
            if not isinstance(report, dict):
                raise RuntimeError("pinned v9 task returned non-object report")
            return report
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
