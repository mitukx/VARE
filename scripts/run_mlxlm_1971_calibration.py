#!/usr/bin/env python3
"""Calibrate the pinned MLX-LM #1971 task in the isolated CPU runner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from vare.mlxlm_1971_repo_task import (  # noqa: E402
    BASELINE_COMMIT,
    FIXED_COMMIT,
    FIXED_SOURCE_SHA256,
    TASK_ID,
    TARGET_PATH,
    MlxLm1971TaskError,
    export_candidate,
    grade_workspace,
    sha256_file,
    verify_baseline,
)
from scripts.run_mlxlm_1912_calibration import run_red_team  # noqa: E402

TASK = "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale"
PROTOCOL = ROOT / TASK / "protocol-v1.json"
LOCKED = [
    "src/vare/mlxlm_1971_repo_task.py",
    "src/vare/mlxlm_1912_repo_task.py",
    "src/vare/isolated_candidate.py",
    "src/vare/container_isolation.py",
    "scripts/run_mlxlm_1971_calibration.py",
    "scripts/run_mlxlm_1912_calibration.py",
    f"{TASK}/grader_adapter.py",
    f"{TASK}/Dockerfile",
    f"{TASK}/task.md",
    f"{TASK}/expected-output-v1.json",
    "tests/test_mlxlm_1971_repo_task.py",
]


def verify_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    actual = {name: sha256_file(ROOT / name) for name in LOCKED}
    if protocol.get("implementation_sha256") != actual:
        raise RuntimeError("protocol implementation hash lock is absent or mismatched")
    locked = set(LOCKED) | {f"{TASK}/protocol-v1.json"}
    flags = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-v", "--", *sorted(locked)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.splitlines()
    indexed: set[str] = set()
    for line in flags:
        if len(line) < 3 or line[1] != " ":
            raise RuntimeError("cannot parse locked input Git index flags")
        tag, relative = line[0], line[2:]
        indexed.add(relative)
        if tag == "S" or tag.islower():
            raise RuntimeError(f"protocol input has a hidden Git index flag: {relative}")
    if indexed != locked:
        raise RuntimeError("protocol input lock contains missing or untracked files")
    if (
        protocol.get("task_id") != TASK_ID
        or protocol.get("baseline_commit") != BASELINE_COMMIT
        or protocol.get("fixed_commit") != FIXED_COMMIT
        or protocol.get("expected_output_sha256")
        != sha256_file(ROOT / f"{TASK}/expected-output-v1.json")
    ):
        raise RuntimeError("protocol task identity or golden output disagrees")
    return protocol


def _git_bytes(repo: Path, revision: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{TARGET_PATH}"],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def make_known_fix_patch(repo: Path, destination: Path) -> str:
    base = _git_bytes(repo, BASELINE_COMMIT).decode("utf-8")
    fixed = _git_bytes(repo, FIXED_COMMIT).decode("utf-8")
    if hashlib.sha256(fixed.encode()).hexdigest() != FIXED_SOURCE_SHA256:
        raise RuntimeError("fixed source SHA-256 disagrees with the pinned revision")
    patch = f"diff --git a/{TARGET_PATH} b/{TARGET_PATH}\n" + "".join(
        difflib.unified_diff(
            base.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            fromfile=f"a/{TARGET_PATH}",
            tofile=f"b/{TARGET_PATH}",
        )
    )
    destination.write_text(patch, encoding="utf-8")
    return patch


def _worktree(repo: Path, destination: Path, revision: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(destination), revision],
        check=True,
        capture_output=True,
        timeout=120,
    )


def _remove_worktree(repo: Path, path: Path) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "remove", "--force", str(path)],
        check=True,
        capture_output=True,
        timeout=30,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = verify_protocol()
    image = protocol["runtime"]["image"]
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    if status.strip():
        raise RuntimeError("implementation worktree must be clean before calibration")
    repo = args.source_repo.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    implementation_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vare-mlxlm-1971-cal-") as folder:
        temp = Path(folder)
        baseline_checkout = temp / "baseline-source"
        _worktree(repo, baseline_checkout, BASELINE_COMMIT)
        verify_baseline(baseline_checkout)
        baseline_workspace = temp / "baseline-candidate"
        baseline_export = export_candidate(baseline_checkout, baseline_workspace)
        baseline = grade_workspace(baseline_workspace, image=image)
        patch_path = temp / "known-fix.patch"
        patch_text = make_known_fix_patch(repo, patch_path)
        fixed_workspace = temp / "fixed-candidate"
        fixed_export = export_candidate(baseline_checkout, fixed_workspace, patch_path)
        fixed = grade_workspace(fixed_workspace, image=image)
        red_team = run_red_team(image, temp)
        source_swap = False
        fixed_checkout = temp / "fixed-source"
        _worktree(repo, fixed_checkout, FIXED_COMMIT)
        try:
            verify_baseline(fixed_checkout)
        except MlxLm1971TaskError:
            source_swap = True
        else:
            source_swap = False
        _remove_worktree(repo, fixed_checkout)
        _remove_worktree(repo, baseline_checkout)
        if fixed_export["candidate_source_sha256"] != FIXED_SOURCE_SHA256:
            raise RuntimeError("known patch does not reproduce pinned fixed source")
        result = {
            "protocol_id": protocol["protocol_id"],
            "task_id": TASK_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_commit": implementation_commit,
            "implementation_worktree_clean_at_start": True,
            "protocol_sha256": sha256_file(PROTOCOL),
            "implementation_sha256": protocol["implementation_sha256"],
            "source_repository": "ml-explore/mlx-lm",
            "baseline_commit": BASELINE_COMMIT,
            "fixed_commit": FIXED_COMMIT,
            "baseline_export": baseline_export,
            "fixed_export": fixed_export,
            "known_patch_sha256": hashlib.sha256(patch_text.encode()).hexdigest(),
            "candidate_image": image,
            "expected_output_sha256": sha256_file(ROOT / f"{TASK}/expected-output-v1.json"),
            "baseline": baseline,
            "known_fix": fixed,
            "red_team_suite": red_team,
            "red_team_sha256": hashlib.sha256(
                json.dumps(red_team, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_swap_rejected": source_swap,
            "wall_seconds": time.monotonic() - started,
            "calibration_passed": baseline["passed"] is False
            and fixed["passed"] is True
            and source_swap
            and red_team["passed"] is True,
            "limitations": [
                "One synthetic small-model correctness task; not a Coding-Agent rollout.",
                "The trusted grader compares candidate outputs to fixed-source golden outputs; hidden pytest files are never given to the candidate.",
                "The pinned container is CPU-only MLX. This does not measure Metal behavior or pretrained-model behavior.",
                "This task-specific adapter does not certify arbitrary Python repositories.",
            ],
        }
    result_path = output / "calibration.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        f"{sha256_file(result_path)}  calibration.json\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["calibration_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
