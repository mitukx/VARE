#!/usr/bin/env python3
"""Fetch pinned, sparse MLX-LM task and trusted calibration inputs.

The candidate checkout is a shallow baseline snapshot containing only the
three editable files. Tests and known-correct source are staged separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence


REPOSITORY = "https://github.com/ml-explore/mlx-lm.git"
BASELINE = "77c33b14373ac70d7abd6f82af15962852adadbb"
KNOWN_FIX = "acb0240394dfc17ffd6db77a3ccecbdbde522ef9"
EDITABLE = (
    "mlx_lm/models/granitemoe.py",
    "mlx_lm/models/granitemoehybrid.py",
    "mlx_lm/models/lfm2_moe.py",
)
TEST_FILE = "tests/test_tuner_trainer.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    return result.stdout.strip()


def fetch_checkout(
    root: Path, revision: str, *, patterns: Sequence[str], cone: bool
) -> Path:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite prepared source: {root}")
    root.mkdir(parents=True)
    run_git(root, "init", "-q")
    run_git(root, "remote", "add", "upstream", REPOSITORY)
    run_git(
        root,
        "fetch",
        "--depth=1",
        "--filter=blob:none",
        "--no-tags",
        "upstream",
        revision,
    )
    run_git(root, "sparse-checkout", "init", "--cone" if cone else "--no-cone")
    run_git(
        root,
        "sparse-checkout",
        "set",
        *(patterns if cone else ("--no-cone", *patterns)),
    )
    run_git(root, "checkout", "--quiet", "--detach", "FETCH_HEAD")
    actual_revision = run_git(root, "rev-parse", "HEAD")
    if actual_revision != revision:
        raise RuntimeError(f"source revision mismatch: {actual_revision} != {revision}")
    run_git(root, "remote", "remove", "upstream")
    if run_git(root, "rev-parse", "--is-shallow-repository") != "true":
        raise RuntimeError("prepared task source is not shallow")
    return root


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def prepare(output_root: Path, task_root: Path) -> dict:
    output_root = output_root.expanduser().resolve()
    task_root = task_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {output_root}")
    if (
        output_root == task_root
        or task_root in output_root.parents
        or output_root in task_root.parents
    ):
        raise ValueError("prepared source and task definition must be separate trees")

    output_root.mkdir(parents=True)
    candidate = fetch_checkout(
        output_root / "candidate-workspace",
        BASELINE,
        patterns=EDITABLE,
        cone=False,
    )
    baseline = fetch_checkout(
        output_root / "trusted-baseline",
        BASELINE,
        patterns=("mlx_lm",),
        cone=True,
    )
    fixed = fetch_checkout(
        output_root / "trusted-known-fix",
        KNOWN_FIX,
        patterns=EDITABLE,
        cone=False,
    )
    evaluator_checkout = fetch_checkout(
        output_root / "trusted-evaluator-checkout",
        KNOWN_FIX,
        patterns=(TEST_FILE,),
        cone=False,
    )
    evaluator = output_root / "trusted-grader-test"
    evaluator.mkdir()
    (evaluator / "test_tuner_trainer.py").write_bytes(
        (evaluator_checkout / TEST_FILE).read_bytes()
    )

    baseline_hashes = {name: sha256_file(candidate / name) for name in EDITABLE}
    if run_git(candidate, "status", "--porcelain") or "":
        raise RuntimeError("candidate source checkout is not clean")
    if (candidate / TEST_FILE).exists():
        raise RuntimeError("grader test unexpectedly exists in the candidate workspace")
    expected_test_hash = (
        "92c8954f802681b8ddecc4874f3cc60f2379e568580b6b4e959846c99e8c6f17"
    )
    grader_test = evaluator / "test_tuner_trainer.py"
    actual_test_hash = sha256_file(grader_test)
    if actual_test_hash != expected_test_hash:
        raise RuntimeError(f"grader test hash mismatch: {actual_test_hash}")
    if (
        sha256_file(baseline / "LICENSE")
        != "ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01"
    ):
        raise RuntimeError("upstream license hash mismatch")
    for name, expected in {
        "mlx_lm/models/granitemoe.py": "26186a3e66429f38900164764a9da8f0d9a32e7ca30941607bf65b69d36415af",
        "mlx_lm/models/granitemoehybrid.py": "a9d4214a84d2ecb998d3ea00c6ff6de82c2b5d0a08680149739405f4466e0a19",
        "mlx_lm/models/lfm2_moe.py": "cf1bda393544ff770a1f1961df2dad9e743a57db68f9574604948e8c322ce52a",
    }.items():
        if (
            baseline_hashes[name] != expected
            or sha256_file(baseline / name) != expected
        ):
            raise RuntimeError(f"pinned baseline file hash mismatch: {name}")
    fixed_hashes = {name: sha256_file(fixed / name) for name in EDITABLE}
    if any(fixed_hashes[name] == baseline_hashes[name] for name in EDITABLE):
        raise RuntimeError("known-fix checkout does not change all three target files")

    manifest = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "baseline_revision": BASELINE,
        "known_fix_revision": KNOWN_FIX,
        "candidate_workspace": str(candidate),
        "trusted_baseline_source": str(baseline),
        "trusted_known_fix_source": str(fixed),
        "trusted_evaluator_checkout": str(evaluator_checkout),
        "trusted_grader_test": str(grader_test),
        "candidate_visible_files": list(EDITABLE),
        "candidate_workspace_has_test_file": False,
        "grader_test_directory_has_git_metadata": (evaluator / ".git").exists(),
        "baseline_file_sha256": baseline_hashes,
        "known_fix_file_sha256": fixed_hashes,
        "grader_test_sha256": actual_test_hash,
        "license_sha256": sha256_file(baseline / "LICENSE"),
        "task_json_sha256": sha256_file(task_root / "task.json"),
        "task_md_sha256": sha256_file(task_root / "TASK.md"),
        "protocol_lock_sha256": sha256_file(task_root / "protocol.lock.json"),
    }
    write_json(output_root / "prepared-source-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.output_root, args.task_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
