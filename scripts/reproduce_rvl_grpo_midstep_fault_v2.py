#!/usr/bin/env python3
"""Reproduce the frozen RVL rollback baseline/fix comparison from clean trees.

The caller supplies a checkout of the pinned RVL source and a new output path.
This script extracts the two frozen VARE commits from Git archives, runs the
actual v2 integration validator in both trees, and checks the declared 12/14
baseline versus 14/14 fixed contract. It does not modify the current checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


BASELINE_COMMIT = "eeae165"
PATCHED_COMMIT = "167e9d1ca3c2041a2cf5b5a6a924e364de18027d"
RVL_COMMIT = "c7e646b043cb56e5ea3c2623bb8a61e065451f72"
MODE_CHECKS = {"module_modes_restored", "root_mode_restored"}


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def extract_commit(repo: Path, commit: str, destination: Path) -> None:
    archive = destination.with_suffix(".tar")
    subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", "--output", str(archive), commit],
        check=True,
    )
    destination.mkdir(parents=True)
    try:
        with tarfile.open(archive) as bundle:
            # Git creates every archive member from tracked paths. Still reject
            # links and path traversal so extraction stays inside the temp tree.
            root = destination.resolve()
            for member in bundle.getmembers():
                target = (destination / member.name).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"unsafe Git archive path: {member.name}")
                if member.issym() or member.islnk():
                    raise ValueError(f"unexpected link in Git archive: {member.name}")
            bundle.extractall(destination, filter="data")
    finally:
        archive.unlink(missing_ok=True)


def run_arm(tree: Path, arm: str, rvl_source: Path, output: Path) -> dict:
    runner = tree / "scripts/validate_rvl_grpo_midstep_fault_v2.py"
    result = subprocess.run(
        [sys.executable, str(runner), "--rvl-source", str(rvl_source)],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        },
    )
    (output / f"{arm}.stdout.json").write_text(result.stdout, encoding="utf-8")
    (output / f"{arm}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if not result.stdout.strip():
        raise RuntimeError(f"{arm} validator produced no JSON output (exit {result.returncode})")
    record = json.loads(result.stdout)
    record["process_exit_code"] = result.returncode
    return record


def validate_arm(record: dict, *, fixed: bool) -> None:
    checks = record.get("checks")
    if not isinstance(checks, dict):
        raise AssertionError("validator output has no checks map")
    failed = {name for name, value in checks.items() if value is not True}
    if fixed:
        if failed or record.get("status") != "pass" or record["process_exit_code"] != 0:
            raise AssertionError(f"fixed arm did not pass every check: {sorted(failed)}")
    else:
        expected_total = 14
        if len(checks) != expected_total or failed != MODE_CHECKS:
            raise AssertionError(f"baseline failures differ from frozen defect: {sorted(failed)}")
        if record.get("status") != "fail" or record["process_exit_code"] != 1:
            raise AssertionError("baseline did not fail only at the declared validator gate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vare-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rvl-source", type=Path, required=True,
                        help="src/rvl_systems directory from the exact pinned RVL checkout")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new or empty directory for retained raw outputs")
    args = parser.parse_args()

    repo = args.vare_repo.resolve()
    source = args.rvl_source.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rvl_root = Path(git(source, "rev-parse", "--show-toplevel"))
    actual_rvl_commit = git(rvl_root, "rev-parse", "HEAD")
    if actual_rvl_commit != RVL_COMMIT:
        raise SystemExit(f"RVL checkout is {actual_rvl_commit}; expected {RVL_COMMIT}")

    with __import__("tempfile").TemporaryDirectory(prefix="vare-rvl-rollback-v2-") as temp:
        root = Path(temp)
        baseline_tree = root / "baseline"
        patched_tree = root / "patched"
        extract_commit(repo, BASELINE_COMMIT, baseline_tree)
        extract_commit(repo, PATCHED_COMMIT, patched_tree)
        baseline = run_arm(baseline_tree, "baseline", source, output)
        patched = run_arm(patched_tree, "patched", source, output)
        validate_arm(baseline, fixed=False)
        validate_arm(patched, fixed=True)

    summary = {
        "protocol": "rvl_grpo_midstep_fault_v2",
        "vare_baseline_commit": git(repo, "rev-parse", BASELINE_COMMIT),
        "vare_patched_commit": git(repo, "rev-parse", PATCHED_COMMIT),
        "rvl_commit": actual_rvl_commit,
        "runtime": {"python": platform.python_version()},
        "baseline": {
            "status": baseline["status"],
            "passed_checks": sum(value is True for value in baseline["checks"].values()),
            "total_checks": len(baseline["checks"]),
            "failed_checks": sorted(name for name, value in baseline["checks"].items() if value is not True),
            "wall_seconds": baseline["wall_seconds"],
        },
        "patched": {
            "status": patched["status"],
            "passed_checks": sum(value is True for value in patched["checks"].values()),
            "total_checks": len(patched["checks"]),
            "failed_checks": sorted(name for name, value in patched["checks"].items() if value is not True),
            "wall_seconds": patched["wall_seconds"],
        },
        "claim_boundary": "Same-host clean-tree reproduction of one frozen injected fault on the exact pinned RVL trainer. This is not an outside-person reproduction, production-incidence estimate, or model-capability result.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
