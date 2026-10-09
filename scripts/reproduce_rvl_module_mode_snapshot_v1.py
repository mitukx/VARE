#!/usr/bin/env python3
"""Reproduce the pinned RVL module-mode snapshot regression and patch.

The driver clones a disposable checkout at the frozen upstream revision, runs
the test-only regression against the baseline, resets that clone, then applies
the combined patch and runs RVL's optional torch test module. It never changes
the caller's RVL checkout and does not install dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/mitukx/Recursive-Verification-Lag.git"
UPSTREAM_COMMIT = "c7e646b043cb56e5ea3c2623bb8a61e065451f72"
TEST_ONLY_PATCH = ROOT / "contributions/rvl-module-mode-test-only-v1.patch"
COMBINED_PATCH = ROOT / "contributions/rvl-module-mode-snapshot-v1.patch"
LOCK = ROOT / "protocols/rvl_module_mode_snapshot_upstream_v1.lock.json"
REGRESSION = (
    "tests.test_mini_lab_torch.TorchAcceptanceTests."
    "test_hf_trainer_transaction_restores_mixed_module_modes"
)
LEGACY_CONTROL = (
    "tests.test_mini_lab_torch.TorchAcceptanceTests."
    "test_hf_trainer_restore_accepts_legacy_snapshot"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout: int = 300
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout
    )


def retain_run(output: Path, name: str, result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    combined = result.stdout + ("\n" if result.stdout and result.stderr else "") + result.stderr
    log_path = output / f"{name}.log"
    log_path.write_text(combined, encoding="utf-8")
    return {
        "argv": result.args,
        "return_code": result.returncode,
        "log": log_path.name,
        "log_sha256": digest(log_path),
    }


def check_test_runtime(python: str) -> dict[str, str]:
    code = (
        "import importlib.metadata as m, json, sys; "
        "print(json.dumps({'python':sys.version.split()[0],"
        "'torch':m.version('torch'),'transformers':m.version('transformers')}))"
    )
    result = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("could not inspect test runtime: " + result.stderr.strip())
    versions = json.loads(result.stdout)
    if tuple(map(int, versions["python"].split(".")[:2])) < (3, 11):
        raise RuntimeError("RVL optional suite uses asyncio.TaskGroup and requires Python >= 3.11")
    if not versions.get("torch") or not versions.get("transformers"):
        raise RuntimeError("test runtime must already have torch and transformers installed")
    return versions


def validate_baseline(result: subprocess.CompletedProcess[str]) -> None:
    output = result.stdout + result.stderr
    expected = (
        result.returncode != 0
        and "test_hf_trainer_transaction_restores_mixed_module_modes" in output
        and "FAIL" in output
        and "test_hf_trainer_restore_accepts_legacy_snapshot" in output
        and "ok" in output
        and "Ran 2 tests" in output
        and "FAILED (failures=1)" in output
    )
    if not expected:
        raise AssertionError("baseline must fail only the mixed-mode regression and pass the legacy control")


def validate_fixed(result: subprocess.CompletedProcess[str]) -> None:
    output = result.stdout + result.stderr
    expected = (
        result.returncode == 0
        and "Ran 13 tests" in output
        and "OK" in output
    )
    if not expected:
        raise AssertionError("patched optional RVL module must pass all 13 tests")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for the clone's baseline/fixed logs and summary")
    parser.add_argument("--python", default=sys.executable,
                        help="Python >=3.11 with torch and transformers already installed")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("refusing to overwrite a non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    expected_hashes = lock["patches"]
    if digest(TEST_ONLY_PATCH) != expected_hashes["test_only_sha256"]:
        raise SystemExit("test-only patch differs from the frozen protocol digest")
    if digest(COMBINED_PATCH) != expected_hashes["combined_sha256"]:
        raise SystemExit("combined patch differs from the frozen protocol digest")
    runtime = check_test_runtime(args.python)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    runs: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="vare-rvl-module-mode-") as temp:
        clone = Path(temp) / "rvl"
        clone_result = run(
            ["git", "clone", "--no-checkout", UPSTREAM, str(clone)],
            cwd=ROOT, env=env, timeout=300,
        )
        runs["clone"] = retain_run(output, "clone", clone_result)
        if clone_result.returncode:
            raise RuntimeError("upstream clone failed; see clone.log")

        checkout = run(
            ["git", "checkout", "--detach", UPSTREAM_COMMIT],
            cwd=clone, env=env,
        )
        runs["checkout"] = retain_run(output, "checkout", checkout)
        if checkout.returncode:
            raise RuntimeError("could not check out the frozen RVL commit; see checkout.log")
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=clone, text=True
        ).strip()
        if actual != UPSTREAM_COMMIT:
            raise RuntimeError(f"checked out {actual}, expected {UPSTREAM_COMMIT}")

        apply_test = run(["git", "apply", str(TEST_ONLY_PATCH)], cwd=clone, env=env)
        runs["apply_test_only"] = retain_run(output, "apply-test-only", apply_test)
        if apply_test.returncode:
            raise RuntimeError("test-only patch failed to apply; see apply-test-only.log")
        baseline = run(
            [args.python, "-m", "unittest", REGRESSION, LEGACY_CONTROL, "-v"],
            cwd=clone, env=env,
        )
        runs["baseline_regression"] = retain_run(output, "baseline-regression", baseline)
        validate_baseline(baseline)

        reset = run(["git", "reset", "--hard", UPSTREAM_COMMIT], cwd=clone, env=env)
        runs["reset"] = retain_run(output, "reset", reset)
        if reset.returncode:
            raise RuntimeError("could not reset disposable clone; see reset.log")
        apply_fix = run(["git", "apply", str(COMBINED_PATCH)], cwd=clone, env=env)
        runs["apply_combined"] = retain_run(output, "apply-combined", apply_fix)
        if apply_fix.returncode:
            raise RuntimeError("combined patch failed to apply; see apply-combined.log")
        whitespace = run(["git", "diff", "--check"], cwd=clone, env=env)
        runs["diff_check"] = retain_run(output, "diff-check", whitespace)
        if whitespace.returncode:
            raise RuntimeError("patched diff has whitespace errors; see diff-check.log")
        fixed = run(
            [args.python, "-m", "unittest", "tests.test_mini_lab_torch", "-v"],
            cwd=clone, env=env,
        )
        runs["patched_full_suite"] = retain_run(output, "patched-full-suite", fixed)
        validate_fixed(fixed)

    record = {
        "status": "pass",
        "upstream_repository": UPSTREAM,
        "upstream_commit": UPSTREAM_COMMIT,
        "patches": {
            "test_only_sha256": digest(TEST_ONLY_PATCH),
            "combined_sha256": digest(COMBINED_PATCH),
        },
        "runtime": {"python": runtime["python"], "torch": runtime["torch"],
                    "transformers": runtime["transformers"], "device": "CPU"},
        "runs": runs,
        "baseline": "mixed-mode regression fails; legacy snapshot control passes",
        "patched": "all 13 optional RVL PyTorch tests pass",
        "clone_disposed_after_run": True,
        "external_human_reproduction": False,
        "external_submission": False,
        "claim_boundary": (
            "Clean-clone, same-host CPU reproduction of a pinned RVL rollback-state omission. "
            "Does not establish production incidence, pretrained-model impact, capability gain, "
            "upstream adoption, or independent outside review."
        ),
    }
    summary = output / "summary.json"
    summary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": record["status"],
        "upstream_commit": UPSTREAM_COMMIT,
        "runtime": record["runtime"],
        "baseline": record["baseline"],
        "patched": record["patched"],
        "summary": str(summary),
        "external_human_reproduction": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
