#!/usr/bin/env python3
"""Calibrate exact full-workspace public-test inventory under Colima."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import calibrate_public_test_execution_v7 as parent_calibration  # noqa: E402
from run_local_agent import run_public_tests  # noqa: E402

PROTOCOL = ROOT / "experiments/coding_agent_phase_a_v6/public-test-execution-v8.json"
LOCK = ROOT / "experiments/coding_agent_phase_a_v6/public-test-execution-v8.lock.json"
IMAGE = "vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_workspace_probe() -> dict:
    with tempfile.TemporaryDirectory(prefix="vare-full-public-workspace-") as temp:
        root = Path(temp) / "workspace"
        (root / "src").mkdir(parents=True)
        (root / "src/value.py").write_text("def value():\n    return 42\n", encoding="utf-8")
        (root / "src/helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        test_bytes = (
            "from value import value\n\ndef test_value():\n    assert value() == 42\n"
        ).encode()
        (root / "test_public.py").write_bytes(test_bytes)
        expected_hashes = {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(root.rglob("*")) if path.is_file()
        }
        expected_files = set(expected_hashes)
        fix = run_public_tests(
            root,
            IMAGE,
            expected_hashes["test_public.py"],
            expected_workspace_files=expected_files,
            expected_file_hashes=expected_hashes,
            allowed_modified_files={"src/value.py"},
        )
        (root / "src/helper.py").write_text("VALUE = 2\n", encoding="utf-8")
        tampered = run_public_tests(
            root,
            IMAGE,
            expected_hashes["test_public.py"],
            expected_workspace_files=expected_files,
            expected_file_hashes=expected_hashes,
            allowed_modified_files={"src/value.py"},
        )
        result = {
            "expected_file_count": len(expected_files),
            "candidate_edit_allowlist": ["src/value.py"],
            "fix": {
                "classification": fix.get("classification"),
                "passed": fix.get("passed"),
                "gradeable": fix.get("gradeable"),
            },
            "unauthorized_dependency_edit": {
                "classification": tampered.get("classification"),
                "passed": tampered.get("passed"),
                "gradeable": tampered.get("gradeable"),
            },
        }
        if not (
            result["expected_file_count"] == 3
            and result["fix"] == {
                "classification": "passed", "passed": True, "gradeable": True,
            }
            and result["unauthorized_dependency_edit"] == {
                "classification": "runner_integrity_error", "passed": None, "gradeable": False,
            }
        ):
            raise RuntimeError("full-workspace inventory calibration failed")
        return result


def run(output: Path) -> dict:
    if os.environ.get("DOCKER_CONTEXT") != "colima":
        raise RuntimeError("DOCKER_CONTEXT=colima is required")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite calibration output: {output}")
    parent_calibration.PROTOCOL = PROTOCOL
    parent_calibration.LOCK = LOCK
    with tempfile.TemporaryDirectory(prefix="vare-v8-parent-calibration-") as temp:
        record = parent_calibration.run(Path(temp) / "parent")
    started = time.monotonic()
    record["full_workspace_inventory"] = full_workspace_probe()
    record["wall_seconds"] = round(record["wall_seconds"] + time.monotonic() - started, 3)
    record["limitations"] = [
        "Synthetic three-file package fixture; not a repository task or model rollout.",
        "Validates exact full-workspace inventory, editable-file exception, staging, and evaluator classification only.",
        "The HF Development cohort associated with PR #107 was previously model-exposed and is not fresh evidence.",
    ]
    output.mkdir(parents=True)
    result_path = output / "calibration.json"
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        f"{sha256_file(result_path)}  calibration.json\n", encoding="utf-8",
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = run(args.output)
    except Exception as exc:
        print(f"calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "calibration_passed": record["calibration_passed"],
        "full_workspace_inventory": record["full_workspace_inventory"],
        "wall_seconds": record["wall_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
