#!/usr/bin/env python3
"""Independently verify a separated MLSys harness calibration bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(run_dir: Path) -> dict:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "evidence_manifest.json"
    run_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = json.loads(run_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if type(files) is not dict:
        raise ValueError("evidence manifest is malformed")
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in {"evidence_manifest.json", "audit.json"}
    }
    if actual != set(files):
        raise ValueError("evidence inventory differs from manifest")
    for name, expected in files.items():
        path = run_dir / name
        if (
            path.stat().st_size != expected.get("bytes")
            or sha256_file(path) != expected.get("sha256")
        ):
            raise ValueError(f"evidence file hash mismatch: {name}")
    controls = record.get("controls", {})
    if controls.get("frozen_nested_zip_checker_accepts_empty_result") is not True:
        raise ValueError("baseline vulnerability reproducer did not hold")
    if controls.get("isolated_grader_rejects_empty_result") is not True:
        raise ValueError("isolated grader did not reject the empty result")
    if controls.get("isolated_grader_accepts_oracle_control") is not True:
        raise ValueError("known-good oracle control did not pass")
    if record.get("classification") != (
        "harness calibration on a previously consumed task; not model or held-out evidence"
    ):
        raise ValueError("result claim classification is missing or invalid")
    source_paths = {
        "protocol_sha256": "protocols/mlsys_separated_candidate_harness_v1.json",
        "task_descriptor_sha256": "benchmarks/mlsys_coding_v1/tasks/global_grad_clip/task.json",
        "hidden_evaluator_sha256": "benchmarks/mlsys_coding_v1/tasks/global_grad_clip/evaluator/test_hidden.py",
        "oracle_sha256": "benchmarks/mlsys_coding_v1/tasks/global_grad_clip/evaluator/oracle/global_grad_clip.py",
        "pack_manifest_sha256": "benchmarks/mlsys_coding_v1/manifest.json",
        "frozen_harness_adapter_sha256": "benchmarks/mlsys_coding_v1/harness_adapter.py",
        "adapter_source_sha256": "scripts/mlsys_task_isolated_calls.py",
        "harness_source_sha256": "src/vare/separated_candidate_harness.py",
        "isolated_grader_source_sha256": "src/vare/isolated_candidate.py",
        "separated_grader_source_sha256": "src/vare/separated_grading.py",
    }
    for field, relative in source_paths.items():
        if sha256_file(ROOT / relative) != record.get(field):
            raise ValueError(f"current source hash differs from run provenance: {relative}")
    import subprocess
    current_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if record.get("source_commit") != current_commit:
        raise ValueError("recorded source commit differs from the audited checkout")
    return {
        "schema_version": 1,
        "files_verified": len(files),
        "task_id": record.get("task_id"),
        "baseline_empty_shape_accepted": True,
        "isolated_empty_shape_rejected": True,
        "oracle_control_passed": True,
        "model_capability_claim": False,
        "audit_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    (args.run_dir / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
