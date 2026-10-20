#!/usr/bin/env python3
"""Independently verify a separated MLSys harness calibration bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_paths(run_dir: Path) -> set[str]:
    """Return payload paths, excluding only root audit bookkeeping files."""
    return {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in {"evidence_manifest.json", "audit.json"}
    }


def source_commit_matches_files(
    repo_root: Path, source_commit: str, source_hashes: dict[str, str],
) -> bool:
    """Verify the recorded commit object and every source blob it names."""
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        return False
    present = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{source_commit}^{{commit}}"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if present.returncode != 0:
        return False
    for relative, expected_hash in source_hashes.items():
        blob = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{source_commit}:{relative}"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != expected_hash:
            return False
    return True


def audit(run_dir: Path) -> dict:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "evidence_manifest.json"
    run_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = json.loads(run_path.read_text(encoding="utf-8"))
    if record.get("protocol_id") != "mlsys-separated-candidate-harness-v2":
        raise ValueError("run does not use the v2 frozen protocol")
    files = manifest.get("files")
    if type(files) is not dict:
        raise ValueError("evidence manifest is malformed")
    actual = evidence_paths(run_dir)
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
    if controls.get("frozen_checker_rejects_empty_result") is not True:
        raise ValueError("frozen checker did not reject the malformed empty result")
    if controls.get("isolated_grader_rejects_empty_result") is not True:
        raise ValueError("isolated grader did not reject the empty result")
    if controls.get("isolated_grader_rejection_reason") != "correctness_failed":
        raise ValueError("empty-result control was not rejected for incorrect output")
    if controls.get("isolated_grader_accepts_oracle_control") is not True:
        raise ValueError("known-good oracle control did not pass")
    raw = record.get("raw", {})
    bad = raw.get("empty_result", {})
    bad_cases = bad.get("grade_evidence", {}).get("cases", [])
    good = raw.get("oracle_control", {})
    if (
        bad.get("passed") is not False
        or bad.get("error") != "correctness_failed"
        or bad.get("nonce_enforced") is not True
        or len(bad_cases) != 5
        or any(
            type(case) is not dict
            or case.get("passed") is not False
            or case.get("failure") != "wrong_output"
            for case in bad_cases
        )
    ):
        raise ValueError("raw empty-result evidence does not show five wrong-output rejections")
    if good.get("passed") is not True or good.get("nonce_enforced") is not True:
        raise ValueError("raw oracle evidence does not show an accepted isolated control")
    if record.get("classification") != (
        "harness calibration on a previously consumed task; not model or held-out evidence"
    ):
        raise ValueError("result claim classification is missing or invalid")
    source_paths = {
        "protocol_sha256": "protocols/mlsys_separated_candidate_harness_v2.json",
        "task_descriptor_sha256": "benchmarks/mlsys_coding_v1/tasks/causal_attention/task.json",
        "hidden_evaluator_sha256": "benchmarks/mlsys_coding_v1/tasks/causal_attention/evaluator/test_hidden.py",
        "oracle_sha256": "benchmarks/mlsys_coding_v1/tasks/causal_attention/evaluator/oracle/causal_attention.py",
        "pack_manifest_sha256": "benchmarks/mlsys_coding_v1/manifest.json",
        "frozen_harness_adapter_sha256": "benchmarks/mlsys_coding_v1/harness_adapter.py",
        "adapter_source_sha256": "scripts/mlsys_task_isolated_calls.py",
        "verifier_source_sha256": "scripts/verify_mlsys_separated_candidate_harness_v2.py",
        "auditor_source_sha256": "scripts/audit_mlsys_separated_candidate_harness_v2.py",
        "harness_source_sha256": "src/vare/separated_candidate_harness.py",
        "isolated_grader_source_sha256": "src/vare/isolated_candidate.py",
        "separated_grader_source_sha256": "src/vare/separated_grading.py",
    }
    if "runner_source_sha256" in record:
        source_paths["runner_source_sha256"] = "scripts/run_separated_coding_eval.py"
    source_hashes = {relative: record.get(field) for field, relative in source_paths.items()}
    for relative, expected_hash in source_hashes.items():
        if sha256_file(ROOT / relative) != expected_hash:
            raise ValueError(f"current source hash differs from run provenance: {relative}")
    source_commit = record.get("source_commit")
    if type(source_commit) is not str or not source_commit_matches_files(ROOT, source_commit, source_hashes):
        raise ValueError("recorded source commit is missing or its source blobs differ from provenance")
    return {
        "schema_version": 1,
        "files_verified": len(files),
        "task_id": record.get("task_id"),
        "frozen_checker_rejected_empty_result": True,
        "isolated_grader_rejected_empty_result": True,
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
