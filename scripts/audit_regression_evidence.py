#!/usr/bin/env python3
"""Audit retained current-regression bundles against locks and a fresh replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/regression_calibration_v1.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise ValueError(message)


def audit_manifest(bundle: Path) -> int:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        fail(f"missing manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    entries = manifest.get("files")
    if not isinstance(entries, dict):
        fail(f"manifest files must be a mapping: {manifest_path}")
    expected: set[str] = set()
    for name, digest in entries.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or not SHA256_RE.fullmatch(str(digest))):
            fail(f"unsafe path or malformed SHA-256 in {manifest_path}: {name}")
        path = bundle.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(bundle.resolve()) or not path.is_file():
            fail(f"manifest entry is missing: {path}")
        observed = sha256_file(path)
        if observed != digest:
            fail(f"manifest hash mismatch: {path}")
        expected.add(relative.as_posix())
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        missing = sorted(expected - actual)
        unlisted = sorted(actual - expected)
        fail(f"manifest inventory mismatch at {bundle}; missing={missing}, unlisted={unlisted}")
    return len(entries)


def validate_locked_inputs(entry: dict[str, Any], task_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    task = read_json(task_root / "task.json")
    lock = read_json(task_root / "protocol.lock.json")
    if task.get("task_id") != entry["task_id"] or lock.get("task_id") != entry["task_id"]:
        fail(f"task id mismatch for {entry['task_id']}")
    if sha256_file(task_root / "protocol.lock.json") != entry["protocol_lock_sha256"]:
        fail(f"protocol lock changed for {entry['task_id']}")
    evaluator = ROOT / task["grader"]["path"]
    expected = lock["locked_hashes"]
    observed = {
        "task_json_sha256": sha256_file(task_root / "task.json"),
        "task_brief_sha256": sha256_file(task_root / "TASK.md"),
        "evaluator_sha256": sha256_file(evaluator),
    }
    for key, value in observed.items():
        if expected.get(key) != value:
            fail(f"current locked input mismatch for {entry['task_id']}: {key}")
    return task, lock


def validate_bundle(entry: dict[str, Any], protocol: dict[str, Any],
                    retained: Path, regenerated: Path) -> int:
    task_root = ROOT / entry["task_root"]
    task, lock = validate_locked_inputs(entry, task_root)
    count = audit_manifest(retained)
    audit_manifest(regenerated)
    retained_manifest = read_json(retained / "manifest.json")
    if retained_manifest.get("task_id") not in (None, entry["task_id"]):
        fail(f"retained manifest task id mismatch: {entry['task_id']}")
    summary = read_json(retained / "summary.json")
    if summary.get("task_id") != entry["task_id"]:
        fail(f"retained summary task id mismatch: {entry['task_id']}")
    if summary.get("protocol_id") != lock.get("protocol_id"):
        fail(f"retained summary protocol id mismatch: {entry['task_id']}")
    recorded_calibration_protocol = summary.get("calibration_protocol_sha256")
    if (recorded_calibration_protocol is not None
            and recorded_calibration_protocol != sha256_file(PROTOCOL_PATH)):
        fail(f"retained calibration protocol hash mismatch: {entry['task_id']}")
    if summary.get("baseline_revision") != task["source"]["base_revision"]:
        fail(f"retained baseline revision mismatch: {entry['task_id']}")
    if "fixed_revision" in summary and summary["fixed_revision"] != entry["fixed_revision"]:
        fail(f"retained fixed revision mismatch: {entry['task_id']}")

    snapshot = retained / "protocol_snapshot"
    evaluator_path = ROOT / task["grader"]["path"]
    expected_snapshot = {
        "task.json": task_root / "task.json",
        "TASK.md": task_root / "TASK.md",
        "protocol.lock.json": task_root / "protocol.lock.json",
    }
    if (snapshot / "calibration_protocol.json").is_file():
        expected_snapshot["calibration_protocol.json"] = PROTOCOL_PATH
    evaluator_snapshot = snapshot / "evaluator/grade.py"
    if not evaluator_snapshot.is_file():
        evaluator_snapshot = snapshot / "grade.py"
    expected_snapshot["evaluator"] = evaluator_path
    observed_snapshot = {
        "task.json": snapshot / "task.json",
        "TASK.md": snapshot / "TASK.md",
        "protocol.lock.json": snapshot / "protocol.lock.json",
        "evaluator": evaluator_snapshot,
    }
    if (snapshot / "calibration_protocol.json").is_file():
        observed_snapshot["calibration_protocol.json"] = snapshot / "calibration_protocol.json"
    for key, current in expected_snapshot.items():
        saved = observed_snapshot[key]
        if not saved.is_file() or sha256_file(saved) != sha256_file(current):
            fail(f"retained protocol snapshot differs from current locked input: {entry['task_id']} {key}")

    expected_outcomes = protocol["expected_outcomes"]
    baseline = read_json(retained / "baseline.grade.json")
    fixed = read_json(retained / "fixed.grade.json")
    outcomes = {"baseline_passed": baseline.get("passed"), "fixed_passed": fixed.get("passed")}
    if outcomes != expected_outcomes:
        fail(f"retained grade outcomes differ from frozen expectations for {entry['task_id']}: {outcomes}")
    if baseline.get("task_id") != entry["task_id"] or fixed.get("task_id") != entry["task_id"]:
        fail(f"retained grade task id mismatch: {entry['task_id']}")

    regenerated_summary = read_json(regenerated / "summary.json")
    if regenerated_summary.get("baseline_revision") != task["source"]["base_revision"]:
        fail(f"replay baseline revision mismatch: {entry['task_id']}")
    if regenerated_summary.get("fixed_revision") != entry["fixed_revision"]:
        fail(f"replay fixed revision mismatch: {entry['task_id']}")
    for label in ("baseline.grade.json", "fixed.grade.json"):
        if read_json(retained / label) != read_json(regenerated / label):
            fail(f"retained grade differs from fresh replay: {entry['task_id']} {label}")
    for label, hashes_key in (("baseline-source", "baseline_source_hashes"),
                              ("fixed-source", "fixed_source_hashes")):
        replay_hashes = regenerated_summary[hashes_key]
        for name, expected_hash in replay_hashes.items():
            retained_path = retained / label / name
            if retained_path.is_file():
                retained_hash = sha256_file(retained_path)
            elif label == "baseline-source":
                baseline_record = read_json(retained / "baseline.json")
                retained_hash = baseline_record.get("baseline_source_sha256")
            else:
                retained_hash = summary.get("fixed_source_sha256")
                if retained_hash is None and name.endswith("/runner.py"):
                    retained_hash = summary.get("fixed_candidate_runner_sha256")
            if retained_hash != expected_hash:
                fail(f"retained source hash differs from fresh pinned replay: {entry['task_id']} {label}/{name}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerated-root", type=Path, required=True,
                        help="output root created by calibrate_regressions.py")
    args = parser.parse_args()
    protocol = read_json(PROTOCOL_PATH)
    regenerated_root = args.regenerated_root.expanduser().resolve()
    if not regenerated_root.is_dir():
        parser.error(f"regenerated evidence root does not exist: {regenerated_root}")
    audit_manifest(regenerated_root)
    root_summary = read_json(regenerated_root / "summary.json")
    if (root_summary.get("protocol_id") != protocol.get("protocol_id")
            or root_summary.get("calibration_protocol_sha256") != sha256_file(PROTOCOL_PATH)):
        parser.error("regenerated root summary does not match the current calibration protocol")
    audited_entries = []
    try:
        for entry in protocol["tasks"]:
            retained = ROOT / "results" / {
                "vare-promotion-gate-nonfinite-metrics-v1": "promotion-gate-metrics-v1",
                "vare-replay-group-freshness-atomicity-v1": "replay-group-freshness-v1",
                "vare-environment-command-output-budget-v4": "environment-output-budget-v4",
            }[entry["task_id"]]
            regenerated = regenerated_root / entry["task_id"]
            file_count = validate_bundle(entry, protocol, retained, regenerated)
            audited_entries.append({"task_id": entry["task_id"], "manifest_files": file_count,
                                    "status": "verified_against_fresh_replay"})
    except Exception as exc:
        print(f"audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "verified", "task_count": len(audited_entries),
                      "tasks": audited_entries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
