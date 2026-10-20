#!/usr/bin/env python3
"""Audit the frozen candidate-pool and coarse-screen evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/mlsys_task_candidate_survey_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(bundle: Path, data_name: str) -> dict:
    manifest = json.loads((bundle / "evidence-manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files")
    if type(files) is not dict or data_name not in files:
        raise ValueError(f"{bundle}: evidence manifest is malformed")
    expected_names = {data_name}
    actual_names = {
        path.name for path in bundle.iterdir()
        if path.is_file() and path.name != "evidence-manifest.json"
    }
    if actual_names != expected_names:
        raise ValueError(f"{bundle}: evidence inventory differs from manifest")
    path = bundle / data_name
    expected = files[data_name]
    if path.stat().st_size != expected.get("bytes") or sha256_file(path) != expected.get("sha256"):
        raise ValueError(f"{bundle}: data hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def audit(root: Path) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pool = verify_bundle(root / "run-1", "candidate-pool.json")
    screen = verify_bundle(root / "screen-1", "screen.json")
    candidates = pool.get("candidates")
    screened = screen.get("candidates")
    if type(candidates) is not list or type(screened) is not list:
        raise ValueError("candidate or screen rows are missing")
    if pool.get("candidate_count_requested") != 57 or len(candidates) != 57:
        raise ValueError("candidate count differs from the frozen 57-entry inventory")
    if any(
        row.get("merged") is not True
        or not row.get("merge_commit_sha")
        or not row.get("base_sha")
        or not row.get("head_sha")
        or row.get("changed_file_count") != len(row.get("changed_paths", []))
        for row in candidates
    ):
        raise ValueError("a candidate is not a merged PR with a merge SHA")
    keys = [(row.get("repository"), row.get("number")) for row in candidates]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate candidate IDs found")
    if len(screened) != len(candidates):
        raise ValueError("screen inventory does not cover the candidate pool")
    for original, row in zip(candidates, screened, strict=True):
        if (original.get("repository"), original.get("number")) != (row.get("repository"), row.get("number")):
            raise ValueError("screen candidate ordering or identity differs from source pool")
        if row.get("base_sha") != original.get("base_sha") or row.get("merge_commit_sha") != original.get("merge_commit_sha"):
            raise ValueError("screen source revisions differ from raw PR metadata")
    provisional = [row for row in screened if row.get("coarse_screen") == "provisional_for_deep_review"]
    if len(provisional) != 24 or screen.get("provisional_count") != 24:
        raise ValueError("coarse shortlist count is not 24")
    if protocol["discovery"]["pool_sha256"] != sha256_file(root / "run-1/candidate-pool.json"):
        raise ValueError("protocol candidate-pool hash mismatch")
    if protocol["discovery"]["coarse_screen"]["screen_sha256"] != sha256_file(root / "screen-1/screen.json"):
        raise ValueError("protocol screen hash mismatch")
    if protocol["task_gate"]["fully_qualified_tasks"] != 0 or protocol["task_gate"]["training_authorized"] is not False:
        raise ValueError("protocol does not preserve the task qualification NO-GO")
    result = {
        "schema_version": 1,
        "audit_passed": True,
        "candidate_count": len(candidates),
        "all_candidates_merged": True,
        "coarse_provisional_count": len(provisional),
        "fully_qualified_count": 0,
        "final_split_frozen": False,
        "training_authorized": False,
        "decision": "NO_GO_PENDING_DEEP_TASK_QUALIFICATION",
        "protocol_sha256": sha256_file(PROTOCOL),
        "candidate_pool_sha256": sha256_file(root / "run-1/candidate-pool.json"),
        "screen_sha256": sha256_file(root / "screen-1/screen.json"),
    }
    audit_path = root / "audit.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"audit.json", "evidence-manifest.json"}:
            relative = path.relative_to(root).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (root / "evidence-manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.bundle.expanduser().resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
