#!/usr/bin/env python3
"""Apply the locked first-pass screen to captured upstream PR metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCREEN_VERSION = "mlsys-task-candidate-coarse-screen-v1"
FRESH_AFTER = "2025-08-09"
SPECIALIZED_TITLE_MARKERS = (
    "gpu", "metal", "cuda", "fsdp", "distributed", "expert parallel",
    "neuron", "architecture", "support ", "add ", "integration test",
    "preemption",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def screen(pool_path: Path, output_dir: Path) -> dict:
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    rows = pool.get("candidates")
    if type(rows) is not list:
        raise ValueError("candidate pool has no candidate list")
    screened = []
    for row in rows:
        reasons = []
        if row.get("merged") is not True or not row.get("merge_commit_sha"):
            reasons.append("not_merged_or_missing_merge_commit")
        merged_date = (row.get("merged_at") or "")[:10]
        if merged_date <= FRESH_AFTER:
            reasons.append("not_strictly_after_model_training_cutoff_proxy")
        if not row.get("test_paths"):
            reasons.append("no_test_path_in_merged_pr_diff")
        if (row.get("additions") or 0) + (row.get("deletions") or 0) > 250:
            reasons.append("diff_exceeds_250_changed_lines")
        if (row.get("changed_file_count") or 0) > 8:
            reasons.append("diff_exceeds_8_files")
        title = (row.get("title") or "").lower()
        if any(marker in title for marker in SPECIALIZED_TITLE_MARKERS):
            reasons.append("title_flags_specialized_or_feature_scope_for_manual_review")
        screened.append({
            "repository": row.get("repository"),
            "number": row.get("number"),
            "title": row.get("title"),
            "url": row.get("url"),
            "base_sha": row.get("base_sha"),
            "head_sha": row.get("head_sha"),
            "merge_commit_sha": row.get("merge_commit_sha"),
            "merged_at": row.get("merged_at"),
            "changed_file_count": row.get("changed_file_count"),
            "changed_paths": row.get("changed_paths"),
            "test_paths": row.get("test_paths"),
            "changed_lines": (row.get("additions") or 0) + (row.get("deletions") or 0),
            "coarse_screen": "provisional_for_deep_review" if not reasons else "excluded_from_shortlist",
            "screen_reasons": reasons,
        })
    shortlist = [row for row in screened if row["coarse_screen"] == "provisional_for_deep_review"]
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "screen_id": SCREEN_VERSION,
        "input_sha256": sha256_file(pool_path),
        "model_training_cutoff_proxy": FRESH_AFTER,
        "candidate_count": len(screened),
        "provisional_count": len(shortlist),
        "excluded_count": len(screened) - len(shortlist),
        "rule": {
            "merged": True,
            "merge_commit_required": True,
            "strict_merged_date_after": FRESH_AFTER,
            "test_path_in_diff_required": True,
            "max_changed_lines": 250,
            "max_changed_files": 8,
            "specialized_title_markers": list(SPECIALIZED_TITLE_MARKERS),
            "meaning": "coarse triage only; not proof of task independence, test reproducibility, licensing fit, or final held-out eligibility",
        },
        "candidates": screened,
    }
    path = output_dir / "screen.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "files": {
        "screen.json": {"bytes": path.stat().st_size, "sha256": sha256_file(path)},
    }}
    (output_dir / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {"candidate_count": len(screened), "provisional_count": len(shortlist), "excluded_count": len(screened)-len(shortlist)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pool", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(screen(args.pool, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
