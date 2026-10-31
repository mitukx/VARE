#!/usr/bin/env python3
"""Re-screen frozen PR metadata using GitHub GraphQL's MERGED state value."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_screen_v4.json"
SOURCE_PROTOCOL_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_discovery_v3.json"
SOURCE_POOL_PATH = ROOT / "results/mlsys-fresh-development-candidate-discovery-v3/candidate-pool.json.gz"
SOURCE_SCREEN_PATH = ROOT / "results/mlsys-fresh-development-candidate-discovery-v3/screen.json.gz"
SOURCE_MANIFEST_PATH = ROOT / "results/mlsys-fresh-development-candidate-discovery-v3/evidence-manifest.json"
SOURCE_COMPRESSED_MANIFEST_PATH = ROOT / "results/mlsys-fresh-development-candidate-discovery-v3/compressed-evidence-manifest.json"
QUEUE_AUDIT_PATH = ROOT / "results/mlsys-candidate-manual-qualification-queue-exposure-v1.json"
SPECIALIZED_TITLE_MARKERS = (
    "gpu", "metal", "cuda", "fsdp", "distributed", "expert parallel",
    "neuron", "architecture", "support ", "add ", "integration test",
    "preemption",
)
TEST_PATH_MARKERS = ("test", "spec", "fixture")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "mlsys-fresh-development-candidate-screen-v4":
        raise ValueError("unexpected screen protocol ID")
    if protocol.get("screen_sha256") != sha256_file(Path(__file__)):
        raise ValueError("screen source differs from the frozen protocol")
    if protocol.get("source_discovery_protocol_sha256") != sha256_file(SOURCE_PROTOCOL_PATH):
        raise ValueError("source discovery protocol differs from the frozen screen protocol")
    if protocol.get("source_pool_gzip_sha256") != sha256_file(SOURCE_POOL_PATH):
        raise ValueError("compressed source pool differs from the frozen screen protocol")
    if protocol.get("source_screen_gzip_sha256") != sha256_file(SOURCE_SCREEN_PATH):
        raise ValueError("preserved v3 screen differs from the frozen screen protocol")
    if protocol.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH):
        raise ValueError("source evidence manifest differs from the frozen screen protocol")
    if protocol.get("source_compressed_manifest_sha256") != sha256_file(SOURCE_COMPRESSED_MANIFEST_PATH):
        raise ValueError("source compression manifest differs from the frozen screen protocol")
    if protocol.get("queue_audit_sha256") != sha256_file(QUEUE_AUDIT_PATH):
        raise ValueError("prior-exposure audit differs from the frozen screen protocol")
    return protocol


def load_pool(protocol: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    with gzip.open(SOURCE_POOL_PATH, "rb") as stream:
        raw = stream.read()
    if sha256_bytes(raw) != protocol.get("source_pool_sha256"):
        raise ValueError("decompressed candidate pool differs from the frozen source hash")
    payload = json.loads(raw.decode("utf-8"))
    candidates = payload.get("candidates")
    if type(candidates) is not list or len(candidates) != protocol.get("source_candidate_count"):
        raise ValueError("frozen source candidate count differs from the screen protocol")
    identities = [(row.get("repository"), row.get("number")) for row in candidates]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate source candidate identities")
    return payload, raw


def prior_identities() -> set[tuple[str, int]]:
    audit = json.loads(QUEUE_AUDIT_PATH.read_text(encoding="utf-8"))
    identities: set[tuple[str, int]] = set()
    for row in audit.get("candidates", []):
        match = re.fullmatch(r"([^/]+/[^#]+)#([0-9]+)", row.get("identity") or "")
        if match:
            identities.add((match.group(1), int(match.group(2))))
    identities.add(("ml-explore/mlx-lm", 1971))
    return identities


def is_test_path(path: str) -> bool:
    lower = path.lower()
    return any(marker in lower for marker in TEST_PATH_MARKERS)


def screen_candidate(
    row: dict[str, Any],
    *,
    date_start: str,
    date_end: str,
    cutoff_proxy: str,
    prior: set[tuple[str, int]],
) -> dict[str, Any]:
    reasons: list[str] = []
    number = row.get("number")
    identity = (row.get("repository"), number) if type(number) is int else None
    merged_date = (row.get("merged_at") or "")[:10]
    if row.get("state") != "MERGED" or not row.get("merged_at") or not row.get("merge_commit_sha"):
        reasons.append("not_merged_or_missing_merge_commit")
    if not row.get("merge_base_sha"):
        reasons.append("merge_base_commit_missing")
    if not date_start <= merged_date <= date_end:
        reasons.append("merged_outside_frozen_date_window")
    if merged_date <= cutoff_proxy:
        reasons.append("not_after_model_training_cutoff_proxy")
    if not row.get("changed_paths_complete"):
        reasons.append("changed_path_inventory_incomplete")
    paths = row.get("changed_paths") if type(row.get("changed_paths")) is list else []
    test_paths = [path for path in paths if type(path) is str and is_test_path(path)]
    if not test_paths:
        reasons.append("no_test_path_in_merged_pr_diff")
    additions, deletions = row.get("additions"), row.get("deletions")
    if type(additions) is not int or type(deletions) is not int:
        reasons.append("changed_line_count_missing")
    elif additions + deletions > 250:
        reasons.append("diff_exceeds_250_changed_lines")
    changed_file_count = row.get("changed_file_count")
    if type(changed_file_count) is not int:
        reasons.append("changed_file_count_missing")
    elif changed_file_count > 8:
        reasons.append("diff_exceeds_8_files")
    title = (row.get("title") or "").lower()
    if any(marker in title for marker in SPECIALIZED_TITLE_MARKERS):
        reasons.append("title_flags_specialized_or_feature_scope_for_manual_review")
    if identity in prior:
        reasons.append("exact_identity_already_in_vare_candidate_or_exposure_audit")
    return {
        **row,
        "test_paths": test_paths,
        "coarse_screen": "provisional_for_manual_review" if not reasons else "excluded_from_shortlist",
        "screen_reasons": reasons,
        "qualification_status": "UNQUALIFIED",
        "split_assignment": "UNASSIGNED",
        "task_family_overlap": "NOT_ASSESSED",
    }


def screen(protocol: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    pool, raw_pool = load_pool(protocol)
    previous = prior_identities()
    rows = [screen_candidate(
        row,
        date_start=protocol["merged_date_start"],
        date_end=protocol["merged_date_end"],
        cutoff_proxy=protocol["model_training_cutoff_proxy"],
        prior=previous,
    ) for row in pool["candidates"]]
    provisional_count = sum(row["coarse_screen"] == "provisional_for_manual_review" for row in rows)
    raw = (json.dumps({
        "schema_version": 1,
        "screen_id": "mlsys-fresh-development-metadata-screen-v4",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_pool_sha256": sha256_bytes(raw_pool),
        "source_pool_gzip_sha256": sha256_file(SOURCE_POOL_PATH),
        "source_candidate_count": len(rows),
        "candidate_count": len(rows),
        "provisional_count": provisional_count,
        "excluded_count": len(rows) - provisional_count,
        "qualification_authorized": False,
        "model_or_agent_inference_authorized": False,
        "training_authorized": False,
        "state_correction": "GraphQL PullRequest.state is MERGED for merged PRs; v3 incorrectly expected the REST API value CLOSED.",
        "candidates": rows,
    }, indent=2, sort_keys=True) + "\n").encode("utf-8")
    screen_path = output_dir / "screen.json"
    screen_path.write_bytes(raw)
    manifest = {
        "schema_version": 1,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "screen_sha256": sha256_file(Path(__file__)),
        "source_pool_sha256": sha256_bytes(raw_pool),
        "source_pool_gzip_sha256": sha256_file(SOURCE_POOL_PATH),
        "queue_audit_sha256": sha256_file(QUEUE_AUDIT_PATH),
        "files": {
            "screen.json": {"bytes": len(raw), "sha256": sha256_bytes(raw)},
        },
    }
    (output_dir / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        "output": str(output_dir),
        "candidate_count": len(rows),
        "provisional_count": provisional_count,
        "screen_sha256": manifest["files"]["screen.json"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    result = screen(protocol, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
