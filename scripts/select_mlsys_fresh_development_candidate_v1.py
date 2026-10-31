#!/usr/bin/env python3
"""Select one candidate mechanically from a frozen, metadata-only PR screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_selection_v1.json"
ADDENDUM_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_selection_v1_addendum_1.json"
ADDENDUM_2_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_selection_v1_addendum_2.json"
SCREEN_PATH = ROOT / "results/mlsys-fresh-development-candidate-discovery-v4/screen.json"
POOL_PATH = ROOT / "results/mlsys-fresh-development-candidate-discovery-v4/candidate-pool.json"
SOURCE_MANIFEST_PATH = ROOT / "results/mlsys-fresh-development-candidate-discovery-v4/evidence-manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_test_path(path: str) -> bool:
    lower = path.lower()
    return any(marker in lower for marker in ("test", "spec", "fixture"))


def load_protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    addendum = json.loads(ADDENDUM_PATH.read_text(encoding="utf-8"))
    addendum_2 = json.loads(ADDENDUM_2_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "mlsys-fresh-development-candidate-selection-v1":
        raise ValueError("unexpected candidate-selection protocol ID")
    if sha256_file(PROTOCOL_PATH) != addendum.get("parent_protocol_sha256"):
        raise ValueError("frozen selection protocol differs from its preselection addendum")
    if sha256_file(ADDENDUM_PATH) != addendum_2.get("parent_addendum_sha256"):
        raise ValueError("first selection addendum differs from addendum 2")
    if addendum_2.get("parent_protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise ValueError("frozen selection protocol differs from addendum 2")
    if sha256_file(Path(__file__)) != addendum_2.get("corrected_selector_sha256"):
        raise ValueError("selector source differs from selection addendum 2")
    expected = protocol["frozen_inputs"]
    for path, key in (
        (SCREEN_PATH, "screen_sha256"),
        (POOL_PATH, "candidate_pool_sha256"),
        (SOURCE_MANIFEST_PATH, "source_manifest_sha256"),
    ):
        if sha256_file(path) != expected[key]:
            raise ValueError(f"frozen selection input differs: {path.relative_to(ROOT)}")
    return protocol, addendum, addendum_2


def eligible_rows(screen: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rules = protocol["selection_rules"]
    eligible: list[dict[str, Any]] = []
    for row in screen.get("candidates", []):
        if row.get("coarse_screen") != "provisional_for_manual_review":
            continue
        if (
            row.get("qualification_status") != "UNQUALIFIED"
            or row.get("split_assignment") != "UNASSIGNED"
            or row.get("task_family_overlap") != "NOT_ASSESSED"
        ):
            raise ValueError("metadata screen promoted a candidate beyond its frozen status")
        repo, number = row.get("repository"), row.get("number")
        if type(repo) is not str or type(number) is not int:
            raise ValueError("provisional candidate has an invalid identity")
        owner = repo.split("/", maxsplit=1)[0]
        if owner in rules["exclude_repository_owners"]:
            continue
        paths = row.get("changed_paths")
        if type(paths) is not list or any(type(path) is not str for path in paths):
            raise ValueError("candidate changed-path metadata is malformed")
        test_paths = [path for path in paths if is_test_path(path)]
        production_paths = [path for path in paths if not is_test_path(path)]
        if not test_paths or not production_paths:
            continue
        if not any(path in paths for path in row.get("test_paths", [])):
            raise ValueError("frozen test-path summary differs from changed paths")
        lowered_paths = [path.lower() for path in paths]
        if any(
            marker in path
            for path in lowered_paths
            for marker in rules["excluded_path_markers"]
        ):
            continue
        title = (row.get("title") or "").lower()
        if any(marker in title for marker in rules["excluded_title_markers"]):
            continue
        if row.get("changed_file_count", 10**9) > rules["maximum_changed_files"]:
            continue
        additions, deletions = row.get("additions"), row.get("deletions")
        if type(additions) is not int or type(deletions) is not int:
            continue
        if additions + deletions > rules["maximum_changed_lines"]:
            continue
        rank = hashlib.sha256(
            f"{rules['rank_domain']}|{repo}#{number}".encode("utf-8"),
        ).hexdigest()
        eligible.append({**row, "selection_rank_sha256": rank})
    return sorted(eligible, key=lambda row: (row["selection_rank_sha256"], row["repository"], row["number"]))


def run(protocol: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    screen_bytes = SCREEN_PATH.read_bytes()
    screen = json.loads(screen_bytes.decode("utf-8"))
    rows = eligible_rows(screen, protocol)
    selected = rows[0] if rows else None
    output_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "selection_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "protocol_addendum_sha256": sha256_file(ADDENDUM_PATH),
        "protocol_addendum_2_sha256": sha256_file(ADDENDUM_2_PATH),
        "source_screen_sha256": sha256_bytes(screen_bytes),
        "source_candidate_pool_sha256": sha256_file(POOL_PATH),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "eligible_candidate_count": len(rows),
        "selected_count": int(selected is not None),
        "selection_rule": "Select exactly the minimum SHA-256 rank; no substitution if later overlap review or source qualification fails.",
        "task_qualification_performed": False,
        "source_or_test_content_accessed": False,
        "development_split_assignment_performed": False,
        "model_or_agent_inference_authorized": False,
        "training_authorized": False,
        "heldout_access_authorized": False,
        "selected_candidate": selected,
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_path = output_dir / "selection.json"
    output_path.write_bytes(raw)
    manifest = {
        "schema_version": 1,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "protocol_addendum_sha256": sha256_file(ADDENDUM_PATH),
        "protocol_addendum_2_sha256": sha256_file(ADDENDUM_2_PATH),
        "selector_sha256": sha256_file(Path(__file__)),
        "files": {"selection.json": {"bytes": len(raw), "sha256": sha256_bytes(raw)}},
    }
    (output_dir / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        "output": str(output_dir),
        "eligible_candidate_count": len(rows),
        "selected_identity": (
            f"{selected['repository']}#{selected['number']}" if selected else None
        ),
        "selection_sha256": manifest["files"]["selection.json"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, _addendum, _addendum_2 = load_protocol()
    result = run(protocol, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["eligible_candidate_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
