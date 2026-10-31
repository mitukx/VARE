#!/usr/bin/env python3
"""Build a split-filtered metadata inventory for a frozen task-overlap review."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/mlsys_fresh_development_task_family_overlap_v5.json"
OUTPUT = "results/mlsys-fresh-development-task-family-overlap-v5"
ALLOWED_SPLITS = {"development", "dev", "train"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_repo_bytes(relative_path: str) -> bytes:
    path = ROOT / relative_path
    if path.is_file():
        return path.read_bytes()
    return subprocess.check_output(["git", "show", f"HEAD:{relative_path}"], cwd=ROOT)


def normalize_split(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_")
    aliases = {"dev": "development", "development": "development", "train": "train"}
    return aliases.get(normalized)


def identity_record(
    row: dict[str, Any], split: str, source: str,
) -> dict[str, Any]:
    """Extract identity fields only after the caller accepted the split."""
    source_pr = row.get("source_pr")
    source_match = None
    if isinstance(source_pr, str):
        parsed = urlparse(source_pr)
        source_match = re.search(r"/([^/]+/[^/]+)/pull/(\d+)$", parsed.path)
    repository_from_source = source_match.group(1) if source_match else None
    pull_from_source = int(source_match.group(2)) if source_match else None
    return {
        "split": split,
        "source": source,
        "repository": row.get("repository") or row.get("source_repository") or repository_from_source,
        "pull_request": row.get("pull_request") or row.get("source_pull_request") or pull_from_source,
        "task_id": row.get("task_id") or row.get("catalogued_task_id"),
        "task_family": row.get("task_family") or row.get("family") or row.get("catalogued_task_family"),
        "title": row.get("title") or row.get("task_title"),
        "changed_paths": row.get("changed_paths"),
        "changed_files_in_frozen_survey": row.get("changed_files_in_frozen_survey"),
        "exposure_classification": row.get("model_exposure_classification"),
    }


def queue_records(document: dict[str, Any], source: str) -> list[dict[str, Any]]:
    records = []
    for row in document.get("candidates", []):
        # Do not read identity fields until any Held-Out marker has been checked.
        split = normalize_split(row.get("historical_split"))
        qualification_status = row.get("qualification_status")
        if qualification_status == "historical_heldout_reserved":
            continue
        if split not in {"development", "train"}:
            continue
        records.append(identity_record(row, split, source))
    return records


def disposition_records(document: dict[str, Any], source: str) -> list[dict[str, Any]]:
    records = []
    for row in document.get("dispositions", []):
        # The split is the only field consulted before any task identity is read.
        split = normalize_split(row.get("original_split"))
        if split not in {"development", "train"}:
            continue
        records.append(identity_record(row, split, source))
    return records


def nested_split_records(
    value: Any, source: str, inherited_split: str | None = None,
) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [
            record
            for item in value
            for record in nested_split_records(item, source, inherited_split)
        ]
    if not isinstance(value, dict):
        return []

    raw_split = value.get("split") or value.get("assigned_split")
    split_assignment = value.get("split_assignment")
    if isinstance(split_assignment, str):
        raw_split = raw_split or split_assignment
    elif isinstance(split_assignment, dict):
        raw_split = raw_split or split_assignment.get("split")
    split = normalize_split(raw_split) or inherited_split
    if isinstance(raw_split, str) and raw_split.strip().lower().replace("-", "_") not in {
        "development", "dev", "train",
    }:
        # Stop at other split labels before walking into their task identities.
        return []

    records = []
    if split in {"development", "train"} and any(
        key in value for key in ("task_id", "source_pull_request", "pull_request")
    ):
        records.append(identity_record(value, split, source))
    for child in value.values():
        if isinstance(child, (dict, list)):
            records.extend(nested_split_records(child, source, split))
    return records


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in records:
        key = (
            row.get("split"), row.get("repository"), row.get("pull_request"),
            row.get("task_id") or row.get("task_family"),
        )
        if key not in unique:
            unique[key] = row
        else:
            prior = unique[key]
            for field, value in row.items():
                if prior.get(field) in (None, "", []):
                    prior[field] = value
            sources = prior.setdefault("duplicate_sources", [prior.get("source")])
            if row["source"] not in sources:
                sources.append(row["source"])
    return list(unique.values())


def verify_inputs(protocol: dict[str, Any]) -> dict[str, bytes]:
    captured = {}
    for relative, expected in protocol["frozen_inputs"].items():
        data = read_repo_bytes(relative)
        if sha256_bytes(data) != expected:
            raise ValueError(f"frozen overlap-review input differs: {relative}")
        captured[relative] = data
    return captured


def build_inventory() -> dict[str, Any]:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    if protocol.get("protocol_id") != "mlsys-fresh-development-task-family-overlap-v5":
        raise ValueError("unexpected overlap-review protocol")
    sidecar = PROTOCOL_PATH.with_suffix(PROTOCOL_PATH.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_bytes(protocol_bytes):
        raise ValueError("overlap-review protocol sidecar mismatch")
    captured = verify_inputs(protocol)

    selection = json.loads(captured[
        "results/mlsys-fresh-development-candidate-selection-v3/selection.json"
    ].decode("utf-8"))
    candidate = selection.get("selected_candidate")
    if not candidate or candidate.get("repository") != protocol["candidate"]["repository"] or candidate.get("number") != protocol["candidate"]["pull_request"]:
        raise ValueError("v3-selected candidate differs from the overlap protocol")

    records: list[dict[str, Any]] = []
    queue = json.loads(captured[
        "results/mlsys-candidate-manual-qualification-queue-exposure-v1.json"
    ].decode("utf-8"))
    records.extend(queue_records(queue, "candidate-manual-qualification-queue"))
    deep = json.loads(captured[
        "results/mlsys-task-deep-qualification-v1-exposure-append-v1.json"
    ].decode("utf-8"))
    records.extend(disposition_records(deep, "deep-qualification-split-dispositions"))

    for relative in (
        "results/mlsys-fresh-development-agent-diagnostic-v1/exposure-append.json",
        "results/mlsys-fresh-development-agent-diagnostic-v2/exposure-append.json",
        "results/mlsys-repo-eval-v1/run-19/exposure-append.json",
        "results/mlsys-fresh-vllm-64575-agent-attempt-v1/exposure-assignment.json",
        "results/mlsys-fresh-vllm-64575-source-qualification-v1/exposure-record.json",
    ):
        document = json.loads(captured[relative].decode("utf-8"))
        records.extend(nested_split_records(document, relative))

    unique = deduplicate(records)
    screen = json.loads(captured[
        "results/mlsys-fresh-development-candidate-discovery-v4/screen.json"
    ].decode("utf-8"))
    candidate_metadata = next(
        (
            row for row in screen["candidates"]
            if row.get("repository") == candidate["repository"]
            and row.get("number") == candidate["number"]
        ),
        None,
    )
    if candidate_metadata is None:
        raise ValueError("selected candidate is absent from the frozen screen")

    screen_by_identity = {
        (row.get("repository"), row.get("number")): row
        for row in screen["candidates"]
    }
    for prior in unique:
        metadata = screen_by_identity.get((prior.get("repository"), prior.get("pull_request")))
        if metadata is not None:
            prior["title"] = prior.get("title") or metadata.get("title")
            prior["changed_paths"] = prior.get("changed_paths") or metadata.get("changed_paths")

    same_repo = [
        row for row in unique
        if row.get("repository") == candidate["repository"]
    ]
    candidate_paths = set(candidate_metadata.get("changed_paths") or [])
    path_comparisons = []
    for row in same_repo:
        prior_paths = set(row.get("changed_paths") or [])
        overlap = sorted(candidate_paths & prior_paths)
        path_comparisons.append({
            "task_id": row.get("task_id"),
            "pull_request": row.get("pull_request"),
            "task_family": row.get("task_family"),
            "changed_path_overlap": overlap,
        })
    return {
        "schema_version": 1,
        "review_protocol_id": protocol["protocol_id"],
        "review_protocol_sha256": sha256_bytes(protocol_bytes),
        "selection_sha256": sha256_bytes(captured[
            "results/mlsys-fresh-development-candidate-selection-v3/selection.json"
        ]),
        "candidate": {
            key: candidate_metadata.get(key)
            for key in (
                "repository", "number", "title", "url", "merge_commit_sha",
                "merge_base_sha", "head_sha", "changed_paths", "test_paths",
                "changed_file_count", "additions", "deletions", "merged_at",
            )
        },
        "previous_development_and_train_records": unique,
        "previous_same_repository_records": same_repo,
        "same_repository_path_comparisons": path_comparisons,
        "direct_identity_overlap": any(
            row.get("repository") == candidate["repository"]
            and row.get("pull_request") == candidate["number"]
            for row in unique
        ),
        "source_or_test_content_accessed": False,
        "heldout_identity_fields_extracted": False,
        "heldout_assets_opened": False,
        "public_pretraining_exposure": "UNKNOWN",
        "inputs_sha256": {
            relative: sha256_bytes(data) for relative, data in captured.items()
        },
    }


def run(output_dir: Path) -> dict[str, str]:
    inventory = build_inventory()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    data = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
    path = output_dir / "metadata-inventory.json"
    path.write_bytes(data)
    manifest = {
        "schema_version": 1,
        "review_protocol_sha256": inventory["review_protocol_sha256"],
        "inventory_script_sha256": sha256_file(Path(__file__)),
        "files": {
            "metadata-inventory.json": {"bytes": len(data), "sha256": sha256_bytes(data)},
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (output_dir / "evidence-manifest.json").write_bytes(manifest_bytes)
    return {
        "prior_dev_train_record_count": str(len(inventory["previous_development_and_train_records"])),
        "same_repository_record_count": str(len(inventory["previous_same_repository_records"])),
        "direct_identity_overlap": str(inventory["direct_identity_overlap"]).lower(),
        "inventory_sha256": sha256_bytes(data),
        "manifest_sha256": sha256_bytes(manifest_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
