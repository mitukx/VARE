#!/usr/bin/env python3
"""Select one remaining fresh Development candidate from a frozen metadata screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_selection_v3.json"
SCREEN_PATH = ROOT / "results/mlsys-fresh-development-candidate-discovery-v4/screen.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def candidate_rank(row: dict[str, Any], rank_domain: str) -> str:
    identity = f"{row['repository']}#{row['number']}"
    return sha256_bytes(f"{rank_domain}|{identity}".encode("utf-8"))


def eligible_rows(screen: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = screen.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("frozen screen candidates must be a list")
    eligible: list[dict[str, Any]] = []
    for row in candidates:
        paths = row.get("changed_paths")
        tests = row.get("test_paths")
        if (
            row.get("coarse_screen") == "provisional_for_manual_review"
            and row.get("screen_reasons") == []
            and row.get("changed_paths_complete") is True
            and isinstance(paths, list)
            and len(paths) > 0
            and isinstance(tests, list)
            and any(path in paths for path in tests)
            and row.get("state") == "MERGED"
            and row.get("qualification_status") == "UNQUALIFIED"
            and row.get("split_assignment") == "UNASSIGNED"
            and row.get("task_family_overlap") == "NOT_ASSESSED"
            and isinstance(row.get("repository"), str)
            and isinstance(row.get("number"), int)
        ):
            eligible.append(row)
    return eligible


def rank_and_exclude(
    rows: list[dict[str, Any]],
    excluded: set[tuple[str, int]],
    rank_domain: str,
) -> list[dict[str, Any]]:
    ranked = []
    for row in rows:
        identity = (row["repository"], row["number"])
        if identity in excluded:
            continue
        ranked.append({**row, "selection_rank_sha256": candidate_rank(row, rank_domain)})
    return sorted(ranked, key=lambda row: (row["selection_rank_sha256"], row["repository"], row["number"]))


def verify_frozen_inputs(protocol: dict[str, Any]) -> None:
    for relative, expected in protocol["frozen_inputs"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen v3 selection input differs: {relative}")
    for path_key, hash_key in (
        ("selector_path", "selector_sha256"),
        ("selector_test_path", "selector_test_sha256"),
    ):
        path = ROOT / protocol[path_key]
        if not path.is_file() or sha256_file(path) != protocol[hash_key]:
            raise ValueError(f"frozen v3 selector artifact differs: {protocol[path_key]}")
    if sha256_file(SCREEN_PATH) != protocol["frozen_inputs"][str(SCREEN_PATH.relative_to(ROOT))]:
        raise ValueError("screen hash does not match the frozen protocol")


def run(output_dir: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "mlsys-fresh-development-candidate-selection-v3":
        raise ValueError("unexpected v3 selection protocol")
    verify_frozen_inputs(protocol)
    screen_bytes = SCREEN_PATH.read_bytes()
    screen = json.loads(screen_bytes.decode("utf-8"))
    candidates = eligible_rows(screen)
    excluded_rows = protocol["selection"]["excluded_identities"]
    excluded = {(row["repository"], row["number"]) for row in excluded_rows}
    ranked = rank_and_exclude(candidates, excluded, protocol["selection"]["rank_domain"])
    selected = ranked[0] if ranked else None

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    selection = {
        "schema_version": 1,
        "selection_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "selector_sha256": sha256_file(Path(__file__)),
        "source_screen_sha256": sha256_bytes(screen_bytes),
        "frozen_inputs": protocol["frozen_inputs"],
        "eligible_candidate_count_before_exclusions": len(candidates),
        "excluded_identities": excluded_rows,
        "eligible_candidate_count_after_exclusions": len(ranked),
        "ranked_candidate_identities": [
            {
                "repository": row["repository"],
                "number": row["number"],
                "selection_rank_sha256": row["selection_rank_sha256"],
            }
            for row in ranked
        ],
        "selected_count": int(selected is not None),
        "source_or_test_content_accessed": False,
        "task_family_overlap_review_performed": False,
        "task_qualification_performed": False,
        "development_split_assignment_performed": False,
        "model_or_agent_inference_authorized": False,
        "training_authorized": False,
        "heldout_access_authorized": False,
        "selected_candidate": selected,
    }
    selection_bytes = (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode()
    selection_path = output_dir / "selection.json"
    selection_path.write_bytes(selection_bytes)
    manifest = {
        "schema_version": 1,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "selector_sha256": sha256_file(Path(__file__)),
        "files": {
            "selection.json": {
                "bytes": len(selection_bytes),
                "sha256": sha256_bytes(selection_bytes),
            },
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (output_dir / "evidence-manifest.json").write_bytes(manifest_bytes)
    return {
        "eligible_candidate_count_before_exclusions": len(candidates),
        "eligible_candidate_count_after_exclusions": len(ranked),
        "selected_identity": (
            f"{selected['repository']}#{selected['number']}" if selected else None
        ),
        "selection_sha256": sha256_bytes(selection_bytes),
        "manifest_sha256": sha256_bytes(manifest_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["eligible_candidate_count_after_exclusions"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
