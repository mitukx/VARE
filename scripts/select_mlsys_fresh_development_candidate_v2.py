#!/usr/bin/env python3
"""Select the next frozen metadata candidate without substituting within v1."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_selection_v2.json"
ADDENDUM_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_selection_v2_addendum_1.json"
V1_SELECTOR_PATH = ROOT / "scripts/select_mlsys_fresh_development_candidate_v1.py"
V1_PROTOCOL_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_selection_v1.json"
V1_ADDENDUM_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_selection_v1_addendum_1.json"
V1_ADDENDUM_2_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_selection_v1_addendum_2.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_v1_selector() -> Any:
    spec = importlib.util.spec_from_file_location("mlsys_selection_v1", V1_SELECTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v1 metadata selector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exclude_identities(
    rows: list[dict[str, Any]], excluded: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if (row.get("repository"), row.get("number")) not in excluded
    ]


def frozen_input_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    return protocol["frozen_inputs"]


def frozen_screen_path(protocol: dict[str, Any]) -> str:
    return protocol["source_scan"]["screen_path"]


def verify_inputs(protocol: dict[str, Any]) -> None:
    addendum = json.loads(ADDENDUM_PATH.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(PROTOCOL_PATH)
    if addendum.get("protocol_id") != "mlsys-fresh-development-candidate-selection-v2-addendum-1":
        raise ValueError("unexpected v2 preselection addendum")
    if addendum.get("parent_protocol_sha256") != protocol_hash:
        raise ValueError("v2 addendum parent protocol hash mismatch")
    corrections = addendum["corrected_input_hashes"]
    for relative_path, expected_hash in frozen_input_hashes(protocol).items():
        path = ROOT / relative_path
        corrected = corrections.get(relative_path)
        if corrected is not None and corrected.get("original_sha256") != expected_hash:
            raise ValueError(f"v2 addendum does not match original input: {relative_path}")
        required_hash = (
            corrected["corrected_sha256"] if corrected is not None else expected_hash
        )
        if not path.is_file() or sha256_file(path) != required_hash:
            raise ValueError(f"frozen v2 selection input differs: {relative_path}")
    v1_protocol = json.loads(V1_PROTOCOL_PATH.read_text(encoding="utf-8"))
    v1_manifest_path = ROOT / "results/mlsys-fresh-development-candidate-selection-v1/evidence-manifest.json"
    v1_manifest = json.loads(v1_manifest_path.read_text(encoding="utf-8"))
    v1_addendum = json.loads(V1_ADDENDUM_PATH.read_text(encoding="utf-8"))
    v1_addendum_2 = json.loads(V1_ADDENDUM_2_PATH.read_text(encoding="utf-8"))
    if sha256_file(V1_PROTOCOL_PATH) != v1_manifest.get("protocol_sha256"):
        raise ValueError("v1 protocol hash differs from its result manifest")
    if sha256_file(V1_PROTOCOL_PATH) != v1_addendum.get("parent_protocol_sha256"):
        raise ValueError("v1 protocol hash differs from its first addendum")
    if sha256_file(V1_ADDENDUM_PATH) != v1_addendum_2.get("parent_addendum_sha256"):
        raise ValueError("v1 first addendum hash differs from its chain")
    if sha256_file(V1_PROTOCOL_PATH) != v1_addendum_2.get("parent_protocol_sha256"):
        raise ValueError("v1 protocol hash differs from its second addendum")
    if sha256_file(V1_SELECTOR_PATH) != v1_addendum_2.get("corrected_selector_sha256"):
        raise ValueError("v1 selector differs from its frozen correction")
    if v1_protocol.get("protocol_id") != "mlsys-fresh-development-candidate-selection-v1":
        raise ValueError("unexpected v1 selection protocol")


def run(output_dir: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "mlsys-fresh-development-candidate-selection-v2":
        raise ValueError("unexpected v2 selection protocol")
    verify_inputs(protocol)
    screen_path = ROOT / frozen_screen_path(protocol)
    screen_bytes = screen_path.read_bytes()
    screen = json.loads(screen_bytes.decode("utf-8"))
    v1_protocol = json.loads(V1_PROTOCOL_PATH.read_text(encoding="utf-8"))
    v1_selector = load_v1_selector()
    eligible = v1_selector.eligible_rows(screen, v1_protocol)
    excluded_rows = protocol["excluded_identities"]
    excluded = {(row["repository"], row["number"]) for row in excluded_rows}
    remaining = exclude_identities(eligible, excluded)
    selected = remaining[0] if remaining else None

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    selection = {
        "schema_version": 1,
        "selection_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "protocol_addendum_sha256": sha256_file(ADDENDUM_PATH),
        "selector_sha256": sha256_file(Path(__file__)),
        "source_screen_sha256": sha256_bytes(screen_bytes),
        "frozen_inputs": frozen_input_hashes(protocol),
        "eligible_candidate_count_before_exclusions": len(eligible),
        "excluded_identities": excluded_rows,
        "eligible_candidate_count_after_exclusions": len(remaining),
        "selected_count": int(selected is not None),
        "selection_rule": "Select the minimum v1 SHA-256 rank from the frozen eligible metadata rows after the frozen prior-selection and overlap exclusions.",
        "source_or_test_content_accessed": False,
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
        "protocol_addendum_sha256": sha256_file(ADDENDUM_PATH),
        "selector_sha256": sha256_file(Path(__file__)),
        "files": {
            "selection.json": {
                "bytes": len(selection_bytes),
                "sha256": sha256_bytes(selection_bytes),
            },
        },
    }
    (output_dir / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        "eligible_candidate_count_before_exclusions": len(eligible),
        "eligible_candidate_count_after_exclusions": len(remaining),
        "selected_identity": (
            f"{selected['repository']}#{selected['number']}" if selected else None
        ),
        "selection_sha256": manifest["files"]["selection.json"]["sha256"],
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
