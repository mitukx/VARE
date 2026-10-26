#!/usr/bin/env python3
"""Run the frozen six-control Development task qualification protocol."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _repository_path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_repository_path) not in sys.path:
        sys.path.insert(0, str(_repository_path))
PROTOCOL = ROOT / "benchmarks/corrected_template_agent_v3/qualification-protocol.json"
PROTOCOL_SHA256 = "72c5a2df36dd4fe73b030c79bb8ae41b3483b44d94968fcddbf0b68adca734d8"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def qualify(output: Path) -> dict[str, Any]:
    raw_protocol = PROTOCOL.read_bytes()
    if _sha(raw_protocol) != PROTOCOL_SHA256:
        raise ValueError("qualification protocol hash mismatch")
    protocol = json.loads(raw_protocol)
    from scripts.grade_mlsys_coding_v2_candidate_v2 import grade

    if output.exists():
        raise FileExistsError(f"append-only qualification output already exists: {output}")
    output.mkdir(parents=True)
    outcomes = []
    for task, controls in protocol["controls"].items():
        task_name = "label-smooth" if task == "label-smooth" else "ema-bias"
        for variant in ("baseline", "known_fix", "incorrect_mutant"):
            item = controls[variant]
            path = ROOT / item["path"]
            data = path.read_bytes()
            if _sha(data) != item["sha256"]:
                raise ValueError(f"qualification control hash mismatch: {path.relative_to(ROOT)}")
            result = grade(task_name, path)
            observed = result.get("success") is True
            isolated = result.get("isolated_grade", {}).get("decision", {}).get("success") is True
            passed = (
                result.get("admitted") is True
                and result.get("isolated_candidate_execution") is True
                and observed is item["expected_success"]
                and isolated is item["expected_success"]
            )
            record = {
                "task_id": controls["task_id"],
                "variant": variant,
                "source_path": item["path"],
                "source_sha256": item["sha256"],
                "expected_success": item["expected_success"],
                "observed_success": observed,
                "isolated_success": isolated,
                "qualification_gate_passed": passed,
                "grade": result,
            }
            record_path = output / task / f"{variant}.json"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            outcomes.append({key: record[key] for key in (
                "task_id", "variant", "expected_success", "observed_success",
                "isolated_success", "qualification_gate_passed",
            )})
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "isolation_protocol_sha256": protocol["isolation_protocol"]["sha256"],
        "task_pack_manifest_sha256": protocol["task_pack"]["manifest_sha256"],
        "qualification_passed": len(outcomes) == 6 and all(row["qualification_gate_passed"] for row in outcomes),
        "qualified_controls": len(outcomes),
        "outcomes": outcomes,
        "model_inference": "NOT_RUN_DURING_QUALIFICATION",
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "container_image": "python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258",
            "candidate_execution": "2 GiB RAM, 1 CPU, 64 pids, no network, read-only root, uid/gid 65534",
        },
        "limitations": [
            "Authored single-file utility contracts, not upstream repository issue fixes.",
            "Qualification outcomes are not model capability or held-out evidence.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = qualify(args.output)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["qualification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
