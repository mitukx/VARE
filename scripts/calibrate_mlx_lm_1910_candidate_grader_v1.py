#!/usr/bin/env python3
"""Calibrate the candidate grader against baseline, fix, and wrong mutations."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any

SPEC = importlib.util.spec_from_file_location(
    "mlx1910_candidate_grader", Path(__file__).with_name("grade_mlx_lm_1910_candidate_v1.py")
)
assert SPEC and SPEC.loader
GRADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRADER)

BASE = GRADER.CAL.BASE
FIXED = GRADER.CAL.FIXED
TARGET = GRADER.CAL.TARGET
EXPECTED_HASHES = {
    "baseline": "2d212e4e6f175178bd80b7db9963ccad613e29419f976e6d6b3272417f953722",
    "known_fix": "d1029aae22ed475e9d7c09d29c37e74dca8df52b0fb046f000f2e3f9b83a6583",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite calibration: {args.output}")
    args.output.mkdir(parents=True)
    repo = args.git_source.resolve(strict=True)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vare-mlx1910-grader-calibration-") as temp:
        temp_root = Path(temp)
        baseline = GRADER.CAL.run(["git", "-C", str(repo), "show", f"{BASE}:{TARGET}"])
        fixed = GRADER.CAL.run(["git", "-C", str(repo), "show", f"{FIXED}:{TARGET}"])
        if baseline.returncode or fixed.returncode:
            raise RuntimeError("pinned task source could not be read")
        source_by_name = {"baseline": baseline.stdout, "known_fix": fixed.stdout}
        for name, body in source_by_name.items():
            if hashlib.sha256(body).hexdigest() != EXPECTED_HASHES[name]:
                raise RuntimeError(f"pinned {name} source hash mismatch")
        anchor = b"        try:\n            value = json.loads(param_value)\n        except json.JSONDecodeError:\n            return param_value\n        return value if isinstance(value, (dict, list)) else param_value\n"
        mutants = {
            "mutant_restore_default_string": (anchor, b'        param_type = "string"\n'),
            "mutant_parse_any_json_scalar": (anchor, b"        return json.loads(param_value)\n"),
            "mutant_object_only": (
                b"        return value if isinstance(value, (dict, list)) else param_value\n",
                b"        return value if isinstance(value, dict) else param_value\n",
            ),
        }
        for label, (old, new) in mutants.items():
            if source_by_name["known_fix"].count(old) != 1:
                raise RuntimeError(f"{label}: mutation anchor was not unique")
            source_by_name[label] = source_by_name["known_fix"].replace(old, new, 1)
        for label, body in source_by_name.items():
            source_file = temp_root / f"{label}.py"
            source_file.write_bytes(body)
            grade_dir = args.output / label
            result = GRADER.grade_source(repo, source_file, grade_dir)
            records.append({"arm": label, "source_sha256": hashlib.sha256(body).hexdigest(),
                            "grader_status": result["status"],
                            "upstream_test_returncode": result["candidate"]["upstream_test"]["returncode"],
                            "oracle_passed": result["candidate"]["oracle"]["passed"]})
    expected = {"baseline": "FAIL", "known_fix": "PASS",
                "mutant_restore_default_string": "FAIL",
                "mutant_parse_any_json_scalar": "FAIL", "mutant_object_only": "FAIL"}
    passed = all(record["grader_status"] == expected[record["arm"]] for record in records)
    summary = {"schema_version": 1,
               "protocol_id": "mlsys-repo-task-mlx-lm-1910-candidate-grader-v1",
               "status": "CALIBRATION_PASS_DEVELOPMENT_ONLY" if passed else "NO_GO",
               "records": records, "expected": expected,
               "claim_limit": "Single Development task; this calibration does not validate held-out generalization."}
    (args.output / "calibration.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": summary["status"], "arms": len(records)}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
