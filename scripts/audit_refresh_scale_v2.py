#!/usr/bin/env python3
"""Audit frozen freshness-scaling records and replay every exported bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vare.durable import audit_durable
from vare.runner import file_hash, strict_json

PROTOCOL = ROOT / "protocols/cpu_refresh_scale_v2.json"
MEASUREMENT = ROOT / "scripts/measure_refresh_scale_v2.py"
IMPLEMENTATION_PATH = "src/vare/durable.py"


def committed_source_hash(revision: str) -> str:
    content = subprocess.run(["git", "-C", str(ROOT), "show", revision + ":" + IMPLEMENTATION_PATH],
                             check=True, capture_output=True, timeout=10).stdout
    import hashlib
    return hashlib.sha256(content).hexdigest()


def audit(parent: Path) -> dict:
    protocol = strict_json(PROTOCOL.read_bytes())
    expected_ref = protocol["comparison"]["baseline_implementation_revision"]
    phases = {}
    for phase in ("baseline", "targeted"):
        directory = parent / phase
        manifest = strict_json((directory / "manifest.json").read_bytes())
        actual_files = {
            str(path.relative_to(directory)): file_hash(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != directory / "manifest.json"
        }
        if manifest.get("files") != actual_files:
            raise ValueError("phase manifest mismatch")
        summary = strict_json((directory / "summary.json").read_bytes())
        if summary["phase"] != phase or summary["protocol_id"] != protocol["protocol_id"]:
            raise ValueError("phase or protocol identity mismatch")
        if summary["protocol_sha256"] != file_hash(PROTOCOL):
            raise ValueError("protocol hash mismatch")
        if summary["measurement_script_sha256"] != file_hash(MEASUREMENT):
            raise ValueError("measurement script hash mismatch")
        if summary["implementation_sha256"] != committed_source_hash(summary["revision"]):
            raise ValueError("measured implementation differs from its recorded commit")
        expected_rows = protocol["workload"]["replications"] * len(protocol["workload"]["job_counts"])
        rows = summary["replications"]
        if len(rows) != expected_rows:
            raise ValueError("incomplete replication set")
        count_function = (lambda n: n * n) if phase == "baseline" else (lambda n: n)
        seen = set()
        for row in rows:
            count = row["job_count"]
            key = (row["replication"], count)
            if key in seen or count not in protocol["workload"]["job_counts"]:
                raise ValueError("unexpected or duplicate workload row")
            seen.add(key)
            if row["completion_input_checks"] != count_function(count):
                raise ValueError("completion input-check count mismatch")
            if row["export_input_checks"] != count:
                raise ValueError("export failed to check every job")
            if row["audited_attempts"] != count or row["audited_outcomes"] != {"internal_error": count}:
                raise ValueError("terminal evidence mismatch")
            bundle = directory / "bundles" / ("%s-rep-%d-jobs-%d" % (phase, row["replication"], count))
            verified = audit_durable(bundle)
            if verified["attempts"] != count or verified["outcomes"] != {"internal_error": count}:
                raise ValueError("offline bundle replay mismatch")
        phases[phase] = summary

    baseline_hash = committed_source_hash(expected_ref)
    baseline, targeted = phases["baseline"], phases["targeted"]
    if baseline["implementation_sha256"] != baseline_hash:
        raise ValueError("baseline code does not match the frozen implementation revision")
    if targeted["implementation_sha256"] == baseline_hash:
        raise ValueError("targeted phase did not change the implementation")
    if baseline["measurement_script_sha256"] != targeted["measurement_script_sha256"]:
        raise ValueError("comparison used different measurement scripts")

    comparisons = []
    for count in protocol["workload"]["job_counts"]:
        before = [r for r in baseline["replications"] if r["job_count"] == count]
        after = [r for r in targeted["replications"] if r["job_count"] == count]
        med_before = sorted(r["completion_seconds"] for r in before)[1]
        med_after = sorted(r["completion_seconds"] for r in after)[1]
        comparisons.append({
            "job_count": count,
            "baseline_completion_checks": sorted(r["completion_input_checks"] for r in before)[1],
            "targeted_completion_checks": sorted(r["completion_input_checks"] for r in after)[1],
            "check_reduction_factor": count,
            "baseline_completion_seconds_median": med_before,
            "targeted_completion_seconds_median": med_after,
            "descriptive_time_ratio": med_before / med_after if med_after else None,
        })
    return {"protocol_id": protocol["protocol_id"], "verified": True,
            "baseline_revision": baseline["revision"], "targeted_revision": targeted["revision"],
            "comparisons": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.bundle_root.resolve()), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
