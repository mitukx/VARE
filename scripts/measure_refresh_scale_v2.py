#!/usr/bin/env python3
"""Frozen CPU measurement with committed implementation and harness fingerprints."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vare.durable as durable
from benchmarks.scheduler.fixtures import make_fixture
from vare.durable import Store, audit_durable
from vare.runner import atomic, encode, file_hash, write_manifest

PROTOCOL = ROOT / "protocols/cpu_refresh_scale_v2.json"
IMPLEMENTATION = ROOT / "src/vare/durable.py"


def run_replication(output: Path, phase: str, replication: int, count: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="vare-refresh-scale-") as temporary:
        parent = Path(temporary)
        root, first = make_fixture(parent, "valid")
        jobs = [replace(first, id="job-%03d" % index) for index in range(count)]
        store = Store.create(parent / "state.db", jobs, root)
        original_inputs = durable.inputs
        calls = 0

        def counted_inputs(job, trusted_root):
            nonlocal calls
            calls += 1
            return original_inputs(job, trusted_root)

        durable.inputs = counted_inputs
        durations = []
        try:
            started = time.perf_counter()
            for index in range(count):
                claim = store.claim("scale-worker")
                if claim is None or claim["job"].id != "job-%03d" % index:
                    raise ValueError("unexpected claim order or missing job")
                record = {"schema_version": 1, "job_id": claim["job"].id,
                          "status": "internal_error", "error_type": "SyntheticMeasurement"}
                transition_started = time.perf_counter()
                result = store.complete(claim, (record, b"", b"", None))
                durations.append(time.perf_counter() - transition_started)
                if result != {"committed": True, "duplicate": False}:
                    raise ValueError("terminal transition was not committed once")
            elapsed = time.perf_counter() - started
            completion_calls = calls

            calls = 0
            bundle = output / "bundles" / ("%s-rep-%d-jobs-%d" % (phase, replication, count))
            bundle.parent.mkdir(parents=True, exist_ok=True)
            store.export(bundle)
            export_calls = calls
            audited = audit_durable(bundle)
            if audited["attempts"] != count or audited["outcomes"] != {"internal_error": count}:
                raise ValueError("export audit did not preserve terminal records")
            return {
                "phase": phase,
                "replication": replication,
                "job_count": count,
                "completion_input_checks": completion_calls,
                "export_input_checks": export_calls,
                "completion_seconds": elapsed,
                "completion_latency_median_seconds": statistics.median(durations),
                "completion_latency_p95_seconds": sorted(durations)[max(0, int(.95 * len(durations)) - 1)],
                "audited_attempts": audited["attempts"],
                "audited_outcomes": audited["outcomes"],
            }
        finally:
            durable.inputs = original_inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "targeted"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = []
    for replication in range(protocol["workload"]["replications"]):
        for count in protocol["workload"]["job_counts"]:
            row = run_replication(output, args.phase, replication, count)
            expected = count * count if args.phase == "baseline" else count
            if row["completion_input_checks"] != expected or row["export_input_checks"] != count:
                raise ValueError("observed input-check count differs from the frozen protocol")
            rows.append(row)
    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True, timeout=10).stdout.strip()
    source_revision = subprocess.run(
        ["git", "-C", str(ROOT), "show", revision + ":src/vare/durable.py"],
        check=True, capture_output=True, timeout=10).stdout
    implementation_hash = file_hash(IMPLEMENTATION)
    if source_revision != IMPLEMENTATION.read_bytes():
        raise ValueError("measurement requires the implementation file to match committed HEAD")
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "phase": args.phase,
        "revision": revision,
        "implementation_sha256": implementation_hash,
        "measurement_script_sha256": file_hash(Path(__file__)),
        "protocol_sha256": file_hash(PROTOCOL),
        "python": sys.version,
        "platform": platform.platform(),
        "replications": rows,
    }
    atomic(output / "summary.json", encode(summary))
    write_manifest(output)
    print(json.dumps({"phase": args.phase, "output": str(output),
                      "replications": len(rows), "implementation_sha256": implementation_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
