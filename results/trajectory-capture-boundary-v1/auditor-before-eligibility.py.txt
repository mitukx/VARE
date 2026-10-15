"""Reconstruct retained capture comparison without importing its runner or parser."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
import xml.etree.ElementTree as ET


def require(value, detail):
    if not value:
        raise ValueError(detail)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def trace(path, *, policy_id, run_id):
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), "unterminated trace")
    rows = [json.loads(line) for line in raw.splitlines()]
    head, seen, calls, results = "0" * 64, {}, set(), set()
    for index, row in enumerate(rows):
        require(set(row) == {"seq", "event", "payload", "prev_sha256", "sha256"}, "ledger schema")
        body = {key: value for key, value in row.items() if key != "sha256"}
        digest = sha(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        require(row["seq"] == index and row["prev_sha256"] == head and row["sha256"] == digest,
                "ledger integrity")
        head = digest
        if index in (0, len(rows) - 1):
            continue
        require(row["event"] == "trajectory_event", "unexpected inner record")
        event = row["payload"]
        identity, parent = event["event_id"], event["parent_event_id"]
        require(identity not in seen and (parent is None or parent in seen), "event provenance")
        if event["kind"] == "tool_call":
            require(event["authorization"] == "allow", "unexpected denied tool")
            calls.add(identity)
        elif event["kind"] == "tool_result":
            require(parent in calls and parent not in results, "unmatched/repeated result")
            require(event["action"] == "success" and event["authorization"] == "not_applicable", "tool failed")
            results.add(parent)
        else:
            raise ValueError("unexpected event in retained command/cost trace")
        seen[identity] = event
    require(rows[0]["event"] == "trajectory_start" and rows[0]["payload"] == {
        "schema": "vare.trajectory-safety.v1", "run_id": run_id, "policy_id": policy_id}, "start identity")
    require(rows[-1]["event"] == "trajectory_end" and rows[-1]["payload"] == {
        "expected_event_count": len(rows) - 2}, "capture count")
    return {"head": head, "event_count": len(rows) - 2, "calls": len(calls), "results": len(results)}


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    assets = manifest["artifact_sha256"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
    require(set(assets) == actual, "artifact inventory differs")
    for name, expected in assets.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), "unsafe artifact path")
        require(sha(path.read_bytes()) == expected, f"artifact changed: {name}")
    records = json.loads((root / "test-results.json").read_text())
    control_ids = {"https", "https-443", "http-80", "uppercase", "other-host"}
    for label in ["baseline", "candidate"]:
        cases = ET.parse(root / f"{label}-pytest.xml").findall(".//testcase")
        verdicts = {c.attrib["name"]: "error" if c.find("error") is not None else
                    "fail" if c.find("failure") is not None else
                    "skip" if c.find("skipped") is not None else "pass" for c in cases}
        require(verdicts == records[label]["cases"] and len(verdicts) == 16, "test records differ")
        counts = Counter(verdicts.values())
        require(counts == Counter({"pass": 5, "fail": 11}) if label == "baseline" else
                counts == Counter({"pass": 16}), "unexpected verdict counts")
        for name, verdict in verdicts.items():
            is_control = any(name.endswith(f"[{item}]") for item in control_ids)
            require(verdict == ("pass" if label == "candidate" or is_control else "fail"), name)

    rows = json.loads((root / "measurements.json").read_text())
    require(len(rows) == 10 and {(r["label"], r["repetition"]) for r in rows} == {
        (label, i) for label in ["baseline", "candidate"] for i in range(5)}, "measurement coverage")
    reconstructed = {}
    for row in rows:
        path = root / f"cost-traces/{row['label']}-{row['repetition']}.jsonl"
        checked = trace(path, policy_id="candidate", run_id="cost-v1")
        require(sha(path.read_bytes()) == row["trace_sha256"], "timing trace identity")
        require(checked == {"head": row["ledger_head_sha256"], "event_count": 64, "calls": 32, "results": 32},
                "timing workload differs")
        require(row["source_sha256"] == records[row["label"]]["source_sha256"], "timed source differs")
        require(row["calls"] == 32 and row["event_count"] == 64 and row["audit_status"] == "pass", "timing status")
        require(row["dispatch_wall_s"] > 0 and row["audit_wall_s"] > 0, "invalid timing")
        require(row["throughput_tools_s"] == 32 / row["dispatch_wall_s"], "throughput differs")
    for label in ["baseline", "candidate"]:
        reconstructed[label] = {f"median_{key}": median(r[key] for r in rows if r["label"] == label)
            for key in ["dispatch_wall_s", "audit_wall_s", "throughput_tools_s",
                        "python_peak_bytes", "process_peak_rss_bytes"]}
    require(reconstructed == json.loads((root / "summary.json").read_text()), "summary differs")

    # Existing authentication contract is reused; trace replay above is a
    # separate implementation. These public fixture keys are not secrets.
    from vare.attestation import EvidenceTrustRoot
    from vare.types import EvaluationReport
    trust = EvidenceTrustRoot(key=b"capture-fixture-public-key-00000000", run_id="trajectory-capture-boundary-v1",
        protocol_sha256=manifest["source_sha256"]["protocols/trajectory_capture_boundary_v1.lock.json"],
        evaluator_sha256=manifest["source_sha256"]["tests/test_trajectory_capture_boundary.py"])
    for label in ["baseline", "candidate"]:
        directory = root / label
        evidence = json.loads((directory / "command-promotion.json").read_text())
        require(evidence["command_result_verified"] and evidence["final_promotion_accepted"], "command control failed")
        require(evidence["early_closed"] == evidence["early_audit_passed"] == (label == "baseline"), "closure result")
        require(evidence["command_wrapper_error"] == ("TrajectoryEvidenceError" if label == "baseline" else None),
                "unexpected wrapper outcome")
        require((directory / "command-result.txt").read_text() == "independently checked command result", "wrong child output")
        require(set(evidence["raw_artifact_sha256"]) == {
            "candidate-trace.jsonl", "incumbent-trace.jsonl", "command-result.txt"}, "raw command inventory")
        for name, digest in evidence["raw_artifact_sha256"].items():
            require(sha((directory / name).read_bytes()) == digest, "raw command identity")
        for policy_id in ["incumbent", "candidate"]:
            checked = trace(directory / f"{policy_id}-trace.jsonl", policy_id=policy_id, run_id=trust.run_id)
            report = EvaluationReport(**evidence[f"{policy_id}_report"])
            require(trust.verifies(report), "fixture receipt invalid")
            require((report.policy_id, report.primary, report.n) ==
                    (policy_id, .5 if policy_id == "incumbent" else 1.0, 1), "command score differs")
            metadata = report.metadata["trajectory_safety"]
            require(metadata["ledger_head_sha256"] == checked["head"] and
                    metadata["event_count"] == metadata["expected_event_count"] == checked["event_count"], "report trace differs")
            expected = (0, 0, 0) if policy_id == "incumbent" else (1, 1, 0) if label == "baseline" else (2, 1, 1)
            require((checked["event_count"], checked["calls"], checked["results"]) == expected, "command capture differs")
    return {"artifacts": len(assets), "regression_cases_per_arm": 16, "timing_traces": 10,
            "command_traces": 4, "signed_reports": 4,
            "scope": "same-host independent trace/summary reconstruction; public fixture receipts, not external authentication"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.bundle.resolve()), indent=2))
