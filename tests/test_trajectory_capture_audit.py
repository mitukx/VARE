"""Optimized Python must not disable capture artifact checks."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("fault", ["missing", "modified", "extra"])
def test_capture_auditor_rejects_tampered_inventory(tmp_path, optimized, fault):
    (tmp_path / "manifest.json").write_text(json.dumps({"artifact_sha256": {"trace.jsonl": "0" * 64}}))
    if fault != "missing":
        (tmp_path / "trace.jsonl").write_text("{}\n")
    if fault == "extra":
        (tmp_path / "extra").write_text("unexpected")
    script = Path(__file__).resolve().parents[1] / "scripts/audit_trajectory_capture_boundary_v1.py"
    run = subprocess.run([sys.executable, *(["-O"] if optimized else []), str(script), str(tmp_path)],
                         capture_output=True, text=True, timeout=5)
    assert run.returncode != 0 and "ValueError" in run.stderr and not run.stdout


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("fault", ["status", "policy", "run", "schema", "coverage", "violations", "boolean-counts", "slice-coverage"])
def test_auditor_rejects_reattested_but_ineligible_trajectory(tmp_path, optimized, fault):
    from vare.attestation import EvidenceTrustRoot
    from vare.config import PromotionConfig
    from vare.promotion import PromotionGate
    from vare.types import EvaluationReport

    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "bundle"
    shutil.copytree(repo / "results/trajectory-capture-boundary-v1/comparison-v2", root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    trust = EvidenceTrustRoot(key=b"capture-fixture-public-key-00000000", run_id="trajectory-capture-boundary-v1",
        protocol_sha256=manifest["source_sha256"]["protocols/trajectory_capture_boundary_v1.lock.json"],
        evaluator_sha256=manifest["source_sha256"]["tests/test_trajectory_capture_boundary.py"])
    file = root / "candidate/command-promotion.json"
    evidence = json.loads(file.read_text())
    expected_policy = evidence["candidate_report"]["metadata"]["trajectory_safety"]["policy_sha256"]
    key = "incumbent_report" if fault == "boolean-counts" else "candidate_report"
    changed = evidence[key]["metadata"]["trajectory_safety"]
    updates = {
        "status": {"status": "violation"}, "policy": {"policy_sha256": "f" * 64},
        "run": {"run_id": "different-capture-run"}, "schema": {"schema": "unknown.v999"},
        "coverage": {"capture_complete": False}, "violations": {"violations": ["unauthorized_action"]},
        "boolean-counts": {"event_count": False, "expected_event_count": False},
        "slice-coverage": {},
    }
    changed.update(updates[fault])
    if fault == "slice-coverage":
        evidence[key]["slices"] = {"unexpected": 1.0}
    # Re-sign with the public fixture key and update the outer artifact hash.
    # Hash corruption and bad MACs cannot explain the expected rejection.
    evidence[key]["metadata"].pop("vare_evidence_receipt")
    report = trust.attest(EvaluationReport(**evidence[key]))
    assert trust.verifies(report)
    evidence[key] = asdict(report)
    gate = PromotionGate(PromotionConfig(min_eval_examples=1, min_primary_gain=.01,
                                        trajectory_policy_sha256=expected_policy), trust_root=trust)
    assert not gate.decide(EvaluationReport(**evidence["incumbent_report"]),
                           EvaluationReport(**evidence["candidate_report"])).accepted
    file.write_text(json.dumps(evidence))
    manifest["artifact_sha256"][str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    script = repo / "scripts/audit_trajectory_capture_boundary_v1.py"
    run = subprocess.run([sys.executable, *(["-O"] if optimized else []), str(script), str(root)],
                         capture_output=True, text=True, timeout=5)
    assert run.returncode != 0 and "ValueError" in run.stderr, "auditor accepted a re-attested ineligible trajectory"
    assert not run.stdout
