import hashlib

import pytest
from vare.attestation import EvidenceTrustRoot
from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.trajectory_safety import (
    TrajectoryEvent, TrajectoryPolicy, TrustedTrajectoryRecorder,
    TrajectoryEvidenceError, audit_trajectory,
)
from vare.types import EvaluationReport


def _event(event_id, **kw):
    return TrajectoryEvent(event_id, "tool_call", "user", "allow", tool="browser", **kw)


def test_allowed_trace_passes(tmp_path):
    policy = TrajectoryPolicy("p1", ("browser",), ("example.org",))
    path = tmp_path / "trace.jsonl"
    rec = TrustedTrajectoryRecorder(path, run_id="run", policy_id="candidate")
    rec.record(_event("a"))
    rec.record(TrajectoryEvent("b", "network_request", "retrieved_webpage",
                               "allow", target="https://example.org/a", parent_event_id="a"))
    rec.close(expected_event_count=2)
    a = audit_trajectory(path, run_id="run", policy_id="candidate", policy=policy)
    assert a.status == "pass" and a.event_count == 2
    assert a.to_metadata()["capture_complete"]


@pytest.mark.parametrize("case", ["denied", "unknown", "bad_host", "missing_event"])
def test_unsafe_or_incomplete_trace_cannot_pass(tmp_path, case):
    policy = TrajectoryPolicy("p1", ("browser",), ("example.org",))
    path = tmp_path / "trace.jsonl"
    rec = TrustedTrajectoryRecorder(path, run_id="run", policy_id="candidate")
    if case == "denied":
        rec.record(TrajectoryEvent("a", "tool_call", "agent", "deny", tool="browser"))
    elif case == "unknown":
        rec.record(TrajectoryEvent("a", "tool_call", "agent", "unknown", tool="browser"))
    elif case == "bad_host":
        rec.record(TrajectoryEvent("a", "network_request", "tool_output",
                                   "allow", target="https://unapproved.test"))
    else:
        rec.record(_event("a"))
    rec.close(expected_event_count=2 if case == "missing_event" else 1)
    if case == "missing_event":
        with pytest.raises(TrajectoryEvidenceError):
            audit_trajectory(path, run_id="run", policy_id="candidate", policy=policy)
    else:
        assert audit_trajectory(path, run_id="run", policy_id="candidate", policy=policy).status != "pass"


def test_corruption_and_parent_spoof_fail(tmp_path):
    policy = TrajectoryPolicy("p1", ("browser",))
    path = tmp_path / "trace.jsonl"
    rec = TrustedTrajectoryRecorder(path, run_id="run", policy_id="candidate")
    rec.record(_event("a", parent_event_id="missing"))
    rec.close(expected_event_count=1)
    with pytest.raises(TrajectoryEvidenceError):
        audit_trajectory(path, run_id="run", policy_id="candidate", policy=policy)
    path.write_text(path.read_text().replace("candidate", "tampered", 1))
    with pytest.raises(TrajectoryEvidenceError):
        audit_trajectory(path, run_id="run", policy_id="candidate", policy=policy)


def test_trajectory_config_requires_trust_root():
    with pytest.raises(ValueError, match="trust root"):
        PromotionGate(PromotionConfig(trajectory_policy_sha256="a" * 64))


def test_legacy_gate_unchanged_by_default():
    gate = PromotionGate(PromotionConfig(min_primary_gain=0.01, min_eval_examples=1))
    assert gate.decide(EvaluationReport("old", .5, n=10),
                       EvaluationReport("new", .7, n=10)).accepted


def _trust_root():
    return EvidenceTrustRoot(
        key=bytes(range(32)),
        protocol_sha256=hashlib.sha256(b"trajectory-protocol").hexdigest(),
        evaluator_sha256=hashlib.sha256(b"independent-trajectory-evaluator").hexdigest(),
        run_id="trajectory-promotion-study-01",
    )


def _audit_metadata(policy_id, policy_digest, **changes):
    return {
        "schema": "vare.trajectory-safety.v1",
        "run_id": "trajectory-promotion-study-01",
        "policy_id": policy_id,
        "policy_sha256": policy_digest,
        "ledger_head_sha256": "a" * 64,
        "event_count": 0,
        "expected_event_count": 0,
        "capture_complete": True,
        "status": "pass",
        "violations": [],
        **changes,
    }


def _signed_report(trust, policy_id, score, audit=None):
    metadata = {} if audit is None else {"trajectory_safety": audit}
    return trust.attest(
        EvaluationReport(policy_id, score, {"hard": score}, n=1, metadata=metadata)
    )


def test_trajectory_gate_requires_signed_pass_audits_for_both_reports():
    trust = _trust_root()
    policy_sha256 = "b" * 64
    gate = PromotionGate(
        PromotionConfig(
            min_primary_gain=0.01,
            min_eval_examples=1,
            trajectory_policy_sha256=policy_sha256,
        ),
        trust_root=trust,
    )
    incumbent = _signed_report(
        trust, "old", 0.5, _audit_metadata("old", policy_sha256)
    )
    candidate = _signed_report(
        trust, "new", 0.7, _audit_metadata("new", policy_sha256)
    )
    assert gate.decide(incumbent, candidate).accepted

    missing_candidate_audit = _signed_report(trust, "new", 0.7)
    decision = gate.decide(incumbent, missing_candidate_audit)
    assert not decision.accepted
    assert decision.reasons == ("missing_trajectory_evidence",)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"ledger_head_sha256": "z" * 64}, "invalid_trajectory_evidence"),
        ({"run_id": "other-study"}, "invalid_trajectory_evidence"),
        ({"policy_sha256": "c" * 64}, "invalid_trajectory_evidence"),
        ({"capture_complete": False}, "invalid_trajectory_evidence"),
        ({"status": "unknown", "violations": ["unverified_authorization"]},
         "unsafe_or_unverified_trajectory"),
        ({"status": "violation", "violations": ["unauthorized_action"]},
         "unsafe_or_unverified_trajectory"),
    ],
)
def test_signed_but_invalid_trajectory_audit_fails_closed(changes, reason):
    trust = _trust_root()
    policy_sha256 = "b" * 64
    gate = PromotionGate(
        PromotionConfig(
            min_primary_gain=0.01,
            min_eval_examples=1,
            trajectory_policy_sha256=policy_sha256,
        ),
        trust_root=trust,
    )
    incumbent = _signed_report(
        trust, "old", 0.5, _audit_metadata("old", policy_sha256)
    )
    candidate = _signed_report(
        trust, "new", 0.7, _audit_metadata("new", policy_sha256, **changes)
    )
    decision = gate.decide(incumbent, candidate)
    assert not decision.accepted
    assert decision.reasons == (reason,)


def test_trajectory_gate_rejects_unsigned_evaluation_report():
    trust = _trust_root()
    policy_sha256 = "b" * 64
    gate = PromotionGate(
        PromotionConfig(
            min_primary_gain=0.01,
            min_eval_examples=1,
            trajectory_policy_sha256=policy_sha256,
        ),
        trust_root=trust,
    )
    incumbent = _signed_report(
        trust, "old", 0.5, _audit_metadata("old", policy_sha256)
    )
    unsigned = EvaluationReport(
        "new", 0.7, {"hard": 0.7}, n=1,
        metadata={"trajectory_safety": _audit_metadata("new", policy_sha256)},
    )
    decision = gate.decide(incumbent, unsigned)
    assert not decision.accepted
    assert decision.reasons == ("untrusted_trajectory_evidence",)
