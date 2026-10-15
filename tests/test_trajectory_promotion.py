from dataclasses import replace

from vare.attestation import EvidenceTrustRoot
from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.trajectory_safety import (
    TrajectoryPolicy, TrustedTrajectoryRecorder, audit_trajectory,
)
from vare.types import EvaluationReport


def _attested_report(tmp_path, trust, policy, policy_id, primary, status="pass"):
    trace = tmp_path / f"{policy_id}.jsonl"
    recorder = TrustedTrajectoryRecorder(trace, run_id=trust.run_id, policy_id=policy_id)
    if status == "violation":
        from vare.trajectory_safety import TrajectoryEvent
        recorder.record(TrajectoryEvent("e1", "tool_call", "agent", "deny", tool="missing"))
    recorder.close(expected_event_count=1 if status == "violation" else 0)
    audit = audit_trajectory(trace, run_id=trust.run_id, policy_id=policy_id, policy=policy)
    report = EvaluationReport(
        policy_id=policy_id, primary=primary, n=12,
        metadata={"trajectory_safety": audit.to_metadata()},
    )
    return trust.attest(report)


def test_independent_signed_trajectory_allows_qualified_candidate(tmp_path):
    policy = TrajectoryPolicy("v1", ("browser",))
    trust = EvidenceTrustRoot(
        key=b"k" * 32, protocol_sha256="a" * 64,
        evaluator_sha256="b" * 64, run_id="isolated-run-0001",
    )
    incumbent = _attested_report(tmp_path, trust, policy, "old", .4)
    candidate = _attested_report(tmp_path, trust, policy, "candidate", .8)
    gate = PromotionGate(
        PromotionConfig(min_eval_examples=1, min_primary_gain=.01, trajectory_policy_sha256=policy.sha256),
        trust_root=trust,
    )
    assert gate.decide(incumbent, candidate).accepted
    tampered = replace(candidate, metadata={
        **candidate.metadata,
        "trajectory_safety": {**candidate.metadata["trajectory_safety"], "status": "pass", "event_count": 999},
    })
    assert not gate.decide(incumbent, tampered).accepted


def test_signed_denied_trajectory_is_not_promotable(tmp_path):
    policy = TrajectoryPolicy("v1", ("browser",))
    trust = EvidenceTrustRoot(
        key=b"k" * 32, protocol_sha256="a" * 64,
        evaluator_sha256="b" * 64, run_id="isolated-run-0001",
    )
    incumbent = _attested_report(tmp_path, trust, policy, "old", .4)
    candidate = _attested_report(tmp_path, trust, policy, "candidate", .9, status="violation")
    gate = PromotionGate(
        PromotionConfig(min_eval_examples=1, min_primary_gain=.01, trajectory_policy_sha256=policy.sha256),
        trust_root=trust,
    )
    decision = gate.decide(incumbent, candidate)
    assert not decision.accepted
    assert "unsafe_or_unverified_trajectory" in decision.reasons


def test_missing_trajectory_metadata_fails_closed(tmp_path):
    policy = TrajectoryPolicy("v1", ("browser",))
    trust = EvidenceTrustRoot(
        key=b"k" * 32, protocol_sha256="a" * 64,
        evaluator_sha256="b" * 64, run_id="isolated-run-0001",
    )
    incumbent = _attested_report(tmp_path, trust, policy, "old", .4)
    candidate = trust.attest(EvaluationReport("candidate", .99, n=12))
    gate = PromotionGate(
        PromotionConfig(min_eval_examples=1, trajectory_policy_sha256=policy.sha256),
        trust_root=trust,
    )
    decision = gate.decide(incumbent, candidate)
    assert not decision.accepted
    assert "missing_trajectory_evidence" in decision.reasons
