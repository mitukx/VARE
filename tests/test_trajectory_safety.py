import pytest
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
