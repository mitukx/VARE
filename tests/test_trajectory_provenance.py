import pytest

from vare.trajectory_provenance import replay_provenance
from vare.trajectory_safety import (
    TrajectoryEvent, TrajectoryEvidenceError, TrajectoryPolicy,
    TrustedTrajectoryRecorder,
)


def test_untrusted_ancestry_is_observed_not_automatically_denied(tmp_path):
    policy = TrajectoryPolicy("v1", ("browser",), ("example.org",))
    path = tmp_path / "trace.jsonl"
    recorder = TrustedTrajectoryRecorder(path, run_id="r", policy_id="candidate")
    recorder.record(TrajectoryEvent("user", "observation", "user", "not_applicable"))
    recorder.record(TrajectoryEvent("page", "observation", "retrieved_webpage",
                                    "not_applicable", parent_event_id="user"))
    recorder.record(TrajectoryEvent("tool", "tool_call", "agent", "allow",
                                    tool="browser", parent_event_id="page"))
    recorder.record(TrajectoryEvent("request", "network_request", "agent", "allow",
                                    target="https://example.org/", parent_event_id="tool"))
    recorder.close(expected_event_count=4)
    replay = replay_provenance(path, run_id="r", policy_id="candidate", policy=policy)
    assert replay.actions_with_untrusted_ancestry == ("tool", "request")
    assert replay.nodes[-1].untrusted_ancestor_ids == ("page",)
    assert replay.observed_untrusted_action_count == 2


def test_explicit_parent_injection_fails_replay(tmp_path):
    policy = TrajectoryPolicy("v1", ("browser",))
    path = tmp_path / "trace.jsonl"
    recorder = TrustedTrajectoryRecorder(path, run_id="r", policy_id="candidate")
    recorder.record(TrajectoryEvent("tool", "tool_call", "agent", "allow",
                                    tool="browser", parent_event_id="missing"))
    recorder.close(expected_event_count=1)
    with pytest.raises(TrajectoryEvidenceError):
        replay_provenance(path, run_id="r", policy_id="candidate", policy=policy)


def test_authorized_action_without_explicit_untrusted_parent(tmp_path):
    policy = TrajectoryPolicy("v1", ("browser",))
    path = tmp_path / "trace.jsonl"
    recorder = TrustedTrajectoryRecorder(path, run_id="r", policy_id="candidate")
    recorder.record(TrajectoryEvent("user", "observation", "user", "not_applicable"))
    recorder.record(TrajectoryEvent("tool", "tool_call", "agent", "allow",
                                    tool="browser", parent_event_id="user"))
    recorder.close(expected_event_count=2)
    replay = replay_provenance(path, run_id="r", policy_id="candidate", policy=policy)
    assert replay.actions_with_untrusted_ancestry == ()
