"""CPU-only contract tests. Constructed failures are not real-model findings."""

from dataclasses import asdict, replace
import json

import pytest

from vare.failure_triage import (
    RewardGroupAudit,
    TaskAudit,
    TriageConfig,
    TriageSnapshot,
    load_snapshot,
    triage,
)


def _snapshot(**overrides):
    snapshot = TriageSnapshot(
        run_id="test-1",
        active_policy_version=4,
        rollout_policy_version=4,
        active_verifier_version=3,
        rollout_verifier_version=3,
        training_cohort_sha256="a" * 64,
        heldout_cohort_sha256="b" * 64,
        proxy_grader_sha256="c" * 64,
        trusted_grader_sha256="d" * 64,
        tasks=tuple(TaskAudit(f"task-{i}", True, i % 2 == 0, i % 2 == 0) for i in range(64)),
        reward_groups=tuple(RewardGroupAudit(f"group-{i}", (0.0, 1.0)) for i in range(12)),
        active_tokens=1000,
        clipped_tokens=100,
    )
    return replace(snapshot, **overrides)


def _signals(snapshot):
    return {f.signal for f in triage(snapshot).findings}


def test_clean_signal_neither_changes_weights_nor_proposes_a_probe():
    report = triage(_snapshot())
    assert report.status == "no_alert"
    assert report.next_probes == ()
    assert report.counts["heldout_tasks"] == 64


def test_paired_false_accepts_requires_trusted_disagreement():
    base = _snapshot()
    tasks = tuple(replace(t, proxy_pass=True, trusted_pass=False) if i < 8 else t
                  for i, t in enumerate(base.tasks))
    report = triage(replace(base, tasks=tasks))
    assert "proxy_false_accepts" in {f.signal for f in report.findings}
    assert report.counts["false_accepts"] == 8
    assert "blind_trusted_verifier_adjudication" in report.next_probes


def test_output_contract_and_reward_saturation_are_separate_observations():
    base = _snapshot()
    tasks = tuple(replace(t, format_valid=False) if i < 12 else t
                  for i, t in enumerate(base.tasks))
    groups = tuple(RewardGroupAudit(g.group_id, (1.0, 1.0)) for g in base.reward_groups)
    report = triage(replace(base, tasks=tasks, reward_groups=groups))
    assert {f.signal for f in report.findings} == {"output_contract_failures", "zero_group_advantage"}
    assert len(report.next_probes) == 2


def test_lag_and_token_clipping_make_probes_not_causal_claims():
    report = triage(_snapshot(rollout_policy_version=2, clipped_tokens=900))
    assert {f.signal for f in report.findings} == {"rollout_freshness", "high_clipped_fraction"}
    assert "Observational" in report.limitations[0]


def test_small_cohort_abstains_even_if_noisy():
    base = _snapshot()
    report = triage(replace(base, tasks=base.tasks[:10], clipped_tokens=990))
    assert report.status == "insufficient_evidence"
    assert report.findings == ()
    assert report.next_probes == ("expand_independent_task_audit",)


def test_group_evidence_must_meet_minimum_without_adding_false_signal():
    base = _snapshot()
    groups = (RewardGroupAudit("only", (1.0, 1.0)),)
    assert _signals(replace(base, reward_groups=groups)) == set()
    full = tuple(RewardGroupAudit(str(i), (0.0, 0.0)) for i in range(8))
    assert "zero_group_advantage" in _signals(replace(base, reward_groups=full))


@pytest.mark.parametrize("override, error", [
    ({"training_cohort_sha256": "b" * 64}, "cohort digests must differ"),
    ({"proxy_grader_sha256": "d" * 64}, "grader digests must differ"),
    ({"heldout_cohort_sha256": "wrong"}, "lowercase SHA-256"),
    ({"rollout_policy_version": 5}, "ahead"),
    ({"rollout_verifier_version": 4}, "ahead"),
    ({"clipped_tokens": 1001}, "exceed"),
    ({"active_tokens": True}, "nonnegative integer"),
    ({"tasks": ()}, "nonempty task"),
    ({"tasks": (TaskAudit("x", True, False, True),) * 2}, "duplicate held-out"),
    ({"tasks": (TaskAudit("x", 1, False, True),)}, "booleans"),
    ({"reward_groups": (RewardGroupAudit("a", (float("nan"), 1.0)),)}, "finite"),
    ({"reward_groups": (RewardGroupAudit("a", (1.0,)),)}, "at least two"),
    ({"reward_groups": (RewardGroupAudit("a", (0.0, 1.0)),) * 2}, "duplicate reward"),
])
def test_malformed_evidence_fails_closed(override, error):
    with pytest.raises(ValueError, match=error):
        triage(_snapshot(**override))


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError, match="finite"):
        triage(_snapshot(), TriageConfig(false_accept_rate=float("nan")))
    with pytest.raises(ValueError, match="positive"):
        triage(_snapshot(), TriageConfig(min_tasks=0))


def test_order_independent_diagnostic_counts():
    snap = _snapshot()
    forward = triage(snap)
    reverse = triage(replace(snap, tasks=tuple(reversed(snap.tasks)),
                              reward_groups=tuple(reversed(snap.reward_groups))))
    assert forward.to_dict() == reverse.to_dict()


def test_json_roundtrip_and_extra_fields_fail_closed():
    raw = json.loads(json.dumps(asdict(_snapshot())))
    report = triage(load_snapshot(raw))
    assert report.status == "no_alert"
    raw["surprise"] = True
    with pytest.raises(ValueError, match="malformed"):
        load_snapshot(raw)


def test_nested_invalid_json_entry_fails_closed():
    raw = json.loads(json.dumps(asdict(_snapshot())))
    raw["reward_groups"][0]["rewards"] = None
    with pytest.raises(ValueError, match="malformed"):
        load_snapshot(raw)
