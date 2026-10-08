import pytest

from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.reward_audit import RewardAuditObservation, audit_reward_labels
from vare.types import EvaluationReport


def _rows(n=128, false=0, family="coding"):
    return [
        RewardAuditObservation(f"heldout-{i}", family, True, i < false)
        for i in range(n)
    ]


def _report(policy, ids, primary, *, audit=None, measured=False, slices=None):
    meta = {"per_task_scores": {k: primary for k in ids}}
    if audit is not None:
        meta["reward_audit"] = [
            {"task_id": r.task_id, "family": r.family,
             "proxy_pass": r.proxy_pass, "trusted_pass": r.trusted_pass}
            for r in audit
        ]
    if measured:
        meta["verifier_disagreement_measured"] = True
    return EvaluationReport(policy_id=policy, primary=primary, n=len(ids),
                            verifier_disagreement=0.0, slices=slices or {},
                            metadata=meta)


def _gate(**kw):
    return PromotionGate(PromotionConfig(
        min_eval_examples=1, min_paired_examples=1,
        min_primary_gain=0.01, paired_confidence_gate=False,
        require_reward_audit=True, reward_audit_min_proxy_positives=32,
        **kw,
    ))


def test_zero_false_accepts_pass_with_enough_heldout_evidence():
    rows = _rows()
    audit = audit_reward_labels(rows)
    assert audit.accepted
    assert audit.overall.upper_bound < 0.25


def test_high_proxy_false_accept_rate_rejected():
    audit = audit_reward_labels(_rows(false=80))
    assert not audit.accepted
    assert "verifier_false_accept_risk:overall" in audit.reasons


def test_small_positive_sample_fails_closed():
    result = audit_reward_labels(_rows(n=3))
    assert not result.accepted
    assert "insufficient_proxy_positives:overall" in result.reasons


def test_one_family_failure_is_not_hidden_by_aggregate():
    rows = _rows(n=128) + [
        RewardAuditObservation(f"bad-{i}", "hard", True, False)
        for i in range(64)
    ]
    result = audit_reward_labels(rows, max_false_accept_ucb=0.40)
    assert not result.accepted
    assert "verifier_false_accept_risk:hard" in result.reasons


def test_duplicate_and_malformed_records_rejected():
    with pytest.raises(ValueError):
        audit_reward_labels(_rows(n=2) + _rows(n=2))
    with pytest.raises(ValueError):
        audit_reward_labels([RewardAuditObservation("x", "coding", 1, True)])


def test_promotion_requires_real_audit_evidence():
    rows = _rows()
    ids = [r.task_id for r in rows]
    inc = _report("inc", ids, 0.6)
    no_audit = _report("cand", ids, 0.8)
    assert "missing_reward_audit" in _gate().decide(inc, no_audit).reasons
    good = _report("cand", ids, 0.8, audit=rows)
    assert _gate().decide(inc, good).accepted
    bad_rows = _rows(false=90)
    bad = _report("cand", ids, 0.8, audit=bad_rows)
    assert "verifier_false_accept_risk:overall" in _gate().decide(inc, bad).reasons


def test_audit_must_match_paired_heldout_task_ids():
    rows = _rows()
    ids = [r.task_id for r in rows]
    inc = _report("inc", ids, 0.6)
    cand = _report("cand", ids + ["extra"], 0.8, audit=rows)
    assert "reward_audit_identity_mismatch" in _gate().decide(inc, cand).reasons


def test_missing_slice_and_unmeasured_disagreement_fail_closed_when_selected():
    rows = _rows()
    ids = [r.task_id for r in rows]
    inc = _report("inc", ids, 0.6, slices={"hard": 0.6})
    cand = _report("cand", ids, 0.8, audit=rows)
    strict = _gate(require_complete_slices=True,
                   require_measured_disagreement=True)
    d = strict.decide(inc, cand)
    assert "evaluation_slice_identity_mismatch" in d.reasons
    assert "unmeasured_verifier_disagreement" in d.reasons
    measured = _report("cand", ids, 0.8, audit=rows,
                       measured=True, slices={"hard": 0.8})
    assert strict.decide(inc, measured).accepted


def test_strict_audit_parameter_validation():
    with pytest.raises(ValueError):
        _gate(reward_audit_alpha=1.0)
    with pytest.raises(ValueError):
        audit_reward_labels(_rows(), min_proxy_positives=0)
