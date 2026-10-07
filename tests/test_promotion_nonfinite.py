import math

import pytest

from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.types import EvaluationReport


def _report(**overrides):
    values = {
        "policy_id": "candidate",
        "primary": 0.73,
        "slices": {"hard": 0.60},
        "cost": 1.0,
        "verifier_disagreement": 0.02,
        "n": 100,
        "metadata": {},
    }
    values.update(overrides)
    return EvaluationReport(**values)


@pytest.mark.parametrize(
    "incumbent,candidate",
    [
        (_report(policy_id="incumbent"), _report(primary=math.nan)),
        (_report(policy_id="incumbent"), _report(primary=math.inf)),
        (_report(policy_id="incumbent", primary=math.nan), _report()),
        (_report(policy_id="incumbent"), _report(slices={"hard": math.nan})),
        (_report(policy_id="incumbent"), _report(cost=math.nan)),
        (_report(policy_id="incumbent"), _report(verifier_disagreement=math.nan)),
    ],
)
def test_promotion_rejects_nonfinite_report_metrics(incumbent, candidate):
    gate = PromotionGate(PromotionConfig(min_primary_gain=0.01, min_eval_examples=10))

    decision = gate.decide(incumbent, candidate)

    assert not decision.accepted
    assert decision.reasons == ("invalid_evaluation_metrics",)
    assert math.isfinite(decision.primary_gain)
    assert math.isfinite(decision.worst_slice_regression)


def test_paired_gate_rejects_nonfinite_task_score():
    gate = PromotionGate(PromotionConfig(
        min_primary_gain=0.01,
        min_eval_examples=1,
        paired_confidence_gate=True,
        min_paired_examples=1,
        paired_bootstrap_samples=20,
    ))
    incumbent = _report(primary=0.70, n=1, metadata={"per_task_scores": {"task": 0.60}})
    candidate = _report(primary=0.73, n=1, metadata={"per_task_scores": {"task": math.nan}})

    decision = gate.decide(incumbent, candidate)

    assert not decision.accepted
    assert "invalid_evaluation_metrics" in decision.reasons
    assert decision.paired_gain is None
    assert decision.paired_lcb is None


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_promotion_rejects_paired_scores_outside_declared_bounds(score):
    gate = PromotionGate(PromotionConfig(min_primary_gain=0.01, min_eval_examples=1))
    incumbent = _report(policy_id="incumbent", n=1,
                        metadata={"per_task_scores": {"task": 0.60}})
    candidate = _report(n=1, metadata={"per_task_scores": {"task": score}})

    decision = gate.decide(incumbent, candidate)

    assert not decision.accepted
    assert "invalid_evaluation_metrics" in decision.reasons


def test_finite_positive_control_remains_accepted():
    gate = PromotionGate(PromotionConfig(min_primary_gain=0.01, min_eval_examples=10))
    incumbent = _report(policy_id="incumbent", primary=0.70)
    candidate = _report(primary=0.73)

    assert gate.decide(incumbent, candidate).accepted
