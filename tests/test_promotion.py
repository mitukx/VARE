import pytest

from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.types import EvaluationReport


def test_promotion_gate_accepts_real_gain_and_rejects_regression():
    gate = PromotionGate(PromotionConfig(min_primary_gain=0.01, max_slice_regression=0.03, min_eval_examples=10))
    inc = EvaluationReport("a", 0.70, {"hard": 0.60}, n=100)
    good = EvaluationReport("b", 0.73, {"hard": 0.60}, n=100)
    assert gate.decide(inc, good).accepted
    bad = EvaluationReport("c", 0.75, {"hard": 0.50}, n=100)
    d = gate.decide(inc, bad)
    assert not d.accepted
    assert "slice_regression" in d.reasons


def test_promotion_rejects_candidate_with_incomplete_slice_coverage():
    gate = PromotionGate(PromotionConfig(min_primary_gain=0.01, min_eval_examples=1))
    incumbent = EvaluationReport("a", 0.50, {"hard": 0.4, "easy": 0.7}, n=10)
    candidate = EvaluationReport("b", 0.80, {"easy": 0.7}, n=10)

    decision = gate.decide(incumbent, candidate)

    assert not decision.accepted
    assert "slice_coverage_mismatch" in decision.reasons


def test_promotion_rejects_identically_omitted_required_slice():
    gate = PromotionGate(PromotionConfig(
        min_primary_gain=0.01,
        min_eval_examples=1,
        required_slice_names=("easy", "hard"),
    ))
    incumbent = EvaluationReport("a", 0.50, {"easy": 0.5}, n=10)
    candidate = EvaluationReport("b", 0.80, {"easy": 0.8}, n=10)

    decision = gate.decide(incumbent, candidate)

    assert not decision.accepted
    assert "slice_coverage_mismatch" in decision.reasons


def test_promotion_accepts_declared_complete_slice_set():
    gate = PromotionGate(PromotionConfig(
        min_primary_gain=0.01,
        min_eval_examples=1,
        required_slice_names=("easy", "hard"),
    ))
    incumbent = EvaluationReport("a", 0.50, {"easy": 0.5, "hard": 0.5}, n=10)
    candidate = EvaluationReport("b", 0.60, {"easy": 0.7, "hard": 0.5}, n=10)

    assert gate.decide(incumbent, candidate).accepted


@pytest.mark.parametrize("slice_names", [("easy", "easy"), ("",), ["easy"]])
def test_promotion_rejects_malformed_required_slice_configuration(slice_names):
    with pytest.raises(ValueError, match="required_slice_names"):
        PromotionGate(PromotionConfig(required_slice_names=slice_names))
