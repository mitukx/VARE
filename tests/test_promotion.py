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
