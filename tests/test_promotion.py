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
