from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.types import EvaluationReport


def _report(policy, scores):
    return EvaluationReport(
        policy_id=policy,
        primary=sum(scores.values()) / len(scores),
        n=len(scores),
        metadata={"per_task_scores": scores},
    )


def test_paired_gate_accepts_consistent_gain():
    old = {f"t{i}": 0.0 for i in range(100)}
    new = {k: (1.0 if i < 30 else 0.0) for i, k in enumerate(old)}
    gate = PromotionGate(
        PromotionConfig(
            min_primary_gain=0.05,
            min_eval_examples=50,
            paired_confidence_gate=True,
            min_paired_examples=50,
            paired_bootstrap_samples=400,
            paired_alpha=0.05,
        )
    )
    decision = gate.decide(_report("a", old), _report("b", new))
    assert decision.accepted
    assert decision.paired_lcb is not None and decision.paired_lcb > 0.05


def test_paired_gate_rejects_missing_or_mismatched_identity():
    gate = PromotionGate(
        PromotionConfig(
            min_primary_gain=0.0,
            min_eval_examples=1,
            paired_confidence_gate=True,
            min_paired_examples=1,
            paired_bootstrap_samples=20,
        )
    )
    inc = _report("a", {"x": 0.0})
    cand = _report("b", {"y": 1.0})
    d = gate.decide(inc, cand)
    assert not d.accepted
    assert "paired_identity_mismatch" in d.reasons
