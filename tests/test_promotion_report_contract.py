from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.types import EvaluationReport


def _report(policy_id, primary, scores, *, n):
    return EvaluationReport(
        policy_id=policy_id,
        primary=primary,
        n=n,
        metadata={"per_task_scores": scores},
    )


def _gate():
    return PromotionGate(
        PromotionConfig(
            min_primary_gain=0.01,
            min_eval_examples=64,
            paired_confidence_gate=True,
            min_paired_examples=1,
            paired_bootstrap_samples=200,
            paired_alpha=0.05,
        )
    )


def test_promotion_rejects_declared_sample_count_larger_than_paired_evidence():
    incumbent = _report("incumbent", 0.50, {"task-0": 0.50}, n=64)
    candidate = _report("candidate", 0.70, {"task-0": 0.70}, n=64)

    decision = _gate().decide(incumbent, candidate)

    assert not decision.accepted
    assert "invalid_evaluation_metrics" in decision.reasons


def test_promotion_rejects_primary_metric_inconsistent_with_task_scores():
    scores = {f"task-{i}": 0.50 for i in range(64)}
    incumbent = _report("incumbent", 0.50, scores, n=64)
    candidate = _report("candidate", 0.70, scores, n=64)

    decision = _gate().decide(incumbent, candidate)

    assert not decision.accepted
    assert "invalid_evaluation_metrics" in decision.reasons


def test_consistent_complete_paired_report_remains_eligible():
    incumbent_scores = {f"task-{i}": 0.50 for i in range(64)}
    candidate_scores = {f"task-{i}": 0.70 for i in range(64)}
    incumbent = _report("incumbent", 0.50, incumbent_scores, n=64)
    candidate = _report("candidate", 0.70, candidate_scores, n=64)

    assert _gate().decide(incumbent, candidate).accepted
