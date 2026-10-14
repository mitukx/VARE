"""Promotion needs a distinct candidate, not two noisy scores for one policy."""

import pytest

from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.types import EvaluationReport


def _gate():
    return PromotionGate(PromotionConfig(min_eval_examples=1, min_primary_gain=0.01))


def test_repeated_same_policy_evaluation_cannot_promote_on_noisy_gain():
    incumbent = EvaluationReport("policy-v0", 0.20, n=64)
    candidate = EvaluationReport("policy-v0", 0.90, n=64)
    decision = _gate().decide(incumbent, candidate)
    assert not decision.accepted
    assert decision.reasons == ("no_distinct_candidate",)
    assert decision.primary_gain == 0.0


@pytest.mark.parametrize("bad_id", ["", "  ", None, 7])
def test_unusable_policy_identity_fails_closed(bad_id):
    incumbent = EvaluationReport("policy-v0", 0.20, n=64)
    candidate = EvaluationReport(bad_id, 0.90, n=64)
    decision = _gate().decide(incumbent, candidate)
    assert not decision.accepted
    assert decision.reasons == ("invalid_policy_identity",)


def test_distinct_candidate_remains_eligible_with_valid_improvement():
    incumbent = EvaluationReport("policy-v0", 0.20, n=64)
    candidate = EvaluationReport("policy-v1", 0.90, n=64)
    assert _gate().decide(incumbent, candidate).accepted
