import asyncio
from vare.demo import run_demo


def test_demo_produces_at_least_one_promoted_improvement():
    rs = asyncio.run(run_demo(rounds=8, rollouts=512, seed=7))
    assert any(r.decision.accepted and r.decision.primary_gain > 0 for r in rs)
    active_scores = [r.candidate_eval.primary for r in rs if r.decision.accepted]
    assert active_scores
