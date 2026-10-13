"""Focused, dependency-free CPU regressions for improvement-value learning."""
import math
from dataclasses import replace

import pytest

from vare.meta_improvement import (
    Candidate, Feedback, INTERVENTIONS, LinearImprovementModel,
    TabularRLVR, canonical_summary_json, run_episode, summarize,
)


def test_proxy_policy_gradient_matches_finite_difference():
    world = TabularRLVR(1)
    p = world.probabilities()[0]
    tpr, fpr = world.tpr[0], world.fpr[0]
    def proxy(logit):
        pp = 1 / (1 + math.exp(-logit))
        return tpr * pp + fpr * (1 - pp)
    h = 1e-5
    fd = (proxy(world.logits[0] + h) - proxy(world.logits[0] - h)) / (2 * h)
    assert fd == pytest.approx(p * (1 - p) * (tpr - fpr), abs=1e-9)


def test_unrepaired_proxy_hack_reduces_true_success():
    w = TabularRLVR(4)
    before = w.development()
    w.apply(INTERVENTIONS[2])
    assert w.development() < before


def test_verifier_repair_can_reverse_bad_gradient():
    w = TabularRLVR(4)
    before = w.development()
    w.apply(INTERVENTIONS[3])
    assert w.development() > before
    assert w.verifier_version == 1
    with pytest.raises(ValueError, match="already repaired"):
        w.apply(INTERVENTIONS[3])


def test_feedback_provenance_and_replay_fail_closed():
    m = LinearImprovementModel()
    world = TabularRLVR(0)
    choice = m.choose(world.candidates(), 20)
    before = world.development()
    world.apply(choice.candidate.intervention)
    valid = Feedback(choice.decision_id, choice.candidate.intervention.name,
                     0, 1, world.verifier_version, "dev:a:1", before,
                     world.development(), choice.candidate.intervention.cost)
    for bad in (replace(valid, action="train:other"),
                replace(valid, after_policy_version=10),
                replace(valid, evaluation_id=""),
                replace(valid, development_after=float("nan"))):
        with pytest.raises(ValueError):
            m.observe(bad)
    with pytest.raises(RuntimeError):
        m.choose(world.candidates(), 20)
    m.observe(valid)
    with pytest.raises(ValueError):
        m.observe(valid)
    choice2 = m.choose(world.candidates(), 20)
    with pytest.raises(ValueError, match="replayed feedback"):
        m.observe(replace(valid, decision_id=choice2.decision_id,
                          action=choice2.candidate.intervention.name,
                          before_policy_version=choice2.candidate.policy_version,
                          after_policy_version=choice2.candidate.policy_version + 1,
                          verifier_version=choice2.candidate.verifier_version + int(choice2.candidate.intervention.repairs_verifier),
                          cost=choice2.candidate.intervention.cost))


def test_invalid_candidates_rejected():
    w = TabularRLVR(1)
    m = LinearImprovementModel()
    candidate = w.candidates()[0]
    with pytest.raises(ValueError):
        m.choose([candidate, candidate], 5)
    with pytest.raises(ValueError):
        m.choose([candidate], 0)
    with pytest.raises(ValueError):
        m.score(Candidate(candidate.intervention, (1.0, float("nan")), 0, 0))


def test_confirmation_never_read_inside_controller_loop():
    r = run_episode("value_ucb", seed=8, budget=32)
    assert r.confirmation_reads_during_training == 0
    assert r.cost_spent <= r.budget
    assert r.feedback
    assert [f.after_policy_version for f in r.feedback] == list(range(1, len(r.feedback) + 1))
    assert all(f.development_before >= 0 and f.development_after <= 1 for f in r.feedback)
    assert all(f.evaluation_id.startswith("dev:") for f in r.feedback)


def test_paired_report_reproducible_and_complete():
    first = canonical_summary_json(range(5), budget=16)
    assert first == canonical_summary_json(range(5), budget=16)
    report = summarize(range(5), budget=16)
    assert len(report["episodes"]) == 20
    assert set(report["paired_vs_value_ucb"]) == {"round_robin", "proxy_greedy", "one_step_oracle"}
    assert all(row["confirmation_reads_during_training"] == 0 for row in report["episodes"])
    with pytest.raises(ValueError):
        summarize([1, 1])
