"""Fail-closed RL training admission at the last use-time boundary."""
from __future__ import annotations

import asyncio

import pytest

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.types import Task
from test_safety_interlock import Hooks, Verifier


def _loop(hooks, *, authorizer=None, strict=True, samples=1):
    return CapabilityLoop(
        hooks=hooks,
        verifier=Verifier(),
        training_evidence_authorizer=authorizer,
        require_authenticated_training=strict,
        config=EngineConfig(
            replay_batch_size=samples,
            samples_per_task=samples,
            promotion=PromotionConfig(min_eval_examples=1, min_primary_gain=0.01),
        ),
    )


def _run(loop):
    return asyncio.run(loop.run_round(
        [Task("task1", "question")], round_index=0,
    ))


def test_secure_training_requires_trusted_authorizer():
    with pytest.raises(ValueError, match="trusted evidence authorizer"):
        _loop(Hooks(), authorizer=None)
    with pytest.raises(ValueError, match="must be bool"):
        CapabilityLoop(
            hooks=Hooks(), verifier=Verifier(),
            training_evidence_authorizer=lambda exp: True,
            require_authenticated_training="yes",
        )


def test_unsigned_or_truthy_nonboolean_evidence_never_updates_weights():
    for authorizer in (lambda _: False, lambda _: "false", lambda _: None):
        hooks = Hooks()
        loop = _loop(hooks, authorizer=authorizer)
        with pytest.raises(RuntimeError, match="unauthenticated_training_reward_evidence"):
            _run(loop)
        assert hooks.trained == 0
        assert hooks.promoted == []


def test_missing_or_crashed_external_authorizer_stops_before_training():
    hooks = Hooks()

    def unavailable(_experience):
        raise OSError("trusted scorer offline")

    loop = _loop(hooks, authorizer=unavailable)
    with pytest.raises(RuntimeError, match="unverifiable"):
        _run(loop)
    assert hooks.trained == 0


def test_mixed_authentication_rejects_entire_grpo_group():
    seen = []

    def validator(exp):
        seen.append(exp.attempt.created_step)
        return exp.attempt.created_step == 0

    hooks = Hooks()
    loop = _loop(hooks, authorizer=validator, samples=2)
    with pytest.raises(RuntimeError, match="unauthenticated_training_reward_evidence"):
        _run(loop)
    assert sorted(seen) == [0, 1]
    assert hooks.trained == 0
    assert hooks.promoted == []


def test_authenticated_complete_group_reaches_training():
    checked = []

    def validator(exp):
        checked.append(exp.attempt.created_step)
        return True

    hooks = Hooks()
    loop = _loop(hooks, authorizer=validator, samples=2)
    result = _run(loop)
    assert sorted(checked) == [0, 1]
    assert hooks.trained == 1
    assert result.decision.accepted
    assert hooks.promoted == ["p1"]
