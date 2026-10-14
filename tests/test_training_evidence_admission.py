"""Fail-closed RL training admission at the last use-time boundary."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
import sys

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


def test_authenticated_rvl_group_rejects_repeated_member_before_training():
    class RepeatedMemberHooks(Hooks):
        async def rollout(self, task, policy_id, policy_version, step):
            attempt = await super().rollout(task, policy_id, policy_version, step)
            group_id = task.metadata["vare_rollout_group"]
            attempt.metadata.update({
                "rvl_group_id": group_id,
                "rvl_generation_index": 0,
                "rvl_generation": {"prompt_id": task.id},
            })
            return attempt

    hooks = RepeatedMemberHooks()
    loop = _loop(hooks, authorizer=lambda _experience: True, samples=2)
    with pytest.raises(RuntimeError, match="training_reward_evidence_unverifiable"):
        _run(loop)
    assert hooks.trained == 0
    assert hooks.promoted == []


def test_authenticated_live_rvl_rollouts_need_no_replay_member_indices():
    class LiveRvlHooks(Hooks):
        async def rollout(self, task, policy_id, policy_version, step):
            attempt = await super().rollout(task, policy_id, policy_version, step)
            attempt.metadata["rvl_generation"] = {"prompt_id": task.id}
            return attempt

    hooks = LiveRvlHooks()
    loop = _loop(
        hooks,
        authorizer=lambda exp: isinstance(
            exp.attempt.metadata.get("rvl_generation"), dict
        ),
        samples=2,
    )
    result = _run(loop)
    assert hooks.trained == 1
    assert result.decision.accepted


def test_engine_and_rvl_hooks_import_in_a_fresh_process():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import vare.engine; import vare.integrations.rvl_grpo",
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
