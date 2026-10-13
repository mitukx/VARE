"""Rollout lifetime and provenance regressions for the CPU control plane."""
from __future__ import annotations

import asyncio

import pytest

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.types import Attempt, EvaluationReport, Task, Verification


class _Verifier:
    active_version = 0

    def __init__(self):
        self.calls = 0

    async def verify(self, attempt):
        self.calls += 1
        return Verification(1.0, True, 1.0, 0, "fixture", trusted=True)


class _Hooks:
    def __init__(self, defect=None):
        self.defect = defect
        self.train_called = False

    def active_policy(self):
        return "current", 1

    async def rollout(self, task, policy_id, policy_version, step):
        if self.defect == "policy":
            policy_id = "another-policy"
        if self.defect == "step":
            step += 10
        if self.defect == "stale":
            policy_version -= 1
        return Attempt(task, "answer", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id, experiences):
        self.train_called = True
        return incumbent_id

    async def evaluate(self, policy_id):
        return EvaluationReport(policy_id, 0.0, n=1)

    async def promote(self, candidate_id):
        raise AssertionError("no promotion")

    async def discard(self, candidate_id):
        raise AssertionError("no candidate")


def _loop(hooks, verifier):
    return CapabilityLoop(
        hooks=hooks,
        verifier=verifier,
        config=EngineConfig(replay_batch_size=2, promotion=PromotionConfig(min_eval_examples=1)),
    )


@pytest.mark.parametrize(("defect", "reason"), [
    ("policy", "rollout policy identity mismatch"),
    ("step", "rollout step identity mismatch"),
])
def test_misbound_rollout_rejected_before_verifier_or_training(defect, reason):
    hooks = _Hooks(defect)
    verifier = _Verifier()
    loop = _loop(hooks, verifier)
    with pytest.raises(ValueError, match=reason):
        asyncio.run(loop.run_round([Task("a", "prompt")], round_index=0))
    assert verifier.calls == 0
    assert not hooks.train_called
    assert len(loop.replay) == 0


def test_permitted_stale_policy_version_keeps_original_identity():
    hooks = _Hooks("stale")
    verifier = _Verifier()
    result = asyncio.run(_loop(hooks, verifier).run_round([Task("a", "prompt")], round_index=0))
    assert result.admitted_experiences == 1
    assert verifier.calls == 1
    assert hooks.train_called


class _ConcurrentHooks(_Hooks):
    def __init__(self, *, fail_in_verifier=False):
        super().__init__()
        self.entered = asyncio.Event()
        self.completed = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.fail_in_verifier = fail_in_verifier

    async def rollout(self, task, policy_id, policy_version, step):
        if task.id == "slow":
            self.entered.set()
            try:
                await self.completed.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        else:
            await self.entered.wait()
            await asyncio.sleep(0)
            if not self.fail_in_verifier:
                raise RuntimeError("rollout failure")
        return Attempt(task, "answer", policy_id, policy_version, step)


class _FailingVerifier(_Verifier):
    async def verify(self, attempt):
        if attempt.task.id == "fail":
            raise RuntimeError("verification failure")
        return await super().verify(attempt)


@pytest.mark.parametrize("in_verifier", [False, True])
def test_failed_round_drains_sibling_before_return(in_verifier):
    class Hooks(_ConcurrentHooks):
        async def rollout(self, task, policy_id, policy_version, step):
            if task.id == "slow":
                return await super().rollout(task, policy_id, policy_version, step)
            await self.entered.wait()
            if not in_verifier:
                raise RuntimeError("rollout failure")
            return Attempt(task, "answer", policy_id, policy_version, step)

    async def case():
        hooks = Hooks()
        verifier = _FailingVerifier() if in_verifier else _Verifier()
        loop = _loop(hooks, verifier)
        # The production curriculum samples with replacement. Force one
        # slow and one failing task so this concurrency test is not a lottery.
        loop.curriculum.choose = lambda tasks, n: list(tasks)
        with pytest.raises(RuntimeError, match="verification failure" if in_verifier else "rollout failure"):
            await asyncio.wait_for(
                loop.run_round([Task("slow", "prompt"), Task("fail", "prompt")], round_index=0),
                timeout=1,
            )
        assert hooks.cancelled.is_set()
        assert not hooks.train_called
        assert len(loop.replay) == 0

    asyncio.run(case())


def test_caller_cancellation_drains_active_rollout_before_unlock():
    async def case():
        hooks = _ConcurrentHooks()
        loop = _loop(hooks, _Verifier())
        running = asyncio.create_task(
            loop.run_round([Task("slow", "prompt")], round_index=0)
        )
        await asyncio.wait_for(hooks.entered.wait(), timeout=1)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, timeout=1)
        assert hooks.cancelled.is_set()
        assert len(loop.replay) == 0

    asyncio.run(case())
