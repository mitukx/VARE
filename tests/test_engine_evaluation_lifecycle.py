"""Owned evaluations must finish before candidate discard and round unlock."""
import asyncio

import pytest

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.types import Attempt, EvaluationReport, Task, Verification
from vare.verifiers import VerifierEnsemble, VerifierMember


class Verifier:
    name = "fixture"
    version = 0
    trusted = True

    async def verify(self, attempt):
        return Verification(1.0, True, 1.0, 0, self.name, trusted=True)


class Hooks:
    def __init__(self, *, fail_policy=None, changed=True, cleanup_error=False):
        self.fail_policy = fail_policy
        self.changed = changed
        self.cleanup_error = cleanup_error
        self.states = {"p0": 0.0}
        self.tasks = []
        self.started = [asyncio.Event(), asyncio.Event()]
        self.cancelling = [asyncio.Event(), asyncio.Event()]
        self.finished = [False, False]
        self.cleaned = [False, False]
        self.cleanup_release = asyncio.Event()
        self.evaluation_release = asyncio.Event()
        self.discard_started = asyncio.Event()
        self.discard_release = asyncio.Event()
        self.discard_error = None
        self.events = []

    def active_policy(self):
        return "p0", 0

    async def rollout(self, task, policy_id, policy_version, step):
        return Attempt(task, "ok", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id, experiences):
        if not self.changed:
            return incumbent_id
        self.states["p1"] = 0.0
        return "p1"

    async def evaluate(self, policy_id):
        index = len(self.tasks)
        self.tasks.append(asyncio.current_task())
        self.started[index].set()
        try:
            await self.started[1 - index].wait()
            if policy_id == self.fail_policy:
                raise RuntimeError("initial evaluation failure")
            try:
                await self.evaluation_release.wait()
            except asyncio.CancelledError:
                self.events.append(f"cancel-{index}")
                self.cancelling[index].set()
                await self.cleanup_release.wait()
                assert policy_id in self.states
                self.cleaned[index] = True
                if self.cleanup_error:
                    raise LookupError("secondary evaluator cleanup error")
                raise
            return EvaluationReport(policy_id, self.states[policy_id], n=1)
        finally:
            self.finished[index] = True
            self.events.append(f"finished-{index}")

    async def promote(self, candidate_id):
        raise AssertionError("equal-scoring fixture must not promote")

    async def discard(self, candidate_id):
        self.events.append("discard-start")
        self.discard_started.set()
        assert all(self.finished), "discard preceded evaluation termination"
        await self.discard_release.wait()
        if self.discard_error:
            raise self.discard_error
        self.states.pop(candidate_id)
        self.events.append("discard-finished")


def make_loop(hooks):
    return CapabilityLoop(
        hooks=hooks, verifier=VerifierEnsemble([VerifierMember(Verifier())]),
        config=EngineConfig(replay_batch_size=1,
                            promotion=PromotionConfig(min_primary_gain=1.0, min_eval_examples=1)),
    )


async def repeat_cancel(task):
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)


async def release_and_retrieve(hooks, task, *extra):
    hooks.evaluation_release.set()
    hooks.cleanup_release.set()
    hooks.discard_release.set()
    await asyncio.gather(task, *extra, return_exceptions=True)
    await asyncio.gather(*hooks.tasks, return_exceptions=True)


@pytest.mark.parametrize("fail_policy", ["p0", "p1"])
@pytest.mark.parametrize("cleanup_error", [False, True])
def test_evaluation_failure_drains_sibling_before_discard(fail_policy, cleanup_error):
    async def run():
        hooks = Hooks(fail_policy=fail_policy, cleanup_error=cleanup_error)
        loop = make_loop(hooks)
        task = asyncio.create_task(loop.run_round([Task("t", "prompt")], round_index=0))
        sibling = 1 if fail_policy == "p0" else 0
        signal = asyncio.create_task(hooks.cancelling[sibling].wait())
        try:
            done, _ = await asyncio.wait({task, signal}, timeout=2,
                                         return_when=asyncio.FIRST_COMPLETED)
            assert signal in done, "round returned without cancelling its sibling evaluation"
            await repeat_cancel(task)
            assert not task.done()
            assert loop._round_lock.locked()
            assert "p1" in hooks.states
            assert not hooks.discard_started.is_set()
            hooks.cleanup_release.set()
            await asyncio.wait_for(hooks.discard_started.wait(), 2)
            await repeat_cancel(task)
            assert not task.done(), "repeated cancel interrupted candidate discard"
            assert loop._round_lock.locked()
            hooks.discard_release.set()
            with pytest.raises(RuntimeError, match="initial evaluation failure"):
                await task
            assert hooks.cleaned[sibling]
            assert all(t.done() for t in hooks.tasks)
            assert hooks.events[-1] == "discard-finished"
            assert "p1" not in hooks.states
            assert not loop._round_lock.locked()
        finally:
            signal.cancel()
            await release_and_retrieve(hooks, task, signal)

    asyncio.run(run())


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("cleanup_error", [False, True])
def test_round_cancellation_waits_for_both_evaluations_and_discard(changed, cleanup_error):
    async def run():
        hooks = Hooks(changed=changed, cleanup_error=cleanup_error)
        loop = make_loop(hooks)
        task = asyncio.create_task(loop.run_round([Task("t", "prompt")], round_index=0))
        try:
            await asyncio.wait_for(asyncio.gather(*(s.wait() for s in hooks.started)), 2)
            task.cancel()
            await asyncio.wait_for(asyncio.gather(*(s.wait() for s in hooks.cancelling)), 2)
            await repeat_cancel(task)
            assert not task.done(), "repeat cancel escaped evaluation cleanup"
            assert loop._round_lock.locked()
            assert not hooks.discard_started.is_set()
            hooks.cleanup_release.set()
            if changed:
                await asyncio.wait_for(hooks.discard_started.wait(), 2)
                await repeat_cancel(task)
                assert not task.done(), "repeat cancel escaped discard"
                hooks.discard_release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert all(hooks.finished) and all(hooks.cleaned)
            assert all(t.done() for t in hooks.tasks)
            assert not loop._round_lock.locked()
            assert set(hooks.states) == {"p0"}
            assert hooks.discard_started.is_set() == changed
        finally:
            await release_and_retrieve(hooks, task)

    asyncio.run(run())


@pytest.mark.parametrize("discard_error", [RuntimeError("discard fixture error"), asyncio.CancelledError()])
def test_discard_failure_is_reported_after_evaluations_end(discard_error):
    async def run():
        hooks = Hooks(fail_policy="p0")
        hooks.cleanup_release.set()
        hooks.discard_release.set()
        hooks.discard_error = discard_error
        loop = make_loop(hooks)
        task = asyncio.create_task(loop.run_round([Task("t", "prompt")], round_index=0))
        try:
            with pytest.raises(RuntimeError, match="policy state is unknown") as caught:
                await task
            assert caught.value.__cause__ is discard_error
            assert all(hooks.finished)
            assert all(t.done() for t in hooks.tasks)
        finally:
            await release_and_retrieve(hooks, task)

    asyncio.run(run())


def test_successful_evaluations_remain_concurrent_and_precede_discard():
    async def run():
        hooks = Hooks()
        hooks.discard_release.set()
        loop = make_loop(hooks)
        task = asyncio.create_task(loop.run_round([Task("t", "prompt")], round_index=0))
        try:
            await asyncio.wait_for(asyncio.gather(*(s.wait() for s in hooks.started)), 2)
            assert not task.done()
            hooks.evaluation_release.set()
            result = await task
            assert not result.decision.accepted
            assert all(hooks.finished)
            assert not any(s.is_set() for s in hooks.cancelling)
            assert hooks.events[-2:] == ["discard-start", "discard-finished"]
        finally:
            await release_and_retrieve(hooks, task)

    asyncio.run(run())


@pytest.mark.parametrize("self_cancel", [False, True])
def test_already_cancelling_evaluator_is_drained_without_another_cancel(self_cancel):
    class AlreadyCancellingHooks(Hooks):
        async def evaluate(self, policy_id):
            index = 0 if policy_id == "p0" else 1
            current = asyncio.current_task()
            if index == 1:
                self.candidate_task = current
            self.tasks.append(current)
            self.started[index].set()
            try:
                if index == 0:
                    await self.cancelling[1].wait()
                    raise RuntimeError("initial evaluation failure")
                if self_cancel:
                    current.cancel()
                try:
                    await self.evaluation_release.wait()
                except asyncio.CancelledError:
                    self.cancelling[1].set()
                    await self.cleanup_release.wait()
                    self.cleaned[1] = True
                    raise
            finally:
                self.finished[index] = True

    async def run():
        hooks = AlreadyCancellingHooks()
        hooks.discard_release.set()
        loop = make_loop(hooks)
        task = asyncio.create_task(loop.run_round([Task("t", "prompt")], round_index=0))
        signal = asyncio.create_task(hooks.discard_started.wait())
        try:
            await asyncio.wait_for(hooks.started[1].wait(), 2)
            if not self_cancel:
                hooks.candidate_task.cancel()
            await asyncio.wait_for(hooks.cancelling[1].wait(), 2)
            done, _ = await asyncio.wait({task, signal}, timeout=0.05)
            assert not done, "engine interrupted an evaluator already in cancellation cleanup"
            assert not hooks.finished[1]
            assert loop._round_lock.locked()
            hooks.cleanup_release.set()
            with pytest.raises(RuntimeError, match="initial evaluation failure"):
                await task
            assert hooks.cleaned[1]
            assert all(hooks.finished)
            assert all(t.done() for t in hooks.tasks)
            assert "p1" not in hooks.states
        finally:
            signal.cancel()
            await release_and_retrieve(hooks, task, signal)

    asyncio.run(run())
