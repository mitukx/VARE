"""Fault injection for rejected candidates and failed incumbent restoration."""
import asyncio

import pytest

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.types import Attempt, EvaluationReport, Task, Verification


class Verifier:
    active_version = 0

    async def verify(self, attempt):
        return Verification(1.0, True, 1.0, 0, "fixture", trusted=True)


class Hooks:
    def __init__(self, phase):
        self.phase = phase
        self.calls = []
        self.states = {"p0"}
        self.discard_started = asyncio.Event()
        self.discard_release = asyncio.Event()
        self.discard_error = None
        self.original_error = OSError("ledger unavailable")

    def active_policy(self):
        self.calls.append("active-policy")
        return "p0", 0

    async def rollout(self, task, policy_id, policy_version, step):
        self.calls.append("rollout")
        return Attempt(task, "answer", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id, experiences):
        self.calls.append("train")
        self.states.add("p1")
        return "p1"

    async def evaluate(self, policy_id):
        self.calls.append("evaluate-" + policy_id)
        if self.phase == "evaluation-error" and policy_id == "p1":
            raise ValueError("evaluation unavailable")
        score = 1.0 if self.phase == "ledger" and policy_id == "p1" else 0.0
        return EvaluationReport(policy_id, score, n=1)

    async def discard(self, candidate_id):
        self.calls.append("discard-start")
        self.discard_started.set()
        await self.discard_release.wait()
        if self.discard_error is not None:
            raise self.discard_error
        self.states.remove(candidate_id)
        self.calls.append("discard-end")

    async def promote(self, candidate_id):
        raise AssertionError("no promotion is authorized by this fixture")


def make_loop(hooks):
    class Ledger:
        def append(self, name, payload):
            if name == "promotion_intent":
                raise hooks.original_error

    return CapabilityLoop(
        hooks=hooks, verifier=Verifier(),
        decision_ledger=Ledger() if hooks.phase == "ledger" else None,
        config=EngineConfig(replay_batch_size=1,
                            promotion=PromotionConfig(min_primary_gain=0.1, min_eval_examples=1)),
    )


@pytest.mark.parametrize("phase", ["reject", "ledger"])
@pytest.mark.parametrize("discard_fails", [False, True])
def test_discard_is_terminal_before_cancellation_or_prior_error_escapes(phase, discard_fails):
    async def case():
        hooks = Hooks(phase)
        failure = OSError("restore failed")
        hooks.discard_error = failure if discard_fails else None
        loop = make_loop(hooks)
        task = asyncio.create_task(loop.run_round([Task("t", "prompt")], round_index=0))
        waiting = None
        try:
            await asyncio.wait_for(hooks.discard_started.wait(), 2)
            waiting = asyncio.create_task(loop.run_round([Task("t2", "prompt")], round_index=1))
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done(), "caller cancellation abandoned candidate discard"
            assert loop._round_lock.locked()
            assert "p1" in hooks.states
            assert not waiting.done()
            assert hooks.calls.count("rollout") == 1
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            hooks.discard_release.set()
            if discard_fails:
                with pytest.raises(RuntimeError, match="policy state is unknown") as caught:
                    await task
                assert caught.value.__cause__ is failure
                assert "p1" in hooks.states
            elif phase == "ledger":
                with pytest.raises(OSError) as caught:
                    await task
                assert caught.value is hooks.original_error
                assert hooks.states == {"p0"}
            else:
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert hooks.states == {"p0"}
            assert not loop._round_lock.locked()
        finally:
            hooks.discard_release.set()
            if waiting is not None:
                waiting.cancel()
            await asyncio.gather(task, *([waiting] if waiting else []), return_exceptions=True)

    asyncio.run(case())


@pytest.mark.parametrize("phase", ["reject", "evaluation-error", "ledger"])
def test_failed_discard_blocks_later_round_before_any_hook(phase):
    async def case():
        hooks = Hooks(phase)
        hooks.discard_error = OSError("restore failed")
        hooks.discard_release.set()
        loop = make_loop(hooks)
        with pytest.raises((OSError, RuntimeError)):
            await loop.run_round([Task("t", "prompt")], round_index=0)
        calls = list(hooks.calls)
        step = loop.step
        # A transiently cleared backend error is not independent evidence that
        # the incumbent's possibly partial restoration is now correct.
        hooks.discard_error = None
        with pytest.raises(RuntimeError, match="candidate recovery required"):
            await loop.run_round([Task("t2", "prompt")], round_index=1)
        assert hooks.calls == calls
        assert loop.step == step
        assert not loop._round_lock.locked()

    asyncio.run(case())
