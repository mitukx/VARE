"""Frozen CPU counterexamples for whole-round ownership and restart fencing."""
import asyncio
import multiprocessing
import os
from pathlib import Path

import pytest

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.promotion_journal import PromotionJournal, PromotionRecoveryRequired
from vare.types import EvaluationReport, Task
from test_promotion_publication_recovery import Hooks, Verifier, digest
from test_checkpoint_bound_promotion import trust


class BoundaryHooks(Hooks):
    def __init__(self, *, fault=None, signer=None):
        super().__init__(signer=signer)
        self.fault = fault
        self.training_entered = asyncio.Event()
        self.training_release = asyncio.Event()

    async def train_candidate(self, incumbent_id, experiences):
        if self.fault == "train":
            self.calls.append("train-failed")
            raise OSError("optimizer state unknown")
        if self.fault == "pause":
            self.calls.append("train-paused")
            self.training_entered.set()
            await self.training_release.wait()
        return await super().train_candidate(incumbent_id, experiences)

    async def evaluate(self, policy_id):
        if self.fault == "discard":
            report = EvaluationReport(policy_id, 0.4 if policy_id == "p0" else 0.1, n=10)
            return self.signer.attest(report, checkpoint_sha256=self.digests[policy_id]) if self.signer else report
        return await super().evaluate(policy_id)

    async def discard(self, candidate_id):
        if self.fault == "discard":
            self.calls.append("discard-failed")
            raise OSError("restore incomplete")
        await super().discard(candidate_id)


class LostAckJournal(PromotionJournal):
    def commit(self, tx_id):
        super().commit(tx_id)
        raise OSError("committed but acknowledgment lost")


def journal(path, cls=PromotionJournal):
    return cls(path, initial_policy_id="p0", initial_checkpoint_sha256=digest("weights-p0"))


def loop(hooks, durable=None):
    return CapabilityLoop(
        hooks=hooks, verifier=Verifier(), promotion_journal=durable,
        evidence_trust_root=hooks.signer,
        checkpoint_hasher=(lambda pid: hooks.digests[pid]) if durable else None,
        config=EngineConfig(replay_batch_size=1,
                            promotion=PromotionConfig(min_primary_gain=0.1, min_eval_examples=1)),
    )


async def run(control):
    return await control.run_round([Task("t", "prompt")], round_index=0)


async def fenced(control, hooks):
    calls = list(hooks.calls)
    error = None
    try:
        await run(control)
    except BaseException as exc:
        error = exc
    assert hooks.calls == calls, "unresolved runtime allowed another hook access"
    assert control.step == 0
    assert isinstance(error, PromotionRecoveryRequired)


def test_two_loops_same_hooks_serialize_before_policy_lookup():
    async def case():
        hooks = BoundaryHooks(fault="pause")
        first, second = loop(hooks), loop(hooks)
        one = asyncio.create_task(run(first))
        two = None
        try:
            await asyncio.wait_for(hooks.training_entered.wait(), 2)
            calls = list(hooks.calls)
            two = asyncio.create_task(run(second))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert hooks.calls == calls, "second loop overlapped a live runtime transaction"
            hooks.fault = None
            hooks.training_release.set()
            results = await asyncio.wait_for(asyncio.gather(one, two), 2)
            assert [r.incumbent_id for r in results] == ["p0", "p1"]
            assert [r.promoted_id for r in results] == ["p1", "p2"]
        finally:
            hooks.training_release.set()
            await asyncio.gather(*[t for t in (one, two) if t is not None], return_exceptions=True)
    asyncio.run(case())


@pytest.mark.parametrize("fault", ["discard", "publish", "train"])
def test_sibling_loop_cannot_bypass_unknown_runtime(fault):
    async def case():
        hooks = BoundaryHooks(fault=None if fault == "publish" else fault)
        hooks.mode = "after-error" if fault == "publish" else "success"
        first, sibling = loop(hooks), loop(hooks)
        with pytest.raises((RuntimeError, OSError)):
            await run(first)
        hooks.fault = None
        hooks.mode = "success"
        await fenced(sibling, hooks)
    asyncio.run(case())


@pytest.mark.parametrize("fault", ["discard", "commit-after"])
def test_reopened_journal_fences_fresh_hooks_before_lookup(tmp_path, fault):
    async def case():
        path = tmp_path / "round.sqlite"
        signer = trust()
        hooks = BoundaryHooks(fault="discard" if fault == "discard" else None, signer=signer)
        durable = journal(path, LostAckJournal if fault == "commit-after" else PromotionJournal)
        with pytest.raises((RuntimeError, OSError)):
            await run(loop(hooks, durable))
        fresh = BoundaryHooks(signer=signer)
        if fault == "commit-after":
            fresh.active = "p1"
            fresh.digests["p1"] = digest("weights-p1")
            fresh.scores["p1"] = 0.6
        await fenced(loop(fresh, journal(path)), fresh)
    asyncio.run(case())


def test_distinct_hooks_same_journal_cannot_enter_during_training(tmp_path):
    async def case():
        path = tmp_path / "round.sqlite"
        first_hooks = BoundaryHooks(fault="pause", signer=trust())
        other_hooks = BoundaryHooks(signer=trust(nonce="second-intent-0002"))
        first = loop(first_hooks, journal(path))
        other = loop(other_hooks, journal(path))
        task = asyncio.create_task(run(first))
        try:
            await asyncio.wait_for(first_hooks.training_entered.wait(), 2)
            await fenced(other, other_hooks)
        finally:
            first_hooks.training_release.set()
            await asyncio.wait_for(task, 2)
    asyncio.run(case())


def crash_in_training(path, marker):
    class CrashHooks(BoundaryHooks):
        async def train_candidate(self, incumbent_id, experiences):
            Path(marker).write_text("entered optimizer boundary\n")
            os._exit(41)
    hooks = CrashHooks(signer=trust())
    asyncio.run(run(loop(hooks, journal(path))))


def test_process_death_before_promotion_intent_fences_restart(tmp_path):
    path, marker = tmp_path / "round.sqlite", tmp_path / "entered.txt"
    journal(path)
    process = multiprocessing.get_context("spawn").Process(
        target=crash_in_training, args=(str(path), str(marker)),
    )
    process.start()
    try:
        process.join(8)
        assert process.exitcode == 41
        assert marker.read_text() == "entered optimizer boundary\n"
        fresh = BoundaryHooks(signer=trust())
        asyncio.run(fenced(loop(fresh, journal(path)), fresh))
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2)


@pytest.mark.parametrize("durable", [False, True])
def test_success_control_preserves_publication(tmp_path, durable):
    hooks = BoundaryHooks(signer=trust() if durable else None)
    control = loop(hooks, journal(tmp_path / "round.sqlite") if durable else None)
    result = asyncio.run(run(control))
    assert result.decision.accepted and result.promoted_id == "p1"
    assert hooks.active == "p1"
