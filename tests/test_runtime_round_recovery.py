"""Maintained operator recovery checks; outside the frozen fault comparison."""
import asyncio
import gc
import multiprocessing
import os
import weakref

import pytest

from vare.promotion_journal import PromotionRecoveryRequired
from vare.runtime import RuntimeCoordinator, coordinator_for
from test_runtime_round_ownership import BoundaryHooks, fenced, journal, loop, run
from test_checkpoint_bound_promotion import trust
from test_promotion_publication_recovery import digest


def test_recovery_requires_owner_verified_pointer_and_no_pending_intent(tmp_path):
    durable = journal(tmp_path / "round.sqlite")
    owner = "round-owner-00000001"
    durable.acquire_round(owner)
    for wrong_owner, policy, checkpoint in (
        ("wrong-owner-00000002", "p0", digest("weights-p0")),
        (owner, "p0", digest("unverified")),
        (owner, "foreign", digest("weights-p0")),
    ):
        with pytest.raises(PromotionRecoveryRequired):
            durable.recover_round(owner_id=wrong_owner, observed_policy_id=policy,
                                  observed_checkpoint_sha256=checkpoint)
        assert durable.status()["round_owner_id"] == owner
    durable.prepare(tx_id="intent-nonce-00000001", incumbent_id="p0",
                    incumbent_sha256=digest("weights-p0"), candidate_id="p1",
                    candidate_sha256=digest("weights-p1"), evaluation_sha256=digest("eval"))
    with pytest.raises(PromotionRecoveryRequired, match="reconcile"):
        durable.recover_round(owner_id=owner, observed_policy_id="p0",
                              observed_checkpoint_sha256=digest("weights-p0"))
    durable.reconcile(tx_id="intent-nonce-00000001", observed_policy_id="p0",
                      observed_checkpoint_sha256=digest("weights-p0"))
    durable.recover_round(owner_id=owner, observed_policy_id="p0",
                          observed_checkpoint_sha256=digest("weights-p0"))
    durable.acquire_round("round-owner-00000002")
    with pytest.raises(PromotionRecoveryRequired, match="ownership"):
        durable.release_round(owner)
    durable.release_round("round-owner-00000002")


def test_registry_retains_fence_while_hooks_live_without_retaining_weak_hooks():
    class UnhashableHooks(BoundaryHooks):
        __hash__ = None
    hooks = UnhashableHooks()
    first = loop(hooks)
    first.runtime_coordinator.failure = OSError("unknown state")
    del first
    gc.collect()
    assert coordinator_for(hooks).failure is not None
    reference = weakref.ref(hooks)
    del hooks
    gc.collect()
    assert reference() is None


def test_explicit_coordinator_for_distinct_hook_proxies():
    async def case():
        shared = RuntimeCoordinator()
        one, two = BoundaryHooks(), BoundaryHooks()
        first, other = loop(one), loop(two)
        # Constructor parameter is the supported injection path.
        from vare.engine import CapabilityLoop
        first = CapabilityLoop(hooks=one, verifier=first.verifier, runtime_coordinator=shared)
        other = CapabilityLoop(hooks=two, verifier=other.verifier, runtime_coordinator=shared)
        assert first._round_lock is other._round_lock
        async with shared.lock:
            with pytest.raises(PromotionRecoveryRequired, match="live"):
                shared.recover()
        shared.failure = OSError("unknown state")
        await fenced(other, two)
        shared.recover()  # fixture operator has no live work or mutable tensor state
        assert shared.failure is None
    asyncio.run(case())


def test_owned_evaluation_failure_releases_after_successful_discard(tmp_path):
    class EvaluationFailure(BoundaryHooks):
        async def evaluate(self, policy_id):
            if policy_id != self.active:
                raise OSError("evaluation failed")
            return await super().evaluate(policy_id)
    hooks = EvaluationFailure(signer=trust())
    durable = journal(tmp_path / "round.sqlite")
    first = loop(hooks, durable)
    with pytest.raises(OSError, match="evaluation failed"):
        asyncio.run(run(first))
    assert durable.status()["round_owner_id"] is None
    assert first.runtime_coordinator.failure is None
    result = asyncio.run(run(loop(BoundaryHooks(signer=trust()), durable)))
    assert result.promoted_id == "p1"


def test_waiter_cancellation_does_not_quarantine_active_owner(tmp_path):
    async def case():
        hooks = BoundaryHooks(fault="pause", signer=trust())
        durable = journal(tmp_path / "round.sqlite")
        first, waiter = loop(hooks, durable), loop(hooks, durable)
        task = asyncio.create_task(run(first))
        waiting = None
        try:
            await asyncio.wait_for(hooks.training_entered.wait(), 2)
            owner = durable.status()["round_owner_id"]
            waiting = asyncio.create_task(run(waiter))
            await asyncio.sleep(0)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert durable.status()["round_owner_id"] == owner
            assert first.runtime_coordinator.failure is None
        finally:
            hooks.training_release.set()
            await asyncio.wait_for(task, 2)
    asyncio.run(case())


def crash_at_publication(path, marker, phase):
    class CrashHooks(BoundaryHooks):
        async def promote(self, candidate_id):
            await super().promote(candidate_id)
            marker.write_text(candidate_id)
            if phase == "publish":
                os._exit(42)
    from vare.promotion_journal import PromotionJournal
    class CrashCommitJournal(PromotionJournal):
        def commit(self, tx_id):
            super().commit(tx_id)
            os._exit(43)
    durable = journal(path, CrashCommitJournal if phase == "commit" else PromotionJournal)
    asyncio.run(run(loop(CrashHooks(signer=trust()), durable)))


@pytest.mark.parametrize("phase", ["publish", "commit"])
def test_process_death_recovery_requires_explicit_round_release(tmp_path, phase):
    path, marker = tmp_path / "round.sqlite", tmp_path / "actual-checkpoint.txt"
    journal(path)
    process = multiprocessing.get_context("spawn").Process(
        target=crash_at_publication, args=(path, marker, phase),
    )
    process.start()
    try:
        process.join(8)
        assert process.exitcode == (42 if phase == "publish" else 43)
        assert marker.read_text() == "p1"
        durable = journal(path)
        fresh = BoundaryHooks(signer=trust(nonce="recovered-intent-0002"))
        fresh.active, fresh.version = "p1", 1
        fresh.serial = 1
        fresh.digests["p1"], fresh.scores["p1"] = digest("weights-p1"), 0.6
        asyncio.run(fenced(loop(fresh, durable), fresh))
        status = durable.status()
        if phase == "publish":
            assert status["pending_tx_id"] == trust().promotion_nonce
            durable.reconcile(tx_id=status["pending_tx_id"], observed_policy_id="p1",
                              observed_checkpoint_sha256=digest("weights-p1"))
        else:
            assert status["pending_tx_id"] is None
        # Even a successful promotion reconciliation does not clear ownership.
        asyncio.run(fenced(loop(fresh, durable), fresh))
        durable.recover_round(owner_id=status["round_owner_id"], observed_policy_id="p1",
                              observed_checkpoint_sha256=digest("weights-p1"))
        result = asyncio.run(run(loop(fresh, durable)))
        assert result.promoted_id == "p2"
        assert durable.status()["round_owner_id"] is None
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2)


def competing_claim(path, barrier, output, owner):
    durable = journal(path)
    barrier.wait(timeout=5)
    try:
        durable.acquire_round(owner)
        output.put((owner, "acquired"))
    except PromotionRecoveryRequired:
        output.put((owner, "refused"))


def test_real_processes_cannot_both_claim_one_round(tmp_path):
    path = tmp_path / "round.sqlite"
    durable = journal(path)
    context = multiprocessing.get_context("spawn")
    barrier, output = context.Barrier(2), context.Queue()
    processes = [context.Process(target=competing_claim,
                 args=(path, barrier, output, f"process-owner-000{i}")) for i in range(2)]
    try:
        for process in processes:
            process.start()
        results = [output.get(timeout=8) for _ in processes]
        for process in processes:
            process.join(8)
            assert process.exitcode == 0
        assert sorted(result for _, result in results) == ["acquired", "refused"]
        winner = next(owner for owner, result in results if result == "acquired")
        assert durable.status()["round_owner_id"] == winner
        durable.recover_round(owner_id=winner, observed_policy_id="p0",
                              observed_checkpoint_sha256=digest("weights-p0"))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(2)
        output.close()
        output.join_thread()


def test_existing_promotion_database_retains_pointer_and_pending_intent(tmp_path):
    path = tmp_path / "round.sqlite"
    durable = journal(path)
    durable.prepare(tx_id="old-intent-00000001", incumbent_id="p0",
                    incumbent_sha256=digest("weights-p0"), candidate_id="p1",
                    candidate_sha256=digest("weights-p1"), evaluation_sha256=digest("eval"))
    # Simulate the previous schema; no promotion record is rewritten.
    with durable._connection() as db:
        db.execute("DROP TABLE runtime_round")
    reopened = journal(path)
    assert reopened.status()["active_policy_id"] == "p0"
    assert reopened.status()["pending_tx_id"] == "old-intent-00000001"
    with pytest.raises(PromotionRecoveryRequired, match="unresolved"):
        reopened.acquire_round("new-owner-00000001")


def test_lost_release_ack_quarantines_same_runtime(tmp_path):
    from vare.promotion_journal import PromotionJournal
    class LostReleaseJournal(PromotionJournal):
        def release_round(self, owner_id):
            super().release_round(owner_id)
            raise OSError("lost release acknowledgment")
    hooks = BoundaryHooks(signer=trust())
    durable = journal(tmp_path / "round.sqlite", LostReleaseJournal)
    control = loop(hooks, durable)
    with pytest.raises(PromotionRecoveryRequired, match="completion"):
        asyncio.run(run(control))
    assert durable.status()["round_owner_id"] is None
    assert durable.status()["active_policy_id"] == "p1"
    assert control.runtime_coordinator.failure is not None
    asyncio.run(fenced(loop(hooks, durable), hooks))
