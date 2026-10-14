"""Caller cancellation cannot interrupt or disappear during accepted publication."""
import asyncio
from dataclasses import asdict

import pytest

from vare.promotion_journal import PromotionRecoveryRequired
from vare.types import Task
from test_promotion_publication_recovery import Hooks, make_loop


def _hold_promotion(hooks, *, fail_after_release=False, clear_cancellation=False):
    hooks.publish_entered = asyncio.Event()
    hooks.cancellation_seen = asyncio.Event()
    hooks.publish_release = asyncio.Event()
    hooks.publish_task = None

    async def promote(candidate_id):
        hooks.calls.append("promote")
        hooks.publish_task = asyncio.current_task()
        hooks.active = candidate_id
        hooks.version += 1
        hooks.publish_entered.set()
        while not hooks.publish_release.is_set():
            try:
                await hooks.publish_release.wait()
            except asyncio.CancelledError:
                # Reproduce a hook that suppresses cancellation while it owns
                # a partially installed policy. It intentionally does not
                # clear Task.cancelling(); the engine must isolate ownership.
                hooks.suppressed_cancellations += 1
                hooks.cancellation_seen.set()
                if clear_cancellation:
                    current = asyncio.current_task()
                    assert current is not None
                    current.uncancel()
        if fail_after_release:
            raise OSError("publication acknowledgment unavailable")

    hooks.suppressed_cancellations = 0
    hooks.promote = promote


def test_unjournaled_cancellation_is_not_lost_when_hook_suppresses_it(tmp_path):
    async def case():
        loop, hooks, _ = make_loop(tmp_path)
        _hold_promotion(hooks)
        owner = asyncio.create_task(
            loop.run_round([Task("t", "prompt")], round_index=0)
        )
        try:
            await asyncio.wait_for(hooks.publish_entered.wait(), 2)
            owner.cancel()
            await asyncio.sleep(0)
            assert not owner.done(), "round returned before publication acknowledgment"
            assert hooks.publish_task.cancelling() == 1
            assert loop._round_lock.locked(), "round ownership released before publication completed"
            assert loop.events.counters["candidate_promoted"] == 0
            assert loop.events.counters["round_complete"] == 0
            hooks.publish_release.set()
            with pytest.raises(asyncio.CancelledError):
                await owner
            assert hooks.active == "p1"
            assert hooks.suppressed_cancellations == 1
            assert loop.events.counters["promotion_cancellation_deferred"] == 0
            assert loop.events.counters["candidate_promoted"] == 0
            assert loop.events.counters["round_complete"] == 0
            calls = list(hooks.calls)
            with pytest.raises(PromotionRecoveryRequired):
                await loop.run_round([Task("next", "next prompt")], round_index=1)
            assert hooks.calls == calls
        finally:
            hooks.publish_release.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(case())


def test_repeated_cancellation_keeps_durable_round_owned_until_commit(tmp_path):
    async def case():
        loop, hooks, journal = make_loop(tmp_path, durable=True)
        _hold_promotion(hooks)
        owner = asyncio.create_task(
            loop.run_round([Task("t", "prompt")], round_index=0)
        )
        try:
            await asyncio.wait_for(hooks.publish_entered.wait(), 2)
            owner.cancel()
            owner.cancel()
            await asyncio.sleep(0)
            assert not owner.done(), "round returned before publication acknowledgment"
            assert hooks.publish_task.cancelling() == 0, "durable publication task was cancelled by caller"
            assert loop._round_lock.locked()
            with pytest.raises(PromotionRecoveryRequired):
                journal.acquire_round("second-owner-00000000")
            assert journal.status()["pending_tx_id"] == "publication-intent-0001"
            hooks.publish_release.set()
            result = await owner
            assert result.promoted_id == hooks.active == "p1"
            assert result.deferred_cancellation_count == 2
            assert owner.cancelling() == 0
            assert hooks.suppressed_cancellations == 0
            assert journal.status()["pending_tx_id"] is None
            journal.assert_clean(policy_id="p1", checkpoint_sha256=hooks.digests["p1"])
            assert loop.events.counters["promotion_cancellation_deferred"] == 1
            assert loop.events.counters["candidate_promoted"] == 1
        finally:
            hooks.publish_release.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(case())


def test_unjournaled_repeated_cancellation_is_delivered_to_hook_only_once(tmp_path):
    async def case():
        loop, hooks, _ = make_loop(tmp_path)
        _hold_promotion(hooks, clear_cancellation=True)
        owner = asyncio.create_task(
            loop.run_round([Task("t", "prompt")], round_index=0)
        )
        try:
            await asyncio.wait_for(hooks.publish_entered.wait(), 2)
            owner.cancel()
            await asyncio.wait_for(hooks.cancellation_seen.wait(), 2)
            assert hooks.suppressed_cancellations == 1
            owner.cancel()
            hooks.publish_release.set()
            with pytest.raises(asyncio.CancelledError):
                await owner
            assert hooks.suppressed_cancellations == 1, "cleanup received repeated cancellation"
            assert loop.events.counters["candidate_promoted"] == 0
            calls = list(hooks.calls)
            with pytest.raises(PromotionRecoveryRequired):
                await loop.run_round([Task("next", "next prompt")], round_index=1)
            assert hooks.calls == calls
        finally:
            hooks.publish_release.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(case())


def test_hook_suppressed_self_cancellation_fails_closed(tmp_path):
    async def case():
        loop, hooks, _ = make_loop(tmp_path)

        async def self_cancelling_promote(candidate_id):
            hooks.calls.append("promote")
            hooks.publish_task = asyncio.current_task()
            hooks.active = candidate_id
            hooks.version += 1
            hooks.publish_task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass

        hooks.promote = self_cancelling_promote
        with pytest.raises(asyncio.CancelledError):
            await loop.run_round([Task("t", "prompt")], round_index=0)
        assert loop.events.counters["candidate_promoted"] == 0
        assert loop.events.counters["round_complete"] == 0
        calls = list(hooks.calls)
        with pytest.raises(PromotionRecoveryRequired):
            await loop.run_round([Task("next", "next prompt")], round_index=1)
        assert hooks.calls == calls

    asyncio.run(case())


def test_durable_hook_self_cancellation_retains_prepared_recovery_state(tmp_path):
    async def case():
        loop, hooks, journal = make_loop(tmp_path, durable=True)

        async def self_cancelling_promote(candidate_id):
            hooks.calls.append("promote")
            hooks.publish_task = asyncio.current_task()
            hooks.active = candidate_id
            hooks.version += 1
            hooks.publish_task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass

        hooks.promote = self_cancelling_promote
        with pytest.raises(PromotionRecoveryRequired):
            await loop.run_round([Task("t", "prompt")], round_index=0)
        status = journal.status()
        assert status["pending_tx_id"] == "publication-intent-0001"
        assert status["round_owner_id"] is not None
        assert loop.events.counters["candidate_promoted"] == 0
        assert loop.events.counters["round_complete"] == 0
        calls = list(hooks.calls)
        with pytest.raises(PromotionRecoveryRequired):
            await loop.run_round([Task("next", "next prompt")], round_index=1)
        assert hooks.calls == calls

    asyncio.run(case())


def test_publish_failure_after_deferred_cancellation_wins_and_keeps_fence(tmp_path):
    async def case():
        loop, hooks, journal = make_loop(tmp_path, durable=True)
        _hold_promotion(hooks, fail_after_release=True)
        owner = asyncio.create_task(
            loop.run_round([Task("t", "prompt")], round_index=0)
        )
        try:
            await asyncio.wait_for(hooks.publish_entered.wait(), 2)
            owner.cancel()
            await asyncio.sleep(0)
            assert not owner.done()
            assert loop.events.counters["candidate_promoted"] == 0
            hooks.publish_release.set()
            with pytest.raises(PromotionRecoveryRequired, match="promotion failed"):
                await owner
            assert owner.cancelling() == 0
            assert journal.status()["pending_tx_id"] == "publication-intent-0001"
            assert loop.events.counters["promotion_cancellation_deferred"] == 0
            assert loop.events.counters["candidate_promoted"] == 0
            calls = list(hooks.calls)
            with pytest.raises(PromotionRecoveryRequired):
                await loop.run_round([Task("next", "next prompt")], round_index=1)
            assert hooks.calls == calls
        finally:
            hooks.publish_release.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(case())


def test_round_result_serialization_includes_default_cancellation_count(tmp_path):
    async def case():
        loop, _, _ = make_loop(tmp_path)
        result = await loop.run_round([Task("t", "prompt")], round_index=0)
        row = asdict(result)
        assert result.deferred_cancellation_count == 0
        assert row["deferred_cancellation_count"] == 0

    asyncio.run(case())
