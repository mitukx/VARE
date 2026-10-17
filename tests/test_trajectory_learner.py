"""Synthetic ownership controls; these hooks do not perform model updates."""
import asyncio
from copy import deepcopy

import pytest

from vare.config import EngineConfig, PromotionConfig, ReplayConfig
from vare.engine import CapabilityLoop
from vare.promotion_journal import PromotionJournal, PromotionRecoveryRequired
from vare.trajectory_learner import TrajectoryLearnerAdapter
from vare.trajectory_store import PolicyCheckpoint, audit_trajectory_store
from test_checkpoint_bound_promotion import trust
from test_trajectory_admission import group, P0
from test_runtime_round_ownership import BoundaryHooks


class NoRolloutVerifier:
    active_version = 0

    async def verify(self, attempt):
        raise AssertionError("ingestion reverified a signed source")


class IngestionHooks(BoundaryHooks):
    async def rollout(self, *args):
        raise AssertionError("ingestion regenerated a stored trajectory")

    async def train_candidate(self, incumbent_id, experiences):
        self.batch = deepcopy(experiences)
        return await super().train_candidate(incumbent_id, experiences)


def setup(tmp_path, *, fault=None):
    store, root = group(tmp_path)
    hooks = IngestionHooks(fault=fault, signer=trust())
    hooks.digests["p0"] = P0.checkpoint_sha256
    policy = lambda: PolicyCheckpoint(hooks.active, hooks.version, hooks.digests[hooks.active])
    adapter = TrajectoryLearnerAdapter(store, trust_root=root, current_policy=policy,
        current_verifier=lambda: ("v", 0, "b" * 64))
    durable = PromotionJournal(tmp_path / "promotion.db", initial_policy_id="p0",
                              initial_checkpoint_sha256=P0.checkpoint_sha256)
    loop = CapabilityLoop(hooks=hooks, verifier=NoRolloutVerifier(), promotion_journal=durable,
        evidence_trust_root=hooks.signer, checkpoint_hasher=lambda pid: hooks.digests[pid],
        training_evidence_authorizer=adapter.authorizes, require_authenticated_training=True,
        config=EngineConfig(replay_batch_size=2, replay=ReplayConfig(capacity=2),
                            promotion=PromotionConfig(min_primary_gain=.1, min_eval_examples=1)))
    return store, adapter, loop, hooks


def test_authenticated_delivery_uses_existing_transaction_and_ack(tmp_path):
    store, adapter, loop, hooks = setup(tmp_path)
    result = asyncio.run(adapter.advance(loop, round_index=0))
    assert result.promoted_id == "p1" and hooks.active == "p1"
    assert {x.attempt.metadata["trajectory_id"] for x in hooks.batch} == {"t0", "t1"}
    assert {x.attempt.metadata["vare_rollout_group"] for x in hooks.batch} == {"g"}
    assert adapter.pending() is None and store.counts() == {"consumed": 2}
    assert len(loop.replay) == 0 and audit_trajectory_store(store.path)["pass"]
    assert asyncio.run(adapter.advance(loop, round_index=1)) is None
    assert hooks.calls.count("train") == 1


def test_authentication_ignores_only_mutable_replay_selection_fields(tmp_path):
    _, adapter, loop, _ = setup(tmp_path)
    reservation, batch = adapter.claim(loop, 0)
    exp = deepcopy(batch[0])
    exp.priority, exp.shift_score, exp.failure_label = 99., .2, "minor"
    assert adapter.authorizes(exp)
    exp.verification.score = .9
    assert not adapter.authorizes(exp)
    exp = deepcopy(batch[0])
    exp.attempt.metadata["rvl_generation"]["response"] = "swapped"
    assert not adapter.authorizes(exp)
    assert reservation == adapter.pending()


@pytest.mark.parametrize("fault", ["train", "discard"])
def test_uncertain_delivery_survives_restart_until_backend_recovery(tmp_path, fault):
    store, adapter, loop, hooks = setup(tmp_path, fault=fault)
    with pytest.raises((OSError, RuntimeError)):
        asyncio.run(adapter.advance(loop, round_index=0))
    pending = adapter.pending()
    assert pending is not None and store.counts() == {"learning": 2}
    assert audit_trajectory_store(store.path)["pass"]
    with pytest.raises(PromotionRecoveryRequired):
        adapter.claim(loop, 1)
    with pytest.raises(PromotionRecoveryRequired):
        adapter.reconcile_after_backend_recovery(loop, observed_policy=P0)
    # Fixture operator has no tensors; real adapters must restore model/Adam/RNG.
    status = loop.promotion_journal.status()
    loop.promotion_journal.recover_round(owner_id=status["round_owner_id"],
        observed_policy_id="p0", observed_checkpoint_sha256=P0.checkpoint_sha256)
    loop._candidate_unresolved = False
    loop._discard_failure = None
    loop.runtime_coordinator.recover()
    assert adapter.reconcile_after_backend_recovery(loop, observed_policy=P0)
    assert store.counts() == {"quarantined": 2} and adapter.pending() is None
    assert audit_trajectory_store(store.path)["pass"]
    assert not adapter.reconcile_after_backend_recovery(loop, observed_policy=P0)
    assert asyncio.run(adapter.advance(loop, round_index=1)) is None


def test_unacknowledged_success_never_retrains_and_preserves_outcome(tmp_path):
    store, adapter, loop, hooks = setup(tmp_path)
    original = adapter._terminal
    adapter._terminal = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("ingestion ack failed"))
    with pytest.raises(OSError, match="ack failed"):
        asyncio.run(adapter.advance(loop, round_index=0))
    assert hooks.active == "p1" and adapter.pending() is not None
    assert loop.promotion_journal.status()["round_owner_id"] is None
    # A delivery still cannot be claimed even though the model transaction ended.
    loop.replay.prune_current(freshness_fn=lambda _: None, grouped=True)
    with pytest.raises(PromotionRecoveryRequired, match="unresolved"):
        adapter.claim(loop, 1)
    adapter._terminal = original
    current = adapter.current_policy()
    assert adapter.reconcile_after_backend_recovery(loop, observed_policy=current)
    assert store.counts() == {"quarantined": 2}
    assert hooks.calls.count("train") == 1 and audit_trajectory_store(store.path)["pass"]


def test_invalid_source_quarantined_before_any_optimizer_call(tmp_path):
    store, adapter, loop, hooks = setup(tmp_path)
    adapter.current_verifier = lambda: ("v", 1, "e" * 64)
    assert asyncio.run(adapter.advance(loop, round_index=0)) is None
    assert "train" not in hooks.calls and store.counts() == {"quarantined": 2}
    assert audit_trajectory_store(store.path)["pass"]


def test_group_budget_error_is_explicit_and_does_not_consume(tmp_path):
    store, adapter, loop, _ = setup(tmp_path)
    from dataclasses import replace
    loop.config = replace(loop.config, replay_batch_size=1)
    with pytest.raises(ValueError, match="complete group exceeds"):
        adapter.claim(loop, 0)
    assert store.counts() == {"verified": 2} and adapter.pending() is None


def test_reserved_state_requires_matching_audit_history(tmp_path):
    store, _, _, _ = setup(tmp_path)
    with store.connection(True) as db:
        db.execute("UPDATE trajectories SET state='learning'")
    with pytest.raises(ValueError, match="durable delivery"):
        audit_trajectory_store(store.path)


def test_duplicate_or_partial_delivery_cannot_train(tmp_path):
    _, adapter, loop, hooks = setup(tmp_path)
    _, batch = adapter.claim(loop, 0)
    for broken in ([batch[0]], [batch[0], batch[0]]):
        with pytest.raises(ValueError, match="incomplete|duplicated"):
            asyncio.run(loop.run_verified_round(broken, round_index=0))
    assert "train" not in hooks.calls


def test_source_cancelled_during_evaluation_cannot_promote(tmp_path):
    store, adapter, loop, hooks = setup(tmp_path)
    evaluate = hooks.evaluate
    async def cancel_then_evaluate(policy_id):
        if policy_id != "p0":
            store.cancel("t0")
        return await evaluate(policy_id)
    hooks.evaluate = cancel_then_evaluate
    with pytest.raises(Exception, match="group changed"):
        asyncio.run(adapter.advance(loop, round_index=0))
    assert hooks.active == "p0" and "promote" not in hooks.calls and "discard" in hooks.calls
    assert adapter.pending() is not None and store.counts() == {"cancelled": 2}
    assert adapter.reconcile_after_backend_recovery(loop, observed_policy=P0)
    assert audit_trajectory_store(store.path)["pass"]
