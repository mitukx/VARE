"""Synthetic ownership controls; these hooks do not perform model updates."""
import asyncio
from copy import deepcopy

import pytest

from vare.config import EngineConfig, PromotionConfig, ReplayConfig
from vare.engine import CapabilityLoop
from vare.promotion_journal import PromotionJournal, PromotionRecoveryRequired
from vare.trajectory_learner import TrajectoryLearnerAdapter
from vare.trajectory_store import AgentStep, PolicyCheckpoint, audit_trajectory_store
from vare.types import Task, Verification
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


def setup(tmp_path, *, fault=None, max_groups=1, reference_families=("default",)):
    store, root = group(tmp_path)
    hooks = IngestionHooks(fault=fault, signer=trust())
    hooks.digests["p0"] = P0.checkpoint_sha256
    def policy():
        return PolicyCheckpoint(hooks.active, hooks.version, hooks.digests[hooks.active])
    adapter = TrajectoryLearnerAdapter(store, trust_root=root, current_policy=policy,
        verifier_identity=("v", 0, "b" * 64), reference_families=reference_families,
        max_groups=max_groups)
    durable = PromotionJournal(tmp_path / "promotion.db", initial_policy_id="p0",
                              initial_checkpoint_sha256=P0.checkpoint_sha256)
    loop = CapabilityLoop(hooks=hooks, verifier=NoRolloutVerifier(), promotion_journal=durable,
        evidence_trust_root=hooks.signer, checkpoint_hasher=lambda pid: hooks.digests[pid],
        training_evidence_authorizer=adapter.authorizes, require_authenticated_training=True,
        config=EngineConfig(replay_batch_size=2 * max_groups,
                            replay=ReplayConfig(capacity=2 * max_groups),
                            promotion=PromotionConfig(min_primary_gain=.1, min_eval_examples=1)))
    return store, adapter, loop, hooks


def add_signed_group(store, root, *, group_id, trajectory_prefix, family, task_id):
    task = Task(task_id, f"prompt for {task_id}", family,
                {"learning_rule": "terminal_action_on_policy_grpo_v1"})
    for member in range(2):
        store.submit(f"{trajectory_prefix}{member}", task, expected_steps=2,
                     group_id=group_id, member=member, group_size=2, now=100)
    for member in range(2):
        lease = store.claim("execute", "worker", now=100)
        for index in range(2):
            generation = {
                "prompt_id": task_id, "prompt": task.prompt, "response": "yes",
                "logprob": -.1, "token_count": 1, "latency_s": 0.,
                "metadata": {
                    "model": "fixture-model", "backend": "fixture",
                    "sampling_temperature": 1.,
                    "logprob_distribution": "neutral_temperature_scaled_policy",
                    "prompt_token_ids": [0], "response_token_ids": [1],
                    "response_token_logprobs": [-.1],
                },
            }
            store.append_step(lease, AgentStep(
                index, f"{trajectory_prefix}{member}-e{index}", P0,
                {"kind": "inference"}, {}, {"cursor": index}, generation,
            ), now=100)
        store.finish(lease, now=100)
    for _ in range(2):
        lease = store.claim("verify", "verifier", now=100)
        head = store.trajectory(lease.trajectory_id)["head"]
        value = Verification(
            .5, True, 1., 0, "v", trusted=True,
            metadata={"verifier_config_sha256": "b" * 64,
                      "trajectory_sha256": head},
        )
        store.record_verification(
            lease, root.attest(lease.trajectory_id, head, value), now=100,
        )



def reserve_fixture(adapter, loop, round_index=0):
    """Inspect signed fixture input without executing a model transaction."""
    from uuid import uuid4
    async def owned():
        async with loop.runtime_coordinator.lock:
            owner = uuid4().hex
            loop.promotion_journal.acquire_round(owner)
            try:
                return adapter.claim(loop,round_index,owner_id=owner)
            finally:
                loop.promotion_journal.release_round(owner)
    return asyncio.run(owned())

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


def test_multi_group_stream_assesses_shift_after_the_complete_batch(tmp_path):
    store, adapter, loop, hooks = setup(
        tmp_path, max_groups=2, reference_families=("default", "second"),
    )
    add_signed_group(store, adapter.root, group_id="h", trajectory_prefix="u",
                     family="second", task_id="task-second")

    result = asyncio.run(adapter.advance(loop, round_index=0))

    assert result.decision.accepted and hooks.active == "p1"
    assert len(hooks.batch) == 4
    assert {item.attempt.metadata["vare_rollout_group"] for item in hooks.batch} == {"g", "h"}
    assert store.counts() == {"consumed": 4}


def test_streamed_shift_uses_reference_independent_of_the_batch(tmp_path):
    store, adapter, loop, hooks = setup(tmp_path, max_groups=2,
                                        reference_families=("default",))
    add_signed_group(store, adapter.root, group_id="h", trajectory_prefix="u",
                     family="unexpected", task_id="task-unexpected")

    with pytest.raises(RuntimeError, match="streamed batch was filtered or truncated"):
        asyncio.run(adapter.advance(loop, round_index=0))

    assert "train" not in hooks.calls and hooks.active == "p0"
    assert store.counts() == {"learning": 4}
    assert adapter.pending() is not None and audit_trajectory_store(store.path)["pass"]


def test_authentication_ignores_only_mutable_replay_selection_fields(tmp_path):
    _, adapter, loop, _ = setup(tmp_path)
    reservation, batch = reserve_fixture(adapter, loop, 0)
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
        reserve_fixture(adapter, loop, 1)
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
    assert loop.promotion_journal.status()["round_owner_id"] is not None
    # A delivery still cannot be claimed even though the model transaction ended.
    loop.replay.prune_current(freshness_fn=lambda _: None, grouped=True)
    with pytest.raises(PromotionRecoveryRequired, match="unresolved"):
        reserve_fixture(adapter, loop, 1)
    adapter._terminal = original
    current = adapter.current_policy()
    status = loop.promotion_journal.status()
    loop.promotion_journal.recover_round(owner_id=status["round_owner_id"], observed_policy_id=current.policy_id, observed_checkpoint_sha256=current.checkpoint_sha256)
    loop.runtime_coordinator.recover()
    assert adapter.reconcile_after_backend_recovery(loop, observed_policy=current)
    assert store.counts() == {"quarantined": 2}
    assert hooks.calls.count("train") == 1 and audit_trajectory_store(store.path)["pass"]


def test_invalid_source_quarantined_before_any_optimizer_call(tmp_path):
    store, adapter, loop, hooks = setup(tmp_path)
    adapter.activate_verifier(("v", 1, "e" * 64))
    assert asyncio.run(adapter.advance(loop, round_index=0)) is None
    assert "train" not in hooks.calls and store.counts() == {"quarantined": 2}
    assert audit_trajectory_store(store.path)["pass"]


def test_group_budget_error_is_explicit_and_does_not_consume(tmp_path):
    store, adapter, loop, _ = setup(tmp_path)
    from dataclasses import replace
    loop.config = replace(loop.config, replay_batch_size=1)
    with pytest.raises(ValueError, match="complete group exceeds"):
        reserve_fixture(adapter, loop, 0)
    assert store.counts() == {"verified": 2} and adapter.pending() is None


def test_reserved_state_requires_matching_audit_history(tmp_path):
    store, _, _, _ = setup(tmp_path)
    with store.connection(True) as db:
        db.execute("UPDATE trajectories SET state='learning'")
    with pytest.raises(ValueError, match="durable delivery"):
        audit_trajectory_store(store.path)


def test_duplicate_or_partial_delivery_cannot_train(tmp_path):
    _, adapter, loop, hooks = setup(tmp_path)
    _, batch = reserve_fixture(adapter, loop, 0)
    async def unused(*args):
        raise AssertionError("invalid group reached transaction completion")
    async def supply(owner):
        return [batch[0]]
    with pytest.raises(ValueError, match="incomplete"):
        asyncio.run(loop.run_verified_round(
            supply, round_index=0, on_complete=unused, reserve_publication=unused,
            delivery_pending=lambda _:True, shift_reference_families=("default",),
        ))
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
    status = loop.promotion_journal.status()
    loop.promotion_journal.recover_round(owner_id=status["round_owner_id"],observed_policy_id=P0.policy_id,observed_checkpoint_sha256=P0.checkpoint_sha256)
    loop.runtime_coordinator.recover()
    assert adapter.reconcile_after_backend_recovery(loop, observed_policy=P0)
    assert audit_trajectory_store(store.path)["pass"]


def test_failed_rejected_ack_fences_even_ordinary_rounds(tmp_path):
    from dataclasses import replace
    from vare.promotion import PromotionGate
    store, adapter, loop, hooks = setup(tmp_path)
    loop.promotion = PromotionGate(replace(loop.config.promotion,min_primary_gain=1.1),trust_root=hooks.signer)
    def lost_ack(*args, **kwargs):
        raise OSError("lost rejected ACK")
    adapter._terminal = lost_ack
    with pytest.raises(OSError,match="lost rejected ACK"):
        asyncio.run(adapter.advance(loop,round_index=0))
    assert hooks.active == "p0" and adapter.pending() is not None
    calls = list(hooks.calls)
    from vare.types import Task
    with pytest.raises(PromotionRecoveryRequired):
        asyncio.run(loop.run_round([Task("new","new")],round_index=1))
    assert hooks.calls == calls
    assert store.counts() == {"learning":2}
    assert audit_trajectory_store(store.path)["pass"]


def test_source_publication_reservation_fences_cancel_and_refresh(tmp_path):
    from vare.durable import FenceError
    store, adapter, loop, hooks = setup(tmp_path)
    publish = loop._publish_candidate
    async def race(candidate, **kwargs):
        assert store.counts() == {"publishing":2}
        assert not store.cancel("t0")  # The earlier publication reservation owns this transition.
        with pytest.raises(FenceError,match="publication owns verifier"):
            adapter.activate_verifier(("v",1,"e"*64))
        assert audit_trajectory_store(store.path)["pass"]
        return await publish(candidate,**kwargs)
    loop._publish_candidate = race
    result = asyncio.run(adapter.advance(loop,round_index=0))
    assert result.decision.accepted and hooks.active=="p1"
    assert store.counts()=={"consumed":2} and audit_trajectory_store(store.path)["pass"]
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM events WHERE json_extract(event,'$.kind')='cancellation_deferred'").fetchone()[0]==1


def test_delivery_claim_occurs_under_both_runtime_fences(tmp_path):
    store, adapter, loop, hooks = setup(tmp_path)
    original = adapter.claim
    observed = []
    def owned_claim(*args, **kwargs):
        assert loop.runtime_coordinator.lock.locked()
        assert kwargs.get("owner_id") is not None
        assert loop.promotion_journal.status()["round_owner_id"]==kwargs["owner_id"]
        observed.append(True)
        return original(*args, **kwargs)
    adapter.claim = owned_claim
    result = asyncio.run(adapter.advance(loop,round_index=0))
    assert result.decision.accepted and observed==[True]
    assert store.counts()=={"consumed":2}


def test_unowned_claim_cannot_change_source_state(tmp_path):
    store, adapter, loop, hooks = setup(tmp_path)
    with pytest.raises(PromotionRecoveryRequired,match="before delivery"):
        adapter.claim(loop,0)
    assert adapter.pending() is None and store.counts()=={"verified":2}


def test_pre_reservation_budget_failure_releases_both_fences(tmp_path):
    from dataclasses import replace
    store, adapter, loop, hooks = setup(tmp_path)
    loop.config = replace(loop.config,replay_batch_size=1)
    with pytest.raises(ValueError,match="complete group exceeds"):
        asyncio.run(adapter.advance(loop,round_index=0))
    assert adapter.pending() is None and store.counts()=={"verified":2}
    assert loop.runtime_coordinator.failure is None
    assert loop.promotion_journal.status()["round_owner_id"] is None
    loop.config = replace(loop.config,replay_batch_size=2)
    assert asyncio.run(adapter.advance(loop,round_index=1)).decision.accepted


def test_cancelled_idle_poll_releases_both_fences(tmp_path):
    import threading
    store, adapter, loop, hooks = setup(tmp_path)
    store.cancel("t0")
    original = adapter.claim
    entered, release = threading.Event(), threading.Event()
    def idle_claim(*args,**kwargs):
        result = original(*args,**kwargs)
        assert result is None
        entered.set()
        assert release.wait(2)
        return result
    adapter.claim = idle_claim
    async def case():
        task=asyncio.create_task(adapter.advance(loop,round_index=0))
        try:
            assert await asyncio.to_thread(entered.wait,2)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        assert loop.runtime_coordinator.failure is None
        assert loop.promotion_journal.status()["round_owner_id"] is None
    asyncio.run(case())
