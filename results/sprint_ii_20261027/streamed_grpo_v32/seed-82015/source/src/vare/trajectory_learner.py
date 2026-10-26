"""Small durable-ingestion adapter to the existing CapabilityLoop transaction.

The model, optimizer, evaluation, publication and restoration remain hook/
PromotionJournal responsibilities. An unresolved delivery never expires into
automatic retraining. Operator recovery first verifies backend state and
reconciles the existing journal; this adapter only quarantines uncertain input.
The caller supplies the expected task-family distribution; it is never inferred
from the batch being admitted.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
from uuid import uuid4

from .durable import FenceError
from .evidence import sha256_file, sha256_json
from .promotion_journal import PromotionRecoveryRequired
from .trajectory_admission import admit_terminal_grpo_group
from .trajectory_store import identity, integer, sha
from .trajectory_workers import owned_io


class TrajectoryLearnerAdapter:
    def __init__(self, store, *, trust_root, current_policy, verifier_identity,
                 reference_families: Sequence[str], max_groups=1, max_members=256):
        if type(max_groups) is not int or not 1 <= max_groups <= 64:
            raise ValueError("invalid bounded group batch")
        if type(max_members) is not int or not 2 <= max_members <= 1024:
            raise ValueError("invalid bounded member batch")
        if (not isinstance(reference_families, Sequence)
                or isinstance(reference_families, (str, bytes))
                or not reference_families
                or any(not isinstance(family, str) or not family.strip()
                       for family in reference_families)):
            raise ValueError("streamed learner requires an independent task-family reference")
        if store.config()["run_id"] != trust_root.run_id:
            raise ValueError("foreign reward trust root")
        self.store, self.root = store, trust_root
        self.current_policy = current_policy
        self.reference_families = tuple(reference_families)
        self.max_groups, self.max_members = max_groups, max_members
        with store.connection(True) as db:
            current = db.execute("SELECT value FROM meta WHERE key='learner_verifier'").fetchone()
            if current is None:
                self._activate_verifier(db, verifier_identity)
            elif tuple(json.loads(current[0])) != tuple(verifier_identity):
                raise ValueError("use the enrolled active verifier; refresh explicitly")

    def _activate_verifier(self, db, verifier_identity):
        name, version, config = verifier_identity
        identity(name)
        integer(version, 0, 2**53)
        sha(config)
        row = db.execute("SELECT config FROM verifiers WHERE name=? AND version=?", (name, version)).fetchone()
        if row is None or row[0] != config:
            raise ValueError("active verifier must be registered")
        current = db.execute("SELECT value FROM meta WHERE key='learner_verifier'").fetchone()
        if current is not None:
            old = json.loads(current[0])
            if list(verifier_identity) == old:
                return False
            if name != old[0] or type(version) is not int or version <= old[1]:
                raise ValueError("verifier refresh must advance the enrolled lineage")
        if db.execute("SELECT 1 FROM trajectories WHERE state='publishing' LIMIT 1").fetchone():
            raise FenceError("publication owns verifier epoch; reconcile before refresh")
        db.execute("INSERT OR REPLACE INTO meta VALUES('learner_verifier',?)", (json.dumps(list(verifier_identity)),))
        self.store.event(db, "learner_verifier_activated", "stream-learner", 0, {"identity": list(verifier_identity)})
        return True

    def activate_verifier(self, verifier_identity):
        """Linearize refresh against source publication, not a mutable callback."""
        with self.store.connection(True) as db:
            return self._activate_verifier(db, verifier_identity)

    def current_verifier(self):
        with self.store.connection() as db:
            return tuple(json.loads(db.execute("SELECT value FROM meta WHERE key='learner_verifier'").fetchone()[0]))

    @staticmethod
    def _pending(db):
        row = db.execute("SELECT event FROM events WHERE json_extract(event,'$.job_id')='stream-learner' AND json_extract(event,'$.kind') IN ('learner_batch_reserved','learner_batch_terminal','learner_batch_reconciled') ORDER BY sequence DESC LIMIT 1").fetchone()
        if row is None:
            return None
        event = json.loads(row[0])
        return event["details"] if event["kind"] == "learner_batch_reserved" else None

    def pending(self):
        with self.store.connection() as db:
            return self._pending(db)

    def _admit(self, group_id, state="verified"):
        name, version, config = self.current_verifier()
        return admit_terminal_grpo_group(
            self.store, group_id, active_policy=self.current_policy(),
            verifier_name=name, verifier_version=version, verifier_config_sha256=config,
            trust_root=self.root, required_state=state,
        )

    def authorizes(self, experience):
        """Recheck signed trace, membership and current versions at use time."""
        try:
            ident = experience.attempt.metadata["trajectory_id"]
            row = self.store.trajectory(ident)
            delivery = self.pending()
            if delivery is None or delivery["groups"].get(row["group_id"], {}).get(ident) != row["head"]:
                return False
            result = self._admit(row["group_id"], "learning")
            if not result.admitted:
                return False
            expected = next(x for x in result.experiences if x.attempt.metadata["trajectory_id"] == ident)
            return (asdict(expected.attempt) == asdict(experience.attempt)
                    and asdict(expected.verification) == asdict(experience.verification))
        except (ValueError, TypeError, KeyError, AttributeError, StopIteration):
            return False

    def _clean_runtime(self, loop, *, owner_id=None):
        if loop.promotion_journal is None:
            raise ValueError("streamed learning requires the existing durable promotion journal")
        loop.runtime_coordinator.assert_usable()
        if loop.runtime_coordinator.lock.locked() and owner_id is None:
            raise PromotionRecoveryRequired("streamed runtime has a live owner")
        policy = self.current_policy()
        if loop.hooks.active_policy() != (policy.policy_id, policy.version):
            raise PromotionRecoveryRequired("streamed checkpoint and runtime identities differ")
        status = loop.promotion_journal.status()
        if status["round_owner_id"] != owner_id:
            raise PromotionRecoveryRequired("reconcile the existing runtime owner first")
        loop.promotion_journal.assert_clean(policy_id=policy.policy_id, checkpoint_sha256=policy.checkpoint_sha256)
        return policy

    def claim(self, loop, round_index, *, owner_id=None):
        if owner_id is None or not loop.runtime_coordinator.lock.locked():
            raise PromotionRecoveryRequired("acquire runtime ownership before delivery reservation")
        policy = self._clean_runtime(loop, owner_id=owner_id)
        if len(loop.replay):
            raise ValueError("streamed learner must own an empty replay before each delivery")
        if self.pending() is not None:
            raise PromotionRecoveryRequired("unresolved streamed delivery; do not retrain automatically")
        batch, groups = [], {}
        for group_id in self.store.verified_groups(self.max_groups):
            result = self._admit(group_id)
            if not result.admitted:
                self.store.close_group(group_id, result.heads, reason="quarantined",
                                       decision={"reasons": list(result.reasons)})
                continue
            budget = min(self.max_members, loop.config.replay_batch_size, loop.config.replay.capacity)
            if len(result.experiences) > budget:
                raise ValueError("complete group exceeds configured learner batch; increase the explicit budget")
            if len(batch) + len(result.experiences) > budget:
                break
            batch.extend(result.experiences)
            groups[group_id] = result.heads
        if not batch:
            return None
        reservation = {"batch_id": uuid4().hex, "round_index": round_index, "groups": groups,
                       "runtime_owner_id": owner_id,
                       "verifier_identity": list(self.current_verifier()),
                       "incumbent": asdict(policy), "source_sha256": sha256_file(__file__),
                       "engine_sha256": sha256_file(Path(__file__).with_name("engine.py"))}
        with self.store.connection(True) as db:
            if self._pending(db) is not None:
                raise PromotionRecoveryRequired("another learner owns the delivery")
            self.store._bounded(db, reservation)
            for group_id, heads in groups.items():
                self._require_group(db, group_id, heads, "verified")
                db.execute("UPDATE trajectories SET state='learning' WHERE group_id=?", (group_id,))
            self.store.event(db, "learner_batch_reserved", "stream-learner", 0, reservation)
        return reservation, tuple(batch)

    @staticmethod
    def _require_group(db, group_id, heads, state):
        rows = db.execute("SELECT id,head,state,group_size FROM trajectories WHERE group_id=?", (group_id,)).fetchall()
        if (not rows or len(rows) != rows[0]["group_size"]
                or {row["id"]: row["head"] for row in rows} != heads
                or any(row["state"] != state for row in rows)):
            raise FenceError("streamed delivery group changed")

    def _terminal(self, reservation, outcome, *, reconcile=False):
        state = "quarantined" if reconcile else "consumed"
        with self.store.connection(True) as db:
            if self._pending(db) != reservation:
                raise PromotionRecoveryRequired("streamed delivery ownership changed")
            for group_id, heads in reservation["groups"].items():
                rows = db.execute("SELECT state FROM trajectories WHERE group_id=?", (group_id,)).fetchall()
                if reconcile and rows and all(row[0] in {"cancelled", "quarantined"} for row in rows):
                    continue  # Preserve previous cancellation/failure history.
                state_before = "publishing" if rows and all(row[0] == "publishing" for row in rows) else "learning"
                self._require_group(db, group_id, heads, state_before)
                db.execute("UPDATE trajectories SET state=?,reason=? WHERE group_id=?", (state, state, group_id))
                self.store.event(db, "group_acknowledged", group_id, 0,
                                 {"heads": heads, "state": state, "decision": outcome})
            details = {"batch_id": reservation["batch_id"], "outcome": outcome}
            self.store._bounded(db, details)
            self.store.event(db, "learner_batch_reconciled" if reconcile else "learner_batch_terminal",
                             "stream-learner", 0, details)

    def _failure(self, reservation, exc):
        with self.store.connection(True) as db:
            self.store.event(db, "learner_batch_error", "stream-learner", 0,
                             {"batch_id": reservation["batch_id"], "type": type(exc).__name__, "message": str(exc)[:512]})

    def _reserve_publication(self, reservation):
        for group in reservation["groups"]:
            if not self._admit(group, "learning").admitted:
                raise FenceError("source changed before publication reservation")
        with self.store.connection(True) as db:
            current = json.loads(db.execute("SELECT value FROM meta WHERE key='learner_verifier'").fetchone()[0])
            if self._pending(db) != reservation or current != reservation["verifier_identity"]:
                raise FenceError("delivery/verifier changed before publication reservation")
            for group, heads in reservation["groups"].items():
                self._require_group(db, group, heads, "learning")
                db.execute("UPDATE trajectories SET state='publishing' WHERE group_id=?", (group,))
            self.store.event(db, "learner_publication_reserved", "stream-learner", 0,
                             {"batch_id": reservation["batch_id"], "groups": reservation["groups"], "verifier_identity": current})

    async def advance(self, loop, *, round_index):
        """One bounded delivery; workers/verifiers can continue independently."""
        reservation = None
        try:
            async def claim(owner_id):
                nonlocal reservation
                claimed = await owned_io(self.claim, loop, round_index, owner_id=owner_id)
                if claimed is None:
                    return None
                reservation, batch = claimed
                return batch
            async def complete(result):
                policy = self.current_policy()
                assert loop.promotion_journal is not None
                loop.promotion_journal.assert_clean(policy_id=policy.policy_id, checkpoint_sha256=policy.checkpoint_sha256)
                await owned_io(self._terminal, reservation,
                           {"round_result_sha256": sha256_json(asdict(result)),
                            "candidate_id": result.candidate_id, "promoted_id": result.promoted_id,
                            "accepted": result.decision.accepted, "checkpoint": asdict(policy),
                            "optimizer_commit": "owned by backend; not inferred from ingestion acknowledgment"})
                loop.replay.prune_current(freshness_fn=lambda _: None, grouped=True)
            async def reserve_publication():
                await owned_io(self._reserve_publication, reservation)
            def delivery_pending(owner_id):
                pending = self.pending()
                return pending is not None and pending["runtime_owner_id"] == owner_id
            result = await loop.run_verified_round(claim, round_index=round_index,
                                                  on_complete=complete, reserve_publication=reserve_publication,
                                                  delivery_pending=delivery_pending,
                                                  shift_reference_families=self.reference_families)
            return result
        except BaseException as exc:
            if reservation is not None:
                await owned_io(self._failure, reservation, exc)
            raise

    def reconcile_after_backend_recovery(self, loop, *, observed_policy):
        """Operator-only, after exact backend/optimizer/RNG recovery and journal reconciliation.

        This never restores model state itself. A caller-supplied pointer alone
        cannot establish backend recovery; the trusted backend owns that proof.
        Uncertain input is quarantined, not automatically replayed.
        """
        actual = self._clean_runtime(loop)
        if actual != observed_policy:
            raise PromotionRecoveryRequired("observed backend checkpoint differs")
        reservation = self.pending()
        if reservation is None:
            loop.replay.prune_current(freshness_fn=lambda _: None, grouped=True)
            return False
        if (reservation["source_sha256"] != sha256_file(__file__)
                or reservation["engine_sha256"] != sha256_file(Path(__file__).with_name("engine.py"))):
            raise PromotionRecoveryRequired("recover with the enrolled learner/engine source")
        assert loop.promotion_journal is not None
        owner = uuid4().hex
        loop.promotion_journal.acquire_round(owner)
        try:
            loop.promotion_journal.assert_clean(policy_id=actual.policy_id, checkpoint_sha256=actual.checkpoint_sha256)
            self._terminal(reservation, {"observed_checkpoint": asdict(actual),
                                        "reason": "operator verified backend and existing journal; uncertain inputs quarantined"}, reconcile=True)
            loop.replay.prune_current(freshness_fn=lambda _: None, grouped=True)
            loop.promotion_journal.release_round(owner)
        except BaseException as exc:
            loop.runtime_coordinator.quarantine(exc)
            raise
        return True
