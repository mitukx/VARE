"""CPU-only checks for checkpoint-locked evidence and crash-recoverable promotions."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib

import pytest

from vare.attestation import EvidenceTrustRoot
from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.promotion import PromotionGate
from vare.promotion_journal import PromotionJournal, PromotionRecoveryRequired
from vare.types import Attempt, EvaluationReport, Task, Verification


OLD = hashlib.sha256(b"model weights v0").hexdigest()
NEW = hashlib.sha256(b"model weights v1").hexdigest()
OTHER = hashlib.sha256(b"untrusted model").hexdigest()
SET = hashlib.sha256(b"frozen held-out questions").hexdigest()
PROTOCOL = hashlib.sha256(b"frozen protocol").hexdigest()
GRADER = hashlib.sha256(b"protected grader").hexdigest()


def trust(*, nonce="one-time-intent-0001", evaluation_set=SET):
    return EvidenceTrustRoot(
        key=bytes(range(32)),
        protocol_sha256=PROTOCOL,
        evaluator_sha256=GRADER,
        run_id="fixture-independent-run-01",
        evaluation_set_sha256=evaluation_set,
        promotion_nonce=nonce,
        strict_checkpoint_binding=True,
    )


def test_bound_reports_reject_checkpoint_or_dataset_swap_and_nonce_reuse():
    signer = trust()
    old = signer.attest(EvaluationReport("p0", 0.5, n=10), checkpoint_sha256=OLD)
    new = signer.attest(EvaluationReport("p1", 0.8, n=10), checkpoint_sha256=NEW)
    gate = PromotionGate(
        PromotionConfig(min_eval_examples=10, min_primary_gain=0.01),
        trust_root=signer,
    )
    assert gate.decide(old, new).accepted
    assert signer.checkpoint_digest(new) == NEW
    assert not gate.decide(
        old, replace(new, metadata={**new.metadata, "vare_checkpoint_sha256": OTHER})
    ).accepted
    assert not PromotionGate(
        PromotionConfig(min_eval_examples=10), trust_root=trust(nonce="different-intent-0002")
    ).decide(old, new).accepted
    assert not PromotionGate(
        PromotionConfig(min_eval_examples=10), trust_root=trust(evaluation_set=OTHER)
    ).decide(old, new).accepted
    with pytest.raises(ValueError, match="checkpoint digest"):
        signer.attest(EvaluationReport("p1", 0.8, n=10))
    with pytest.raises(ValueError, match="binding"):
        signer.attest(
            EvaluationReport("p1", 0.8, n=10, metadata={"vare_checkpoint_sha256": NEW}),
            checkpoint_sha256=NEW,
        )


def test_journal_restart_requires_reconciliation_and_rejects_wrong_snapshot(tmp_path):
    path = tmp_path / "control" / "promotion.sqlite"
    journal = PromotionJournal(path, initial_policy_id="p0", initial_checkpoint_sha256=OLD)
    journal.assert_clean(policy_id="p0", checkpoint_sha256=OLD)
    journal.prepare(
        tx_id="nonce-for-transaction-1", incumbent_id="p0", incumbent_sha256=OLD,
        candidate_id="p1", candidate_sha256=NEW, evaluation_sha256=SET,
    )
    recovered = PromotionJournal(path, initial_policy_id="p0", initial_checkpoint_sha256=OLD)
    with pytest.raises(PromotionRecoveryRequired, match="unresolved"):
        recovered.assert_clean(policy_id="p0", checkpoint_sha256=OLD)
    with pytest.raises(PromotionRecoveryRequired, match="unknown runtime"):
        recovered.reconcile(
            tx_id="nonce-for-transaction-1", observed_policy_id="p1",
            observed_checkpoint_sha256=OTHER,
        )
    assert recovered.status()["pending_tx_id"] == "nonce-for-transaction-1"
    assert recovered.reconcile(
        tx_id="nonce-for-transaction-1", observed_policy_id="p1",
        observed_checkpoint_sha256=NEW,
    ) == "committed"
    recovered.assert_clean(policy_id="p1", checkpoint_sha256=NEW)
    with pytest.raises(PromotionRecoveryRequired, match="pointer disagree"):
        recovered.assert_clean(policy_id="p0", checkpoint_sha256=OLD)
    with pytest.raises(PromotionRecoveryRequired, match="nonce already consumed"):
        recovered.prepare(
            tx_id="nonce-for-transaction-1", incumbent_id="p1", incumbent_sha256=NEW,
            candidate_id="p2", candidate_sha256=OTHER, evaluation_sha256=SET,
        )


def test_pending_promotion_can_be_aborted_only_with_verified_incumbent(tmp_path):
    journal = PromotionJournal(tmp_path / "txn.sqlite", initial_policy_id="p0", initial_checkpoint_sha256=OLD)
    journal.prepare(
        tx_id="nonce-for-transaction-2", incumbent_id="p0", incumbent_sha256=OLD,
        candidate_id="p1", candidate_sha256=NEW, evaluation_sha256=SET,
    )
    assert journal.reconcile(
        tx_id="nonce-for-transaction-2", observed_policy_id="p0",
        observed_checkpoint_sha256=OLD,
    ) == "aborted"
    journal.assert_clean(policy_id="p0", checkpoint_sha256=OLD)
    with pytest.raises(PromotionRecoveryRequired):
        journal.reconcile(
            tx_id="nonce-for-transaction-2", observed_policy_id="p0",
            observed_checkpoint_sha256=OLD,
        )


class StubVerifier:
    name = "trusted"
    version = 0
    trusted = True
    active_version = 0

    async def verify(self, attempt):
        return Verification(1.0, True, 1.0, 0, self.name, trusted=True)


class Hooks:
    def __init__(self, signer, *, crash_after_publish=False, forged_hash=False,
                 evaluation_verifier_version=None):
        self.signer = signer
        self.active = "p0"
        self.digests = {"p0": OLD, "p1": NEW}
        self.crash_after_publish = crash_after_publish
        self.forged_hash = forged_hash
        self.evaluation_verifier_version = evaluation_verifier_version
        self.promoted = []
        self.discarded = []

    def active_policy(self):
        return self.active, int(self.active == "p1")

    async def rollout(self, task, policy_id, policy_version, step):
        return Attempt(task, "answer", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id, experiences):
        return "p1"

    async def evaluate(self, policy_id):
        metadata = ({} if self.evaluation_verifier_version is None else
                    {"verifier_version": self.evaluation_verifier_version})
        return self.signer.attest(
            EvaluationReport(policy_id, 0.8 if policy_id == "p1" else 0.5,
                             n=10, metadata=metadata),
            checkpoint_sha256=(OTHER if self.forged_hash and policy_id == "p1"
                               else self.digests[policy_id]),
        )

    async def promote(self, candidate_id):
        self.promoted.append(candidate_id)
        self.active = candidate_id
        if self.crash_after_publish:
            raise RuntimeError("injected crash after switching live champion")

    async def discard(self, candidate_id):
        self.discarded.append(candidate_id)


def _loop(hooks, journal, signer, *, ledger=None,
          require_current_evaluation_verifier=False):
    return CapabilityLoop(
        hooks=hooks,
        verifier=StubVerifier(),
        decision_ledger=ledger,
        evidence_trust_root=signer,
        promotion_journal=journal,
        checkpoint_hasher=lambda pid: hooks.digests[pid],
        require_current_evaluation_verifier=require_current_evaluation_verifier,
        config=EngineConfig(
            replay_batch_size=1,
            promotion=PromotionConfig(min_eval_examples=1, min_primary_gain=0.01),
        ),
    )


def _round(loop):
    return asyncio.run(loop.run_round([Task("task-1", "question")], round_index=0))


def test_engine_commits_verified_checkpoint_and_durable_pointer(tmp_path):
    signer = trust()
    journal = PromotionJournal(tmp_path / "durable.sqlite", initial_policy_id="p0", initial_checkpoint_sha256=OLD)
    hooks = Hooks(signer)
    result = _round(_loop(hooks, journal, signer))
    assert result.decision.accepted
    assert result.promoted_id == "p1"
    assert journal.status()["active_checkpoint_sha256"] == NEW
    journal.assert_clean(policy_id="p1", checkpoint_sha256=NEW)
    assert hooks.discarded == []


def test_engine_interrupt_after_partial_promotion_blocks_new_round(tmp_path):
    signer = trust()
    path = tmp_path / "durable.sqlite"
    journal = PromotionJournal(path, initial_policy_id="p0", initial_checkpoint_sha256=OLD)
    hooks = Hooks(signer, crash_after_publish=True)
    loop = _loop(hooks, journal, signer)
    with pytest.raises(PromotionRecoveryRequired, match="reconcile"):
        _round(loop)
    assert journal.status()["pending_tx_id"] == signer.promotion_nonce
    hooks.crash_after_publish = False
    with pytest.raises(PromotionRecoveryRequired, match="unresolved"):
        _round(loop)
    restarted = PromotionJournal(path, initial_policy_id="p0", initial_checkpoint_sha256=OLD)
    assert restarted.reconcile(
        tx_id=signer.promotion_nonce,
        observed_policy_id=hooks.active,
        observed_checkpoint_sha256=hooks.digests[hooks.active],
    ) == "committed"
    restarted.assert_clean(policy_id="p1", checkpoint_sha256=NEW)


def test_engine_refuses_signed_but_nonmatching_checkpoint(tmp_path):
    signer = trust()
    journal = PromotionJournal(tmp_path / "durable.sqlite", initial_policy_id="p0", initial_checkpoint_sha256=OLD)
    hooks = Hooks(signer, forged_hash=True)
    result = _round(_loop(hooks, journal, signer))
    assert not result.decision.accepted
    assert result.decision.reasons == ("checkpoint_attestation_mismatch",)
    assert hooks.promoted == []
    assert hooks.discarded == ["p1"]
    journal.assert_clean(policy_id="p0", checkpoint_sha256=OLD)


def test_engine_refuses_signed_candidate_evaluation_from_stale_verifier(tmp_path):
    signer = trust()
    journal = PromotionJournal(tmp_path / "durable.sqlite",
                               initial_policy_id="p0",
                               initial_checkpoint_sha256=OLD)
    hooks = Hooks(signer, evaluation_verifier_version=1)
    result = _round(_loop(hooks, journal, signer,
                          require_current_evaluation_verifier=True))
    assert not result.decision.accepted
    assert result.decision.reasons == ("evaluation_verifier_version_mismatch",)
    assert hooks.promoted == []
    assert hooks.discarded == ["p1"]
    journal.assert_clean(policy_id="p0", checkpoint_sha256=OLD)


def test_engine_requires_evaluation_verifier_identity_when_strict(tmp_path):
    signer = trust()
    journal = PromotionJournal(tmp_path / "durable.sqlite",
                               initial_policy_id="p0",
                               initial_checkpoint_sha256=OLD)
    hooks = Hooks(signer)
    result = _round(_loop(hooks, journal, signer,
                          require_current_evaluation_verifier=True))
    assert not result.decision.accepted
    assert result.decision.reasons == ("evaluation_verifier_version_mismatch",)
    assert hooks.promoted == []
    assert hooks.discarded == ["p1"]
    journal.assert_clean(policy_id="p0", checkpoint_sha256=OLD)


def test_ledger_precommit_error_reconciles_prepared_intent_as_abort(tmp_path):
    class BrokenLedger:
        def append(self, event, payload):
            raise OSError("simulated full disk")

    signer = trust()
    journal = PromotionJournal(tmp_path / "durable.sqlite", initial_policy_id="p0", initial_checkpoint_sha256=OLD)
    hooks = Hooks(signer)
    with pytest.raises(OSError, match="full disk"):
        _round(_loop(hooks, journal, signer, ledger=BrokenLedger()))
    assert hooks.promoted == []
    assert hooks.discarded == ["p1"]
    journal.assert_clean(policy_id="p0", checkpoint_sha256=OLD)


def test_engine_constructor_requires_strict_signed_evidence(tmp_path):
    journal = PromotionJournal(tmp_path / "durable.sqlite", initial_policy_id="p0", initial_checkpoint_sha256=OLD)
    with pytest.raises(ValueError, match="strict attestation"):
        CapabilityLoop(
            hooks=Hooks(trust()), verifier=StubVerifier(),
            promotion_journal=journal,
            checkpoint_hasher=lambda pid: OLD,
        )
