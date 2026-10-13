"""CPU publication faults: runtime acknowledgment and same-loop recovery fence."""
import asyncio
import hashlib

import pytest

from vare.attestation import EvidenceTrustRoot
from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.promotion_journal import PromotionJournal, PromotionRecoveryRequired
from vare.types import Attempt, EvaluationReport, Task, Verification


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Verifier:
    active_version = 0

    async def verify(self, attempt):
        return Verification(1.0, True, 1.0, 0, "fixture", trusted=True)


class Hooks:
    def __init__(self, *, mode="success", signer=None):
        self.mode = mode
        self.signer = signer
        self.active = "p0"
        self.version = 0
        self.serial = 0
        self.calls = []
        self.digests = {"p0": digest("weights-p0")}
        self.scores = {"p0": 0.4}
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    def active_policy(self):
        self.calls.append("active-policy")
        return self.active, self.version

    async def rollout(self, task, policy_id, policy_version, step):
        self.calls.append("rollout")
        return Attempt(task, "answer", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id, experiences):
        self.calls.append("train")
        self.serial += 1
        candidate = f"p{self.serial}"
        self.digests[candidate] = digest("weights-" + candidate)
        self.scores[candidate] = self.scores[incumbent_id] + 0.2
        return candidate

    async def evaluate(self, policy_id):
        self.calls.append("evaluate-" + policy_id)
        report = EvaluationReport(policy_id, self.scores[policy_id], n=10)
        return self.signer.attest(report, checkpoint_sha256=self.digests[policy_id]) if self.signer else report

    async def promote(self, candidate_id):
        self.calls.append("promote")
        if self.mode == "before-error":
            raise OSError("publication unavailable")
        if self.mode == "before-cancel":
            raise asyncio.CancelledError("publication cancelled")
        if self.mode == "noop":
            return
        self.active = "foreign" if self.mode == "wrong-policy" else candidate_id
        self.version += 1
        if self.mode == "after-error":
            raise OSError("partial publication")
        if self.mode == "after-cancel":
            raise asyncio.CancelledError("partial publication cancelled")
        if self.mode == "checkpoint-mismatch":
            self.digests[candidate_id] = digest("different installed weights")
        if self.mode == "caller-cancel":
            self.entered.set()
            try:
                await self.release.wait()
            finally:
                self.finished.set()

    async def discard(self, candidate_id):
        self.calls.append("discard")


class FaultyJournal(PromotionJournal):
    def __init__(self, path, *, fault=None):
        super().__init__(path, initial_policy_id="p0", initial_checkpoint_sha256=digest("weights-p0"))
        self.fault = fault

    def commit(self, tx_id):
        if self.fault == "commit-before":
            raise OSError("journal write unavailable")
        super().commit(tx_id)
        if self.fault == "commit-after":
            raise OSError("lost journal commit acknowledgment")


def make_loop(tmp_path, *, mode="success", durable=False, journal_fault=None):
    signer = EvidenceTrustRoot(
        key=bytes(range(32)), run_id="publication-fixture-run-01",
        protocol_sha256=digest("protocol"), evaluator_sha256=digest("grader"),
        evaluation_set_sha256=digest("held-out fixture"),
        promotion_nonce="publication-intent-0001", strict_checkpoint_binding=True,
    ) if durable else None
    hooks = Hooks(mode=mode, signer=signer)
    journal = FaultyJournal(tmp_path / "publication.sqlite", fault=journal_fault) if durable else None
    loop = CapabilityLoop(
        hooks=hooks, verifier=Verifier(), evidence_trust_root=signer,
        promotion_journal=journal,
        checkpoint_hasher=(lambda pid: hooks.digests[pid]) if durable else None,
        config=EngineConfig(replay_batch_size=1,
                            promotion=PromotionConfig(min_primary_gain=0.1, min_eval_examples=1)),
    )
    return loop, hooks, journal


async def assert_reentry_fenced(loop, hooks):
    calls = list(hooks.calls)
    step = loop.step
    error = None
    try:
        await loop.run_round([Task("next", "next prompt")], round_index=1)
    except BaseException as exc:
        error = exc
    assert hooks.calls == calls, "failed publication permitted another hook access"
    assert loop.step == step
    assert isinstance(error, PromotionRecoveryRequired)
    assert "promotion recovery required" in str(error)
    assert not loop._round_lock.locked()


@pytest.mark.parametrize("mode", ["before-error", "after-error", "before-cancel", "after-cancel"])
def test_publication_fault_without_journal_fences_reentry(tmp_path, mode):
    async def case():
        loop, hooks, _ = make_loop(tmp_path, mode=mode)
        with pytest.raises((OSError, asyncio.CancelledError, PromotionRecoveryRequired)):
            await loop.run_round([Task("t", "prompt")], round_index=0)
        assert loop.events.counters["candidate_promoted"] == 0
        assert loop.events.counters["round_complete"] == 0
        hooks.mode = "success"
        await assert_reentry_fenced(loop, hooks)
        assert "discard" not in hooks.calls

    asyncio.run(case())


@pytest.mark.parametrize("mode", ["noop", "wrong-policy"])
def test_hook_success_requires_candidate_runtime_acknowledgment(tmp_path, mode):
    async def case():
        loop, hooks, _ = make_loop(tmp_path, mode=mode)
        error = None
        result = None
        try:
            result = await loop.run_round([Task("t", "prompt")], round_index=0)
        except PromotionRecoveryRequired as exc:
            error = exc
        assert result is None, "unacknowledged candidate was reported as promoted"
        assert error is not None
        assert "policy" in str(error)
        assert loop.events.counters["candidate_promoted"] == 0
        assert loop.events.counters["round_complete"] == 0
        await assert_reentry_fenced(loop, hooks)
        assert "discard" not in hooks.calls

    asyncio.run(case())


@pytest.mark.parametrize("fault", ["checkpoint-mismatch", "commit-before", "commit-after"])
def test_durable_publication_completion_fault_fences_before_policy_lookup(tmp_path, fault):
    async def case():
        loop, hooks, journal = make_loop(
            tmp_path, durable=True,
            mode="checkpoint-mismatch" if fault == "checkpoint-mismatch" else "success",
            journal_fault=fault if fault.startswith("commit") else None,
        )
        with pytest.raises((OSError, PromotionRecoveryRequired)):
            await loop.run_round([Task("t", "prompt")], round_index=0)
        assert loop.events.counters["candidate_promoted"] == 0
        status = journal.status()
        if fault == "commit-after":
            assert status["pending_tx_id"] is None
            assert status["active_policy_id"] == "p1"
        else:
            assert status["pending_tx_id"] == "publication-intent-0001"
            assert status["active_policy_id"] == "p0"
        hooks.mode = "success"
        journal.fault = None
        await assert_reentry_fenced(loop, hooks)
        assert "discard" not in hooks.calls

    asyncio.run(case())


def test_real_caller_cancellation_during_publication_fences_reentry(tmp_path):
    async def case():
        loop, hooks, _ = make_loop(tmp_path, mode="caller-cancel")
        task = asyncio.create_task(loop.run_round([Task("t", "prompt")], round_index=0))
        try:
            await asyncio.wait_for(hooks.entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert hooks.finished.is_set()
            assert loop.events.counters["candidate_promoted"] == 0
            hooks.mode = "success"
            await assert_reentry_fenced(loop, hooks)
        finally:
            hooks.release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(case())


@pytest.mark.parametrize("durable", [False, True])
def test_valid_publication_control_commits_and_reports_success(tmp_path, durable):
    async def case():
        loop, hooks, journal = make_loop(tmp_path, durable=durable)
        result = await loop.run_round([Task("t", "prompt")], round_index=0)
        assert result.decision.accepted and result.promoted_id == hooks.active == "p1"
        assert loop.events.counters["candidate_promoted"] == 1
        assert loop.events.counters["round_complete"] == 1
        assert "discard" not in hooks.calls
        if durable:
            assert journal.status()["pending_tx_id"] is None
            journal.assert_clean(policy_id="p1", checkpoint_sha256=hooks.digests["p1"])

    asyncio.run(case())
