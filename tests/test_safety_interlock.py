"""CPU fault injection for the operator-controlled safety interlock."""
import asyncio
import hashlib

import pytest

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.safety import SafetyInterlock, SafetyInterlockDenied, sign_control_lease
from vare.types import Attempt, EvaluationReport, Task, Verification


KEY = bytes(range(32))
RUN = "independent-study-001"
NOW = 1_000_000


def make_fence(tmp_path, *, actions=("rollout", "train", "evaluate"),
               candidates=(), expires=NOW + 60, protected_assets=None):
    path = tmp_path / "external" / "authority.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        sign_control_lease(
            key=KEY, run_id=RUN, incumbent_id="p0", expires_at=expires,
            allowed_actions=actions, approved_candidates=candidates,
        ),
        encoding="utf-8",
    )
    return SafetyInterlock(
        path, KEY, RUN, protected_assets=protected_assets or {}, clock=lambda: NOW
    )


def test_lease_uses_exact_action_candidate_and_incumbent_scope(tmp_path):
    fence = make_fence(
        tmp_path, actions=("rollout", "train", "evaluate", "promote"),
        candidates=("p1",),
    )
    for action in ("rollout", "train", "evaluate"):
        fence.require(action, incumbent_id="p0")
    fence.require("promote", incumbent_id="p0", candidate_id="p1")
    for action, champion, candidate in (
        ("promote", "p0", "p2"), ("promote", "other", "p1"),
        ("not-a-phase", "p0", None),
    ):
        with pytest.raises(SafetyInterlockDenied):
            fence.require(action, incumbent_id=champion, candidate_id=candidate)


def test_missing_expired_malformed_and_wrong_signature_fail_closed(tmp_path):
    fence = make_fence(tmp_path)
    fence.require("train", incumbent_id="p0")
    file = fence.control_path
    content = file.read_text(encoding="utf-8")
    file.write_text(content.replace('"train"', '"promote"'))
    with pytest.raises(SafetyInterlockDenied):
        fence.require("train", incumbent_id="p0")
    file.write_text(content + "garbage")
    with pytest.raises(SafetyInterlockDenied):
        fence.require("train", incumbent_id="p0")
    file.unlink()
    with pytest.raises(SafetyInterlockDenied):
        fence.require("train", incumbent_id="p0")
    expired = make_fence(tmp_path, expires=NOW)
    with pytest.raises(SafetyInterlockDenied):
        expired.require("train", incumbent_id="p0")


def test_protected_evaluator_mutation_and_symlink_are_rejected(tmp_path):
    source = tmp_path / "grader.py"
    source.write_bytes(b"trusted grader")
    pinned = hashlib.sha256(source.read_bytes()).hexdigest()
    fence = make_fence(tmp_path, protected_assets={source: pinned})
    fence.require("evaluate", incumbent_id="p0")
    source.write_bytes(b"rewritten grader")
    with pytest.raises(SafetyInterlockDenied):
        fence.require("evaluate", incumbent_id="p0")
    source.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"trusted grader")
    source.symlink_to(outside)
    with pytest.raises(SafetyInterlockDenied):
        fence.require("evaluate", incumbent_id="p0")


class Verifier:
    name = "fixture"
    version = 0
    trusted = True
    active_version = 0

    async def verify(self, attempt):
        return Verification(1.0, True, 1.0, 0, self.name, trusted=True)


class Hooks:
    def __init__(self, *, after_train=None, after_eval=None):
        self.trained = 0
        self.promoted = []
        self.discarded = []
        self.after_train = after_train
        self.after_eval = after_eval
        self.active_id = "p0"

    def active_policy(self):
        return self.active_id, int(self.active_id != "p0")

    async def rollout(self, task, policy_id, policy_version, step):
        return Attempt(task, "1", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id, experiences):
        self.trained += 1
        if self.after_train:
            self.after_train()
        return "p1"

    async def evaluate(self, policy_id):
        if policy_id == "p1" and self.after_eval:
            self.after_eval()
        return EvaluationReport(policy_id, 0.7 if policy_id == "p1" else 0.5, n=10)

    async def promote(self, candidate_id):
        self.promoted.append(candidate_id)
        self.active_id = candidate_id

    async def discard(self, candidate_id):
        self.discarded.append(candidate_id)


def run_loop(fence, hooks, ledger=None):
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=Verifier(),
        safety_interlock=fence,
        decision_ledger=ledger,
        config=EngineConfig(
            replay_batch_size=1,
            promotion=PromotionConfig(min_primary_gain=0.01, min_eval_examples=1),
        ),
    )
    return asyncio.run(loop.run_round([Task("task1", "question")], round_index=0))


def test_promotion_requires_operator_approval_and_discard(tmp_path):
    fence = make_fence(tmp_path)
    hooks = Hooks()
    result = run_loop(fence, hooks)
    assert result.decision.reasons == ("safety_interlock_denied",)
    assert not result.decision.accepted
    assert hooks.trained == 1
    assert hooks.discarded == ["p1"]
    assert hooks.promoted == []


def test_exact_candidate_authorized_promotes(tmp_path):
    fence = make_fence(
        tmp_path, actions=("rollout", "train", "evaluate", "promote"), candidates=("p1",)
    )
    hooks = Hooks()
    result = run_loop(fence, hooks)
    assert result.decision.accepted
    assert result.promoted_id == "p1"
    assert hooks.promoted == ["p1"]


def test_training_blocked_when_operator_revokes_train(tmp_path):
    fence = make_fence(tmp_path, actions=("rollout",))
    hooks = Hooks()
    with pytest.raises(SafetyInterlockDenied):
        run_loop(fence, hooks)
    assert hooks.trained == 0
    assert hooks.promoted == []


def test_revocation_after_training_discards_candidate(tmp_path):
    fence = make_fence(tmp_path)
    hooks = Hooks(after_train=lambda: fence.control_path.unlink())
    with pytest.raises(SafetyInterlockDenied):
        run_loop(fence, hooks)
    assert hooks.discarded == ["p1"]
    assert hooks.promoted == []


def test_ledger_precommit_failure_blocks_promotion(tmp_path):
    class BrokenLedger:
        def append(self, name, payload):
            raise OSError("ledger unavailable")

    fence = make_fence(
        tmp_path, actions=("rollout", "train", "evaluate", "promote"), candidates=("p1",)
    )
    hooks = Hooks()
    with pytest.raises(OSError, match="ledger unavailable"):
        run_loop(fence, hooks, BrokenLedger())
    assert hooks.promoted == []
    assert hooks.discarded == ["p1"]


def test_evaluator_mutated_during_evaluation_aborts_and_discards(tmp_path):
    asset = tmp_path / "grader.py"
    asset.write_bytes(b"verified-evaluator")
    pinned = hashlib.sha256(asset.read_bytes()).hexdigest()
    fence = make_fence(tmp_path, protected_assets={asset: pinned})
    hooks = Hooks(after_eval=lambda: asset.write_bytes(b"tampered-evaluator"))
    with pytest.raises(SafetyInterlockDenied):
        run_loop(fence, hooks)
    assert hooks.discarded == ["p1"]
    assert hooks.promoted == []


def test_successful_ledger_precommit_precedes_promotion(tmp_path):
    events = []

    class Ledger:
        def append(self, name, payload):
            events.append(name)

    class OrderHooks(Hooks):
        async def promote(self, candidate_id):
            events.append("promote")
            await super().promote(candidate_id)

    fence = make_fence(
        tmp_path, actions=("rollout", "train", "evaluate", "promote"), candidates=("p1",)
    )
    result = run_loop(fence, OrderHooks(), Ledger())
    assert result.decision.accepted
    assert events == ["promotion_intent", "promote", "promotion_decision"]


@pytest.mark.parametrize("phase", ["pre", "post"])
@pytest.mark.parametrize("discard_fails", [False, True])
def test_revocation_rollback_owns_discard_until_terminal(tmp_path, phase, discard_fails):
    """Revocation and repeated cancellation must not unlock partial rollback."""
    async def run():
        fence = make_fence(tmp_path)
        started = asyncio.Event()
        release = asyncio.Event()
        discard_error = OSError("restore unavailable")

        class BlockingHooks(Hooks):
            def __init__(self):
                super().__init__(
                    after_train=fence.control_path.unlink if phase == "pre" else None,
                    after_eval=fence.control_path.unlink if phase == "post" else None,
                )
                self.evaluated = []
                self.rollouts = 0
                self.candidate_exists = False

            async def rollout(self, *args):
                self.rollouts += 1
                return await super().rollout(*args)

            async def train_candidate(self, *args):
                result = await super().train_candidate(*args)
                self.candidate_exists = True
                return result

            async def evaluate(self, policy_id):
                self.evaluated.append(policy_id)
                return await super().evaluate(policy_id)

            async def discard(self, candidate_id):
                started.set()
                await release.wait()
                if discard_fails:
                    raise discard_error
                self.candidate_exists = False
                await super().discard(candidate_id)

        hooks = BlockingHooks()
        loop = CapabilityLoop(
            hooks=hooks, verifier=Verifier(), safety_interlock=fence,
            config=EngineConfig(replay_batch_size=1),
        )
        task = asyncio.create_task(loop.run_round([Task("t", "prompt")], round_index=0))
        next_round = None
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert hooks.evaluated == ([] if phase == "pre" else ["p0", "p1"])
            next_round = asyncio.create_task(
                loop.run_round([Task("next", "prompt")], round_index=1)
            )
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done(), "cancellation abandoned candidate restoration"
            assert loop._round_lock.locked()
            assert hooks.candidate_exists
            assert not next_round.done()
            assert hooks.rollouts == 1, "next round entered before rollback completed"
            release.set()
            if discard_fails:
                with pytest.raises(RuntimeError, match="policy state is unknown") as caught:
                    await task
                assert caught.value.__cause__ is discard_error
                assert hooks.candidate_exists
            else:
                with pytest.raises(SafetyInterlockDenied):
                    await task
                assert not hooks.candidate_exists
                assert hooks.discarded == ["p1"]
            assert not hooks.promoted
        finally:
            release.set()
            if next_round is not None:
                next_round.cancel()
            await asyncio.gather(task, *([next_round] if next_round else []),
                                 return_exceptions=True)

    asyncio.run(run())
