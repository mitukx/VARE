"""CPU regressions: independently signed reward vs unkeyed RLVR replay metadata."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from vare.config import LagConfig
from vare.controls import RVLOuterPlanner
from vare.integrations.rvl import RVLTokenReplayReader
from vare.integrations.rvl_grpo import RVLGRPOHooks
from vare.reward_evidence import (
    RewardEvidenceManifest,
    RewardEvidenceRoot,
    validate_complete_rvl_reward_batch,
)
from test_rvl_integration import _make_replay


KEY = bytes(range(32))
PROTOCOL = hashlib.sha256(b"operator-frozen-RLVR-protocol").hexdigest()
EVALUATOR = hashlib.sha256(b"isolated-independent-grader").hexdigest()
CONFIG = "c" * 64
RUN_ID = "study-independent-review-001"


def _fixture(tmp_path, *, trusted=True, group_id="g1"):
    path = tmp_path / "replay.sqlite"
    _make_replay(path)
    with sqlite3.connect(path) as db:
        raw = db.execute("SELECT payload FROM groups WHERE id='g1'").fetchone()[0]
        members = json.loads(raw)
        for member in members:
            member["metadata"]["trusted"] = trusted
        # Update the unkeyed *behavior* digest consistently. This simulates a
        # replay writer rewriting its DB; it is not independent authentication.
        group_hash = RVLTokenReplayReader._canonical_digest([
            {k: v for k, v in member["generation"].items() if k != "latency_s"}
            for member in members
        ])
        db.execute(
            "UPDATE groups SET id=?, payload=?, hash=? WHERE id='g1'",
            (group_id, json.dumps(members), group_hash),
        )
    return RVLTokenReplayReader(path)


def _root(*, run_id=RUN_ID, evaluator=EVALUATOR, key=KEY):
    return RewardEvidenceRoot(
        key=key,
        run_id=run_id,
        protocol_sha256=PROTOCOL,
        evaluator_sha256=evaluator,
        verifier_config_sha256=CONFIG,
    )


def _read(reader):
    return reader.ready_experiences(
        current_policy_version=6, current_verifier_version=5,
    )


def _manifest(root, experiences):
    return RewardEvidenceManifest(
        root=root, receipts={root.record_key(e): root.attest(e) for e in experiences},
    )


def _plan(reader, *, manifest=None, strict=False):
    return RVLOuterPlanner(LagConfig(max_policy_lag=2, max_verifier_lag=2)).plan(
        reader, current_policy_version=6, current_verifier_version=5,
        reward_manifest=manifest, require_authenticated_rewards=strict,
    )


class _RecordingTrainer:
    def __init__(self):
        self.weight = 0
        self.train_calls = 0
        self.config = SimpleNamespace(advantage_eps=1e-6, clip_advantage=5.0)

    def snapshot_training_state(self):
        return self.weight

    def restore_training_state(self, snapshot):
        self.weight = snapshot

    def train_step(self, _samples, *, advantages=None):
        self.train_calls += 1
        self.weight += 1


def _strict_rvl_hooks(trainer, manifest):
    return RVLGRPOHooks(
        backend=object(),
        trainer=trainer,
        eval_tasks=[],
        score_fn=lambda _task, _response: 0.0,
        generation_factory=lambda **fields: SimpleNamespace(**fields),
        verified_generation_factory=lambda **fields: SimpleNamespace(**fields),
        training_evidence_authorizer=manifest.verified,
        require_authenticated_training=True,
    )


@pytest.mark.parametrize("flag", ["false", "true", 0, 1, [], {}, None])
def test_nonboolean_trust_metadata_never_becomes_trusted(tmp_path, flag):
    reader = _fixture(tmp_path, trusted=flag)
    with pytest.raises(ValueError, match="literal boolean"):
        _read(reader)


def test_untrusted_reward_cannot_inflate_curriculum_or_be_signed(tmp_path):
    reader = _fixture(tmp_path, trusted=False)
    experiences = _read(reader)
    assert not any(e.verification.trusted for e in experiences)
    plan = _plan(reader)
    assert plan.fresh_experiences == 2
    assert plan.excluded_untrusted_experiences == 2
    assert plan.family_stats == {}
    assert plan.family_weights == {}
    assert plan.authenticated_curriculum_experiences == 0
    with pytest.raises(ValueError, match="inconsistent reward provenance"):
        _root().attest(experiences[0])


def test_signature_controls_curriculum_while_unsigned_metadata_cannot(tmp_path):
    reader = _fixture(tmp_path)
    experiences = _read(reader)
    root = _root()
    manifest = _manifest(root, experiences)
    good = _plan(reader, manifest=manifest, strict=True)
    assert good.family_stats["math"]["count"] == 2
    assert good.family_stats["math"]["failures"] == 1
    assert good.family_weights["math"] > 1.0
    assert good.authenticated_curriculum_experiences == 2
    assert good.excluded_unattested_experiences == 0
    assert not good.control.freeze_training

    missing = _plan(reader, manifest=RewardEvidenceManifest(root, {}), strict=True)
    assert missing.family_weights == {}
    assert missing.excluded_unattested_experiences == 2
    assert missing.control.freeze_training
    assert "no_authenticated_reward_evidence" in missing.control.reasons

    # An unsigned but self-declared trusted record is diagnostic only in
    # strict mode. Legacy nonstrict mode is NOT authenticated.
    legacy = _plan(reader)
    assert legacy.family_weights["math"] > 1
    assert legacy.authenticated_curriculum_experiences == 0
    with pytest.raises(ValueError, match="operator-owned reward manifest"):
        _plan(reader, strict=True)


def test_signed_receipt_binds_reward_policy_verifier_and_exact_generation(tmp_path):
    reader = _fixture(tmp_path)
    before = _read(reader)
    root = _root()
    receipt = root.attest(before[0])
    assert root.verifies(before[0], receipt)
    assert not root.verifies(before[0], "not-a-valid-MAC")
    assert not root.verifies(before[0], None)
    assert not _root(run_id="second-independent-run-01").verifies(before[0], receipt)
    assert not _root(evaluator=hashlib.sha256(b"forged-grader").hexdigest()).verifies(before[0], receipt)
    assert not _root(key=bytes(reversed(range(32)))).verifies(before[0], receipt)
    changed = [
        replace(before[0], verification=replace(before[0].verification, score=0.0)),
        replace(before[0], verification=replace(before[0].verification, verifier_version=4)),
        replace(before[0], verification=replace(before[0].verification, passed=False)),
        replace(before[0], verification=replace(before[0].verification, trusted=False)),
        replace(before[0], attempt=replace(before[0].attempt, policy_version=3)),
        replace(before[0], attempt=replace(before[0].attempt, output="different answer")),
        replace(
            before[0],
            attempt=replace(before[0].attempt, metadata={
                **before[0].attempt.metadata,
                "rvl_generation": {
                    **before[0].attempt.metadata["rvl_generation"],
                    "response": "different answer",
                },
            }),
        ),
    ]
    for tampered in changed:
        assert not root.verifies(tampered, receipt)


def test_sqlite_replay_reward_rewrite_cannot_reuse_signed_receipt(tmp_path):
    reader = _fixture(tmp_path)
    initial = _read(reader)
    root = _root()
    manifest = _manifest(root, initial)
    with sqlite3.connect(reader.path) as db:
        raw = db.execute("SELECT payload FROM groups WHERE id='g1'").fetchone()[0]
        entries = json.loads(raw)
        entries[1]["reward"] = 1.0  # forge a repaired RLVR failure
        db.execute("UPDATE groups SET payload=? WHERE id='g1'", (json.dumps(entries),))
    modified = _read(reader)
    assert modified[1].verification.score == 1.0
    assert not manifest.verified(modified[1])
    plan = _plan(reader, manifest=manifest, strict=True)
    assert plan.family_stats["math"]["count"] == 1
    assert plan.excluded_unattested_experiences == 1
    assert plan.family_weights["math"] == 1.0


def test_receipt_replay_across_group_id_is_rejected(tmp_path):
    old = _fixture(tmp_path)
    experiences = _read(old)
    root = _root()
    manifest = _manifest(root, experiences)
    with sqlite3.connect(old.path) as db:
        db.execute("UPDATE groups SET id='another-group' WHERE id='g1'")
    changed = _read(old)
    assert all(not manifest.verified(exp) for exp in changed)


def test_signed_receipts_do_not_override_verifier_lag_or_runtime_provenance(tmp_path):
    reader = _fixture(tmp_path)
    root = _root()
    manifest = _manifest(root, _read(reader))
    plan = RVLOuterPlanner(LagConfig(max_policy_lag=1, max_verifier_lag=1)).plan(
        reader, current_policy_version=6, current_verifier_version=5,
        reward_manifest=manifest, require_authenticated_rewards=True,
    )
    assert plan.family_weights == {}
    assert plan.excluded_stale_experiences == 2
    assert plan.control.freeze_training
    assert "verifier_lag_exceeded" in plan.control.reasons


def test_rvl_trainer_direct_call_checks_signed_manifest_before_trainer_step(tmp_path):
    reader = _fixture(tmp_path)
    experiences = _read(reader)
    manifest = _manifest(_root(), experiences)
    trainer = _RecordingTrainer()
    hooks = _strict_rvl_hooks(trainer, manifest)

    candidate = asyncio.run(hooks.train_candidate("policy-0", experiences))

    assert candidate == "policy-1"
    assert trainer.train_calls == 1
    # The candidate snapshot is retained by the hook, but the incumbent is
    # restored before train_candidate returns.
    assert trainer.weight == 0


@pytest.mark.parametrize(
    "mutation",
    ["reward", "verifier_config", "policy_version", "generation"],
)
def test_rvl_trainer_direct_call_rejects_edited_signed_evidence_before_step(
    tmp_path, mutation
):
    reader = _fixture(tmp_path)
    experiences = _read(reader)
    manifest = _manifest(_root(), experiences)
    target = experiences[0]
    if mutation == "reward":
        target.verification.score = 0.25
    elif mutation == "verifier_config":
        target.verification.metadata["rvl_verification_config_sha256"] = "d" * 64
    elif mutation == "policy_version":
        target.attempt.policy_version += 1
    else:
        target.attempt.metadata["rvl_generation"]["metadata"][
            "response_token_ids"
        ][0] += 1

    trainer = _RecordingTrainer()
    hooks = _strict_rvl_hooks(trainer, manifest)
    with pytest.raises(RuntimeError, match="unauthenticated_training_reward_evidence"):
        asyncio.run(hooks.train_candidate("policy-0", experiences))
    assert trainer.train_calls == 0
    assert trainer.weight == 0


def test_rvl_trainer_strict_mode_requires_authorizer():
    trainer = _RecordingTrainer()
    with pytest.raises(ValueError, match="trusted evidence authorizer"):
        RVLGRPOHooks(
            backend=object(),
            trainer=trainer,
            eval_tasks=[],
            score_fn=lambda _task, _response: 0.0,
            generation_factory=lambda **fields: SimpleNamespace(**fields),
            verified_generation_factory=lambda **fields: SimpleNamespace(**fields),
            require_authenticated_training=True,
        )


def test_rvl_trainer_rejects_empty_batch_before_authorization_or_step(tmp_path):
    trainer = _RecordingTrainer()
    hooks = _strict_rvl_hooks(trainer, RewardEvidenceManifest(_root(), {}))

    with pytest.raises(ValueError, match="training batch cannot be empty"):
        asyncio.run(hooks.train_candidate("policy-0", []))
    assert trainer.train_calls == 0
    assert trainer.weight == 0


@pytest.mark.parametrize("batch_kind", ["same_object_twice", "copied_member_twice", "partial_group"])
def test_rvl_trainer_rejects_incomplete_or_duplicated_signed_group_before_step(
    tmp_path, batch_kind
):
    reader = _fixture(tmp_path)
    experiences = _read(reader)
    manifest = _manifest(_root(), experiences)
    if batch_kind == "same_object_twice":
        batch = [experiences[0], experiences[0]]
    elif batch_kind == "copied_member_twice":
        batch = [experiences[0], copy.deepcopy(experiences[0])]
    else:
        batch = experiences[:1]

    # Individual receipts remain valid for a repeated member; the batch-level
    # identity check must reject the duplicate or missing group member.
    assert all(manifest.verified(experience) for experience in batch)
    trainer = _RecordingTrainer()
    hooks = _strict_rvl_hooks(trainer, manifest)
    with pytest.raises(RuntimeError, match="training_reward_evidence_unverifiable"):
        asyncio.run(hooks.train_candidate("policy-0", batch))
    assert trainer.train_calls == 0
    assert trainer.weight == 0


def test_replay_group_size_declaration_cannot_trigger_proportional_allocation(tmp_path):
    experiences = _read(_fixture(tmp_path))
    experience = experiences[0]
    huge_group_size = 10**12
    experience.attempt.metadata["vare_rollout_group_size"] = huge_group_size
    experience.attempt.task.metadata["vare_rollout_group_size"] = huge_group_size

    with pytest.raises(ValueError, match="authenticated RVL group .* is incomplete"):
        validate_complete_rvl_reward_batch([experience])
