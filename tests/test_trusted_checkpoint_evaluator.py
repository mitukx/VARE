"""End-to-end CPU protocol: persisted bytes -> separate evaluator -> signed promotion."""
from __future__ import annotations

import asyncio
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from vare.attestation import EvidenceTrustRoot
from vare.checkpoint_artifacts import CheckpointIntegrityError, TrustedCheckpointStore
from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.promotion_journal import PromotionJournal
from vare.types import Attempt, EvaluationReport, Task, Verification
import vare.trusted_evaluation as trusted_worker


PROTOCOL = hashlib.sha256(b"frozen-integration-protocol").hexdigest()
RUN = "trusted-eval-test-run-2026"
NONCE = "trusted-eval-promotion-01"
KEY = bytes(range(32))


def _setup(tmp_path):
    store = TrustedCheckpointStore(tmp_path / "operator-objects", max_bytes=1024 * 1024)
    snapshot_paths = {}
    digests = {}
    for policy, model in {
        "p0": {"schema_version": 1, "weight": 0, "bias": -1},
        "p1": {"schema_version": 1, "weight": 1, "bias": -0.5},
    }.items():
        path = tmp_path / (policy + ".json")
        path.write_text(json.dumps(model, sort_keys=True), encoding="utf-8")
        snapshot_paths[policy] = path
        digests[policy] = store.register_file(policy, path)
    heldout = tmp_path / "operator-heldout.json"
    heldout.write_text(json.dumps({
        "schema_version": 1,
        "tasks": [
            {"id": "a", "x": 0, "label": 0, "family": "simple"},
            {"id": "b", "x": 1, "label": 1, "family": "simple"},
            {"id": "c", "x": 2, "label": 1, "family": "hard"},
            {"id": "d", "x": -1, "label": 0, "family": "hard"},
        ],
    }, sort_keys=True), encoding="utf-8")
    eval_digest = hashlib.sha256(heldout.read_bytes()).hexdigest()
    key_file = tmp_path / "operator-key"
    key_file.write_bytes(KEY)
    trust = EvidenceTrustRoot(
        key=KEY,
        protocol_sha256=PROTOCOL,
        evaluator_sha256=hashlib.sha256(Path(trusted_worker.__file__).read_bytes()).hexdigest(),
        run_id=RUN,
        evaluation_set_sha256=eval_digest,
        promotion_nonce=NONCE,
        strict_checkpoint_binding=True,
    )
    return store, digests, snapshot_paths, heldout, key_file, trust


def _signed_worker_report(store, policy_id, heldout, key_file, trust, *, expected_set=None):
    argv = [
        sys.executable, "-m", "vare.trusted_evaluation",
        "--checkpoint-store", str(store.root),
        "--policy-id", policy_id,
        "--heldout-set", str(heldout),
        "--heldout-sha256", expected_set or trust.evaluation_set_sha256,
        "--key-file", str(key_file),
        "--protocol-sha256", PROTOCOL,
        "--run-id", RUN,
        "--promotion-nonce", NONCE,
    ]
    return subprocess.run(argv, capture_output=True, text=True, timeout=25, check=False)


def _decode_report(output):
    assert output.returncode == 0, output.stderr
    raw = json.loads(output.stdout)
    assert set(raw) == {entry.name for entry in fields(EvaluationReport)}
    return EvaluationReport(**raw)


def test_store_copies_real_bytes_and_prevents_policy_rebinding(tmp_path):
    store, digests, snapshots, _, _, _ = _setup(tmp_path)
    for policy in ("p0", "p1"):
        assert store.digest(policy) == digests[policy]
        assert store.read_verified(policy)[0] == snapshots[policy].read_bytes()
    # The original candidate file can change; the trusted copy is unaffected.
    snapshots["p1"].write_bytes(b"malicious replacement")
    assert store.digest("p1") == digests["p1"]
    with pytest.raises(CheckpointIntegrityError, match="different bytes"):
        store.register_file("p1", snapshots["p1"])
    with pytest.raises(CheckpointIntegrityError, match="no registered"):
        store.digest("unknown")


def test_registered_blob_corruption_symlink_and_too_large_read_fail(tmp_path):
    store, digests, _, _, _, _ = _setup(tmp_path)
    with pytest.raises(CheckpointIntegrityError, match="in-memory"):
        store.read_verified("p0", max_read_bytes=1)
    object_path = store.objects / digests["p0"]
    object_path.write_bytes(b"modified stored bytes")
    with pytest.raises(CheckpointIntegrityError, match="hash or size mismatch"):
        store.digest("p0")
    object_path.unlink()
    another = tmp_path / "other"
    another.write_bytes(b"faked")
    object_path.symlink_to(another)
    with pytest.raises(CheckpointIntegrityError, match="symlinked"):
        store.digest("p0")


def test_external_worker_loads_model_scores_tasks_and_signs_actual_digest(tmp_path):
    store, digests, _, heldout, key_file, trust = _setup(tmp_path)
    old = _decode_report(_signed_worker_report(store, "p0", heldout, key_file, trust))
    new = _decode_report(_signed_worker_report(store, "p1", heldout, key_file, trust))
    assert old.primary == 0.5
    assert new.primary == 1.0
    assert old.n == new.n == 4
    assert trust.verifies(old) and trust.verifies(new)
    assert trust.checkpoint_digest(old) == digests["p0"]
    assert trust.checkpoint_digest(new) == digests["p1"]
    # A candidate or controller can't reassign a signed report to another model.
    new.policy_id = "p0"
    assert not trust.verifies(new)


def test_worker_rejects_drifted_heldout_or_mutated_checkpoint(tmp_path):
    store, digests, _, heldout, key_file, trust = _setup(tmp_path)
    heldout.write_text(heldout.read_text() + " ", encoding="utf-8")
    response = _signed_worker_report(store, "p1", heldout, key_file, trust)
    assert response.returncode != 0
    assert "operator-pinned content" in response.stderr
    store.objects.joinpath(digests["p1"]).write_bytes(b"tampered")
    response = _signed_worker_report(store, "p1", heldout, key_file, trust)
    assert response.returncode != 0


class AcceptVerifier:
    name = "cpu-fixture"
    version = 0
    trusted = True
    active_version = 0

    async def verify(self, attempt):
        return Verification(1.0, True, 1.0, 0, self.name, trusted=True)


class FileBackedHooks:
    def __init__(self, reports):
        self.reports = reports
        self.active = "p0"
        self.promotions = []

    def active_policy(self):
        return self.active, int(self.active == "p1")

    async def rollout(self, task, policy_id, policy_version, step):
        return Attempt(task, "fixture", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id, experiences):
        return "p1"

    async def evaluate(self, policy_id):
        return self.reports[policy_id]

    async def promote(self, candidate_id):
        self.promotions.append(candidate_id)
        self.active = candidate_id

    async def discard(self, candidate_id):
        raise AssertionError("accepted verified candidate must not be discarded")


def test_engine_uses_worker_signed_real_bytes_for_durable_promotion(tmp_path):
    store, digests, _, heldout, key_file, trust = _setup(tmp_path)
    reports = {
        policy: _decode_report(_signed_worker_report(store, policy, heldout, key_file, trust))
        for policy in ("p0", "p1")
    }
    hooks = FileBackedHooks(reports)
    journal = PromotionJournal(
        tmp_path / "operator-journal.sqlite",
        initial_policy_id="p0",
        initial_checkpoint_sha256=digests["p0"],
    )
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=AcceptVerifier(),
        promotion_journal=journal,
        checkpoint_hasher=store.digest,
        evidence_trust_root=trust,
        config=EngineConfig(
            replay_batch_size=1,
            promotion=PromotionConfig(min_primary_gain=0.01, min_eval_examples=4),
        ),
    )
    result = asyncio.run(
        loop.run_round([Task("fixture", "held-out is separate")], round_index=0)
    )
    assert result.decision.accepted
    assert result.promoted_id == "p1"
    assert hooks.promotions == ["p1"]
    assert journal.status()["active_checkpoint_sha256"] == digests["p1"]
    journal.assert_clean(policy_id="p1", checkpoint_sha256=store.digest("p1"))
