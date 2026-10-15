"""Actual evaluator process, rollback, object restoration and durable promotion."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.promotion_journal import PromotionJournal
from vare.types import EvaluationReport, Task
from test_trusted_checkpoint_evaluator import AcceptVerifier, FileBackedHooks, _setup


@pytest.mark.skipif(os.name != "posix", reason="POSIX IO recovery experiment")
@pytest.mark.parametrize("fault", ["fifo", "oversized"])
def test_worker_fault_rolls_back_then_verified_restore_promotes(tmp_path, fault):
    store, digests, snapshots, heldout, key_file, initial_trust = _setup(tmp_path)
    # Fresh integration-only inputs, disjoint from optimization. This frozen
    # linear classifier is a protocol control, never model capability evidence.
    heldout.write_text(json.dumps({"schema_version": 1, "tasks": [
        {"id": f"checkpoint-io-{i}", "x": x, "label": int(x >= 0.5), "family": "io-control"}
        for i, x in enumerate([-7, -0.25, 0.1, 0.75, 4, 11])
    ]}, sort_keys=True))
    trust = replace(initial_trust, evaluation_set_sha256=hashlib.sha256(heldout.read_bytes()).hexdigest())
    worker_outcomes = []

    class Hooks(FileBackedHooks):
        async def evaluate(self, policy_id):
            argv = [
                sys.executable, "-m", "vare.trusted_evaluation",
                "--checkpoint-store", str(store.root), "--policy-id", policy_id,
                "--heldout-set", str(heldout), "--heldout-sha256", trust.evaluation_set_sha256,
                "--key-file", str(key_file), "--protocol-sha256", trust.protocol_sha256,
                "--run-id", trust.run_id, "--promotion-nonce", trust.promotion_nonce,
            ]
            started = time.perf_counter()
            worker = asyncio.create_task(asyncio.to_thread(
                subprocess.run, argv, capture_output=True, text=True, timeout=2, cwd=tmp_path,
            ))
            cancellation = None
            # Cancelling to_thread alone abandons a live subprocess. Own its
            # bounded completion even if a sibling evaluator fails first.
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError as exc:
                    cancellation = exc
                except Exception:
                    break
            try:
                result = worker.result()
            except subprocess.TimeoutExpired:
                worker_outcomes.append({"policy_id": policy_id, "timeout": True})
                raise RuntimeError("checkpoint evaluator exceeded external watchdog") from None
            worker_outcomes.append({"policy_id": policy_id, "timeout": False,
                                    "wall_s": time.perf_counter() - started,
                                    "returncode": result.returncode,
                                    "stdout": result.stdout, "stderr": result.stderr})
            if cancellation is not None:
                raise cancellation
            if result.returncode:
                raise RuntimeError("checkpoint evaluator rejected input")
            report = EvaluationReport(**json.loads(result.stdout))
            assert trust.verifies(report)
            assert trust.checkpoint_digest(report) == digests[policy_id]
            return report

        async def discard(self, candidate_id):
            assert candidate_id == "p1"
            self.discarded.append(candidate_id)
            self.active = "p0"

    hooks = Hooks({})
    hooks.discarded = []
    journal = PromotionJournal(tmp_path / "journal.sqlite", initial_policy_id="p0",
                               initial_checkpoint_sha256=digests["p0"])
    loop = CapabilityLoop(
        hooks=hooks, verifier=AcceptVerifier(), evidence_trust_root=trust,
        promotion_journal=journal, checkpoint_hasher=store.digest,
        config=EngineConfig(replay_batch_size=1,
                            promotion=PromotionConfig(min_eval_examples=6, min_primary_gain=0.01)),
    )
    obj = store.objects / digests["p1"]
    obj.unlink()
    if fault == "fifo":
        os.mkfifo(obj)
    else:
        with obj.open("wb") as out:
            out.truncate(512 * 1024)
    task = Task("checkpoint-io-development", "exercise file-backed candidate recovery")

    async def scenario():
        with pytest.raises(RuntimeError, match="checkpoint evaluator"):
            await loop.run_round([task], round_index=0)
        failed_status = journal.status()
        assert hooks.active == "p0" and hooks.discarded == ["p1"]
        assert hooks.promotions == [] and failed_status["round_owner_id"] is None
        assert failed_status["pending_tx_id"] is None
        journal.assert_clean(policy_id="p0", checkpoint_sha256=store.digest("p0"))
        # Trusted operator restores the exact original bytes after the failed
        # workers have exited. Registration re-verifies the immutable binding.
        obj.unlink()
        assert store.register_file("p1", snapshots["p1"]) == digests["p1"]
        result = await loop.run_round([task], round_index=1)
        assert result.decision.accepted and result.promoted_id == "p1"
        assert hooks.promotions == ["p1"]
        reopened = PromotionJournal(journal.path, initial_policy_id="p0",
                                    initial_checkpoint_sha256=digests["p0"])
        reopened.assert_clean(policy_id="p1", checkpoint_sha256=store.digest("p1"))
        assert reopened.status()["round_owner_id"] is None
        evidence_dir = os.environ.get("VARE_CHECKPOINT_IO_EVIDENCE_DIR")
        if evidence_dir:
            output = Path(evidence_dir)
            output.mkdir(parents=True, exist_ok=True)
            (output / f"recovery-{fault}.json").write_text(json.dumps({
                "fault": fault, "failed_round_status": failed_status,
                "reopened_status": reopened.status(), "round_result": asdict(result),
                "worker_outcomes": worker_outcomes,
                "heldout_sha256": trust.evaluation_set_sha256, "checkpoint_sha256": digests,
                "scope": "linear-classifier protocol control; no training or model-capability result",
            }, indent=2) + "\n")

    asyncio.run(scenario())
    # The watchdog makes the baseline recoverable too, but masks an IO hang.
    # Require the store itself to reject the FIFO, independently of that guard.
    assert not any(row["timeout"] for row in worker_outcomes), "checkpoint IO relied on evaluator watchdog"
    assert any("CheckpointIntegrityError" in row.get("stderr", "") for row in worker_outcomes)
