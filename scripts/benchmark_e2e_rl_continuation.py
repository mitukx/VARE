"""V2 CPU continuation/learner-SIGKILL experiment, reusing the frozen v1 backend."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_e2e_rl_real_model as base
from vare.attestation import EvidenceTrustRoot
from vare.checkpoint_artifacts import TrustedCheckpointStore
from vare.config import EngineConfig, LagConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.evidence import sha256_json
from vare.integrations.rvl import RVLTokenReplayReader
from vare.promotion_journal import PromotionJournal
from vare.reward_evidence import RewardEvidenceManifest, RewardEvidenceRoot
from vare.telemetry import EventLog
from vare.types import Attempt, Experience, Task, Verification

ROOT = base.ROOT
PROTOCOL = ROOT / "benchmarks/e2e_rl_continuation_v2/protocol.json"
WORKER_STATE = {}


def atomic_json(path, value):
    """Operator-owned progress/pointer, fsynced before a fault is exposed."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def decode_experience(value):
    attempt = dict(value["attempt"])
    attempt["task"] = Task(**attempt["task"])
    return Experience(Attempt(**attempt), Verification(**value["verification"]),
                      value["policy_lag"], value["verifier_lag"], value["shift_score"])


def fresh_tasks(seed, round_index, count, evaluation=False):
    rng = random.Random(seed * 100000 + round_index * 100 + int(evaluation))
    kind = "audit" if evaluation else "train"
    return [Task(f"v2-{kind}-s{seed}-r{round_index}-i{i}",
                 f"v2 {kind} {rng.getrandbits(64):016x}: return A. Answer:", "transport")
            for i in range(count)]


def generate_synced(prompt, seed, path, checkpoint_sha, model_sha, step, skip_sync=False):
    """A fixture worker acknowledges the tensors it actually loaded, not a label."""
    import torch
    started = time.perf_counter()
    if WORKER_STATE.get("checkpoint_sha256") != checkpoint_sha and not skip_sync:
        path = Path(path)
        base.require(path.stat().st_size < 16 * 1024**2, "oversized worker checkpoint")
        raw = path.read_bytes()
        base.require(hashlib.sha256(raw).hexdigest() == checkpoint_sha, "worker checkpoint bytes differ")
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        base.WM.load_state_dict(state["model"])
        observed = base.tensor_digest(base.WM.state_dict())
        base.require(observed == model_sha and state["step"] == step, "worker model/step differs")
        WORKER_STATE.update(checkpoint_sha256=checkpoint_sha, model_sha256=observed, step=state["step"])
    result = base.generate(prompt, seed)
    return {**result, "loaded_checkpoint_sha256": WORKER_STATE.get("checkpoint_sha256"),
            "loaded_model_sha256": WORKER_STATE.get("model_sha256"),
            "loaded_step": WORKER_STATE.get("step"), "synchronization_s": time.perf_counter()-started-result["service_s"]}


def check_worker_identity(data, checkpoint_sha, model_sha, step):
    base.require(data["loaded_checkpoint_sha256"] == checkpoint_sha, "worker served wrong checkpoint")
    base.require(data["loaded_model_sha256"] == model_sha and data["loaded_step"] == step,
                 "worker served wrong model/version")


class SyncedPool:
    """Only adapts v1's fixture generate call; owns no scheduler or trainer."""
    def __init__(self, pool, hooks):
        self.pool, self.hooks = pool, hooks

    def submit(self, fn, prompt, seed, crash):
        base.require(fn is base.generate and not crash, "unexpected fixture dispatch")
        h = self.hooks
        return self.pool.submit(generate_synced, prompt, h.seed * 1000000 + h.round_index * 10000 + h.dispatch_step,
                                str(h.checkpoint_path), h.expected_checkpoint_sha,
                                h.expected_model_sha, h.active_step)


class RoundTrace(base.Trace):
    def __init__(self, path, attempt):
        super().__init__(path)
        self.round_index, self.attempt = -1, attempt

    def record(self, stage, start, **extra):
        super().record(stage, start, round_index=self.round_index, learner_attempt=self.attempt, **extra)


class ContinuationHooks(base.ModelHooks):
    def __init__(self, model_dir, directory, pool, eval_pool, config, attempt, trace, cutpoint):
        import torch
        # A restart must not invoke v1's fresh-start constructor: it publishes p0.
        self.model_dir, self.directory = model_dir, directory
        self.eval_pool, self.guarded = eval_pool, config["mode"] == "guarded"
        self.seed, self.trace, self.fault = config["seed"], trace, "none"
        self.round_index, self.attempt, self.cutpoint = -1, attempt, cutpoint
        self.kill_round = config["kill_round"]
        self.signer = None
        self.model, self.tokenizer, self.choices = base.load_model(model_dir)
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=0.001, weight_decay=0)
        self.step, self.active, self.active_step = 0, "policy-0", 0
        self.train_calls, self.optimizer_steps, self.evaluations = 0, 0, []
        self.evaluation_started = asyncio.Event()
        self.store = TrustedCheckpointStore(directory / "checkpoints") if self.guarded else None
        self.pool = SyncedPool(pool, self)
        self.rollouts, self.journal = [], None
        if not (directory / "active.json").exists():
            self.save("policy-0")
            atomic_json(directory / "active.json", {"policy_id": "policy-0"})
        self.active = json.loads((directory / "active.json").read_text())["policy_id"]
        self.restore(self.active)
        self.active_step = self.step
        self.initial_state_digest = base.tensor_digest(self.read("policy-0"))
        self.optimizer.register_step_post_hook(self._after_optimizer)

    def active_policy(self):
        return self.active, self.active_step

    def state_digest(self):
        return base.tensor_digest(base.snapshot(self.model, self.optimizer, self.step))

    def pause(self, point):
        if self.cutpoint == point and self.round_index == self.kill_round:
            atomic_json(self.directory / "cutpoint.json", {
                "cutpoint": point, "round_index": self.round_index, "pid": os.getpid(),
                "pgid": os.getpgrp(), "time": time.perf_counter(), "active_policy": self.active,
                "volatile_model_sha256": base.tensor_digest(self.model.state_dict())})
            os.kill(os.getpid(), signal.SIGSTOP)
            raise RuntimeError("stopped learner unexpectedly resumed instead of being SIGKILLed")

    def _after_optimizer(self, *_):
        append_json(self.directory / "optimizer_attempts.jsonl", {
            "round_index": self.round_index, "learner_attempt": self.attempt,
            "candidate_state_sha256": base.tensor_digest(base.snapshot(self.model, self.optimizer, self.step+1)),
            "candidate_model_sha256": base.tensor_digest(self.model.state_dict()), "time": time.perf_counter()})
        self.pause("post_optimizer")

    def save(self, policy_id):
        actual = self.candidate_id if policy_id == "policy-1" and self.round_index >= 0 else policy_id
        super().save(actual)
        if actual != "policy-0":
            self.pause("candidate_saved")

    def begin_round(self, round_index):
        self.round_index, self.trace.round_index = round_index, round_index
        self.candidate_id = f"policy-{round_index+1:04d}"
        self.round_dir = self.directory / "rounds" / f"r{round_index:04d}"
        self.round_dir.mkdir(parents=True, exist_ok=True)
        self.incumbent_id = self.active
        self.incumbent_state_sha = self.state_digest()
        self.expected_checkpoint_sha = self.digest(self.active)
        self.expected_model_sha = base.tensor_digest(self.model.state_dict())
        self.checkpoint_path = self.store.objects / self.expected_checkpoint_sha if self.store else self.directory / f"{self.active}.pt"
        self.rollouts, self.evaluations = [], []

    async def rollout(self, task, policy_id, policy_version, step):
        started = time.perf_counter()
        self.dispatch_step = step  # submit happens synchronously before yielding
        attempt = await super().rollout(task, policy_id, policy_version, step)
        data = attempt.metadata["worker"]
        check_worker_identity(data, self.expected_checkpoint_sha, self.expected_model_sha, policy_version)
        generation = attempt.metadata["rvl_generation"]
        generation["metadata"].update(loaded_checkpoint_sha256=data["loaded_checkpoint_sha256"],
                                       loaded_model_sha256=data["loaded_model_sha256"], loaded_step=data["loaded_step"])
        attempt.metadata["rvl_generation_behavior_sha256"] = RVLTokenReplayReader._generation_behavior_digest(generation)
        attempt.metadata["rvl_group_behavior_sha256"] = RVLTokenReplayReader._canonical_digest([generation])
        self.rollouts.append(asdict(attempt))
        append_json(self.round_dir / f"rollouts-attempt{self.attempt}.jsonl", asdict(attempt))
        self.trace.record("rollout_verified", started, loaded_checkpoint_sha256=data["loaded_checkpoint_sha256"])
        return attempt

    async def train_candidate(self, incumbent_id, experiences):
        atomic_json(self.round_dir / f"batch-attempt{self.attempt}.json", {
            "incumbent_id": incumbent_id, "checkpoint_sha256": self.expected_checkpoint_sha,
            "model_sha256": self.expected_model_sha, "step": self.active_step,
            "members": [{"created_step":e.attempt.created_step,"policy_id":e.attempt.policy_id,
                         "policy_version":e.attempt.policy_version,
                         "loaded_checkpoint_sha256":e.attempt.metadata["worker"]["loaded_checkpoint_sha256"],
                         "reward":e.verification.score,"logprob":e.attempt.logprob} for e in experiences]})
        result = await super().train_candidate(incumbent_id, experiences)
        base.require(result == "policy-1", "v1 fixture contract changed")
        self.update = json.loads((self.directory / "optimizer_update.json").read_text())
        atomic_json(self.round_dir / f"update-attempt{self.attempt}.json", self.update)
        return self.candidate_id

    async def evaluate(self, policy_id):
        report = await super().evaluate(policy_id)
        atomic_json(self.round_dir / f"evaluation-{policy_id}-attempt{self.attempt}.json", self.evaluations[-1])
        return report

    async def promote(self, candidate_id):
        with self.trace.span("promotion"):
            self.restore(candidate_id)
            intent = {"round_index": self.round_index, "incumbent_id": self.incumbent_id,
                      "learner_attempt": self.attempt,
                      "candidate_id": candidate_id, "candidate_checkpoint_sha256": self.digest(candidate_id),
                      "candidate_state_sha256": self.state_digest(), "step": self.step,
                      "evaluations": self.evaluations, "update": self.update,
                      "evaluation_set_sha256": self.signer.evaluation_set_sha256,
                      "promotion_nonce": self.signer.promotion_nonce}
            atomic_json(self.directory / "publication-intent.json", intent)
            if self.guarded:
                base.require(self.journal.status()["pending_tx_id"] == self.signer.promotion_nonce, "promotion was not prepared")
                self.pause("journal_prepared")
            self.active, self.active_step = candidate_id, self.step
            atomic_json(self.directory / "active.json", {"policy_id": candidate_id})
            self.pause("pointer_published")

    async def discard(self, candidate_id):
        with self.trace.span("rollback"):
            self.restore(self.active)
            base.require(self.state_digest() == self.incumbent_state_sha, "rollback did not restore actual incumbent")


class CanaryGrader(base.Grader):
    async def verify(self, attempt):
        result = await super().verify(attempt)
        path = self.hooks.directory / "canary.json"
        if self.hooks.round_index == 0 and attempt.created_step == 0 and not path.exists():
            atomic_json(path, asdict(Experience(attempt, result, 0, 0, 0.0)))
        return result


def cursor_for(hooks, round_index):
    return {"completed_round": round_index, "active_policy": hooks.active,
            "active_checkpoint_sha256": hooks.digest(hooks.active),
            "active_state_sha256": hooks.state_digest(), "accepted_updates": hooks.active_step}


def recover_runtime(hooks, journal):
    """Called only by a fresh process after supervisor proves old owners dead."""
    directory = hooks.directory
    cursor_path = directory / "cursor.json"
    base.require(cursor_path.exists() or hooks.active == "policy-0", "missing durable cursor for advanced policy")
    cursor = json.loads(cursor_path.read_text()) if cursor_path.exists() else cursor_for(hooks, -1)
    before = journal.status() if journal else None
    observed_sha = hooks.digest(hooks.active)
    if journal and before["pending_tx_id"]:
        journal.reconcile(tx_id=before["pending_tx_id"], observed_policy_id=hooks.active,
                          observed_checkpoint_sha256=observed_sha)
    if journal:
        journal.assert_clean(policy_id=hooks.active, checkpoint_sha256=observed_sha)
    advanced = hooks.active != cursor["active_policy"]
    if advanced:
        intent = json.loads((directory / "publication-intent.json").read_text())
        base.require(intent["round_index"] == cursor["completed_round"]+1, "noncontiguous recovered publication")
        base.require(intent["candidate_id"] == hooks.active and intent["candidate_checkpoint_sha256"] == observed_sha,
                     "recovery pointer does not match prepared intent")
        base.require(intent["candidate_state_sha256"] == hooks.state_digest(), "recovered Adam/model/RNG differs")
        # Journal.prepare was reachable only after the signed gate. Verify the
        # retained actual checkpoint/score identities again before roll-forward.
        records = {r["policy_id"]: r for r in intent["evaluations"]}
        for pid in (intent["incumbent_id"], intent["candidate_id"]):
            record = records[pid]
            base.require(record["checkpoint_sha256"] == hooks.digest(pid), "prepared evaluation checkpoint changed")
            base.require(statistics.fmean(record["scores"].values()) == record["report"]["primary"], "prepared scores differ")
            replayed = hooks.eval_pool.submit(base.evaluate_checkpoint, str(hooks.model_dir), str(directory),
                                              pid, hooks.guarded, intent["evaluation_set_sha256"]).result(timeout=120)
            base.require(replayed["scores"] == record["scores"] and replayed["greedy"] == record["greedy"],
                         "prepared evaluation does not replay from actual checkpoint")
            base.require(record["report"]["metadata"]["per_task_scores"] == replayed["scores"], "prepared score map changed")
        base.require(records[hooks.active]["report"]["primary"] >= records[intent["incumbent_id"]]["report"]["primary"],
                     "recovered publication fails diagnostic gate")
        event_file = directory / "rounds" / f"r{intent['round_index']:04d}" / f"engine-attempt{intent['learner_attempt']}.jsonl"
        events = [json.loads(line) for line in event_file.read_text().splitlines()]
        screened = sum(e.get("count",0) for e in events if e["event"]=="replay_experiences_screened")
        recovered = {**cursor_for(hooks, intent["round_index"]), "accepted": True, "recovered_publication": True,
                     "stale_replay_rejected": screened,
                     "candidate_id": hooks.active, "candidate_state_sha256": hooks.state_digest(), "update": intent["update"]}
        atomic_json(directory / "rounds" / f"r{intent['round_index']:04d}" / "committed.json", recovered)
        cursor = cursor_for(hooks, intent["round_index"])
    else:
        base.require(hooks.state_digest() == cursor["active_state_sha256"], "cursor/model/Adam/RNG mismatch")
    if journal and journal.status()["round_owner_id"]:
        journal.recover_round(owner_id=journal.status()["round_owner_id"], observed_policy_id=hooks.active,
                              observed_checkpoint_sha256=observed_sha)
    atomic_json(cursor_path, cursor)
    atomic_json(directory / f"recovery-attempt{hooks.attempt}.json", {
        "journal_before": before, "journal_after": journal.status() if journal else None,
        "roll_forward": advanced, "cursor": cursor, "time": time.perf_counter()})
    return cursor


async def learner_rounds(hooks, journal, config, cursor):
    root = hooks.directory
    reward_key = os.urandom(32)
    for index in range(cursor["completed_round"]+1, config["rounds"]):
        hooks.begin_round(index)
        train = fresh_tasks(hooks.seed, index, config["tasks"])
        evaluation = fresh_tasks(hooks.seed, index, config["eval_tasks"], True)
        base.require(not ({t.prompt for t in train} & {t.prompt for t in evaluation}), "task overlap")
        atomic_json(hooks.round_dir / "training_tasks.json", [asdict(t) for t in train])
        atomic_json(hooks.round_dir / "evaluation_tasks.json", [asdict(t) for t in evaluation])
        atomic_json(root / "evaluation_tasks.json", [asdict(t) for t in evaluation])
        hooks.signer = EvidenceTrustRoot(os.urandom(32), base.file_sha(PROTOCOL), base.file_sha(__file__), uuid.uuid4().hex,
                                         base.file_sha(root / "evaluation_tasks.json"), uuid.uuid4().hex, True)
        reward = RewardEvidenceRoot(reward_key, hooks.signer.run_id, base.file_sha(PROTOCOL), base.file_sha(__file__),
                                    sha256_json({"grader": "return-A", "version": 1}))
        grader = CanaryGrader(hooks, reward)
        event_path = hooks.round_dir / f"engine-attempt{hooks.attempt}.jsonl"
        event_path.touch()
        loop = CapabilityLoop(hooks=hooks, verifier=grader,
                              config=EngineConfig(rollout_concurrency=config["workers"], replay_batch_size=config["tasks"],
                                                  lag=LagConfig(max_policy_lag=0, max_verifier_lag=0),
                                                  promotion=PromotionConfig(min_primary_gain=0, min_eval_examples=config["eval_tasks"])),
                              seed=hooks.seed+index*100, event_log=EventLog(event_path),
                              evidence_trust_root=hooks.signer if hooks.guarded else None,
                              promotion_journal=journal, checkpoint_hasher=hooks.digest if hooks.guarded else None,
                              training_evidence_authorizer=RewardEvidenceManifest(reward, grader.receipts).verified if hooks.guarded else None,
                              require_authenticated_training=hooks.guarded)
        canary_path = root / "canary.json"
        canary_inserted = False
        if canary_path.exists():
            canary = decode_experience(json.loads(canary_path.read_text()))
            if canary.attempt.policy_version < hooks.active_step:
                base.require(loop.replay.add_group([canary], [1.0]), "old canary insertion failed")
                canary_inserted = True
        assess = loop.lag.assess
        def timed_assess(**kwargs):
            with hooks.trace.span("freshness"):
                return assess(**kwargs)
        loop.lag.assess = timed_assess
        sample = loop.replay.sample_current
        def timed_sample(*args, **kwargs):
            with hooks.trace.span("replay"):
                return sample(*args, **kwargs)
        loop.replay.sample_current = timed_sample
        start, cpu, prior_train_calls = time.perf_counter(), time.process_time(), hooks.train_calls
        result = await loop.run_round(train, round_index=index, rollout_count=config["tasks"])
        base.require(hooks.train_calls == prior_train_calls+1 and hooks.candidate_state_digest, "no real candidate update")
        base.require(hooks.update["batch_size"] == config["tasks"], "training batch budget differs")
        expected = hooks.candidate_state_digest if result.decision.accepted else hooks.incumbent_state_sha
        base.require(hooks.state_digest() == expected, "post-decision runtime differs from actual checkpoint")
        events = [json.loads(line) for line in event_path.read_text().splitlines()]
        screened = sum(e.get("count", 0) for e in events if e["event"] == "replay_experiences_screened")
        base.require(screened == int(canary_inserted), "real stale canary was not screened exactly once")
        committed = {**cursor_for(hooks, index), "candidate_id": result.candidate_id,
                     "candidate_state_sha256": hooks.candidate_state_digest, "accepted": result.decision.accepted,
                     "update": hooks.update, "stale_replay_rejected": screened, "result": asdict(result)}
        atomic_json(hooks.round_dir / "committed.json", committed)
        cursor = cursor_for(hooks, index)
        atomic_json(root / "cursor.json", cursor)
        hooks.trace.record("round", start, cpu_s=time.process_time()-cpu)
        hooks.pause("round_committed")
        if hooks.attempt:
            progress = root / "post_restart_update.json"
            if not progress.exists():
                atomic_json(progress, {"completed_round": index, "state_sha256": hooks.state_digest(),
                                       "candidate_state_sha256": hooks.candidate_state_digest, "time": time.perf_counter()})
        print(json.dumps({"round": index, "attempt": hooks.attempt, "active": hooks.active, "step": hooks.active_step}), flush=True)
    base.require(hooks.active_step >= 2, "fewer than two real promotions; continuation not demonstrated")
    if journal:
        journal.assert_clean(policy_id=hooks.active, checkpoint_sha256=hooks.digest(hooks.active))
    atomic_json(root / "learner_done.json", {"cursor": cursor, "journal": journal.status() if journal else None,
                                             "final_state_sha256": hooks.state_digest(), "time": time.perf_counter()})


def learner(config_path, attempt, cutpoint):
    config = json.loads(Path(config_path).read_text())
    base.require(config.get("source_sha256"), "missing pinned learner sources")
    base.require(all(base.file_sha(ROOT/name)==digest for name,digest in config["source_sha256"].items()),
                 "learner source differs from pinned campaign")
    import torch
    directory = Path(config["directory"])
    if attempt:
        proof = json.loads((directory / "kill_record.json").read_text())
        base.require(proof["live_old_descendants"] == 0, "old learner work still executing")
    base.torch_setup()
    warm, tokenizer, choices = base.load_model(config["model_dir"])
    opt = torch.optim.AdamW([p for p in warm.parameters() if p.requires_grad], lr=0.001, weight_decay=0)
    (-warm(torch.tensor([tokenizer.encode("v2 warmup: return A. Answer:")])).logits[0,-1,choices].log_softmax(-1)[0]).backward()
    opt.step()
    del warm, tokenizer, opt
    torch.manual_seed(config["seed"])
    pool, pool_start = base.pool_start(config["model_dir"], config["workers"])
    evaluation, eval_start = base.pool_start(config["model_dir"], 2)
    try:
        trace = RoundTrace(directory / "trace.jsonl", attempt)
        hooks = ContinuationHooks(config["model_dir"], directory, pool, evaluation, config, attempt, trace, cutpoint)
        journal = PromotionJournal(directory / "promotion.sqlite", initial_policy_id="policy-0",
                                   initial_checkpoint_sha256=hooks.digest("policy-0")) if hooks.guarded else None
        hooks.journal = journal
        if journal:
            commit = journal.commit
            def committing(tx_id):
                commit(tx_id)
                hooks.pause("journal_committed")
            journal.commit = committing
        cursor = recover_runtime(hooks, journal)
        atomic_json(directory / f"setup-attempt{attempt}.json", {"rollout_pool_start_s": pool_start,
                                                                "evaluation_pool_start_s": eval_start})
        asyncio.run(learner_rounds(hooks, journal, config, cursor))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        evaluation.shutdown(wait=True, cancel_futures=True)


def wait_segment(process, root, cutpoint, deadline):
    import psutil
    while process.poll() is None:
        base.require(time.perf_counter() < deadline, "learner progress deadline exceeded")
        marker = root / "cutpoint.json"
        if cutpoint != "none" and marker.exists():
            point = json.loads(marker.read_text())
            owner = psutil.Process(process.pid)
            if owner.status() != psutil.STATUS_STOPPED:
                time.sleep(0.01)
                continue
            base.require(point["pid"] == process.pid and os.getpgid(process.pid) == process.pid, "foreign fault target")
            members = [{"pid": p.pid, "created": p.create_time()} for p in owner.children(recursive=True)]
            base.require(all(os.getpgid(m["pid"]) == process.pid for m in members), "descendant escaped owned process group")
            killed = time.perf_counter()
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            live = []
            for _ in range(500):
                live = []
                for member in members:
                    try:
                        p = psutil.Process(member["pid"])
                        if p.create_time() == member["created"] and p.status() != psutil.STATUS_ZOMBIE:
                            live.append(p.pid)
                    except psutil.NoSuchProcess:
                        pass
                if not live:
                    break
                time.sleep(0.02)
            proof = {"learner_pid": process.pid, "returncode": process.returncode, "cutpoint": cutpoint,
                     "killed_at": killed, "old_descendants": members, "live_old_descendants": len(live)}
            atomic_json(root / "kill_record.json", proof)
            base.require(process.returncode == -signal.SIGKILL and not live, "learner/descendants did not terminate")
            return True
        time.sleep(0.01)
    base.require(process.returncode == 0, f"learner exited {process.returncode}; inspect retained segment log")
    base.require(cutpoint == "none", "fault boundary was not reached")
    return False


def numeric_update(record):
    return {k:v for k,v in record.items() if k != "optimizer_steps"}


def run_case(directory, config, cutpoint="none"):
    directory.mkdir(parents=True, exist_ok=False)
    config = {**config, "directory": str(directory.resolve())}
    atomic_json(directory / "config.json", config)
    case_start = time.perf_counter()
    with base.Resources(directory / "resources.jsonl") as resources:
        for attempt in range(2 if cutpoint != "none" else 1):
            selected = cutpoint if attempt == 0 else "none"
            with (directory / f"segment-{attempt}.log").open("w") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__)), "--learner", str(directory / "config.json"),
                                            "--attempt", str(attempt), "--cutpoint", selected],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    interrupted = wait_segment(process, directory, selected, time.perf_counter()+120)
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)
                    raise
            if not interrupted:
                break
    done = json.loads((directory / "learner_done.json").read_text())
    rounds = [json.loads((directory / "rounds" / f"r{i:04d}" / "committed.json").read_text()) for i in range(config["rounds"])]
    trace = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    round_times = [r["duration_s"] for r in trace if r["stage"] == "round"]
    attempts = [json.loads(line) for line in (directory / "optimizer_attempts.jsonl").read_text().splitlines()]
    case_end = time.perf_counter()
    case_wall_s = case_end-case_start
    atomic_json(directory / "case_timing.json", {
        "start_monotonic_s": case_start,
        "end_monotonic_s": case_end,
    })
    summary = {"protocol_id": config["protocol_id"], "protocol_conformant": config["protocol_conformant"],
               "seed": config["seed"], "workers": config["workers"], "mode": config["mode"], "cutpoint": cutpoint,
               "rounds": config["rounds"], "completed_rounds": len(rounds), "physical_optimizer_attempts": len(attempts),
               "accepted_updates": done["cursor"]["accepted_updates"], "final_state_sha256": done["final_state_sha256"],
               "final_journal": done["journal"], "resources": resources.metrics(), "stages": base.stage_metrics(trace),
               "case_wall_s": case_wall_s, "round_wall_s": sum(round_times),
               "round_latency_p50_s": base.percentile(round_times, .5), "round_latency_p95_s": base.percentile(round_times,.95),
               "stale_replay_rejected": sum(r.get("stale_replay_rejected",0) for r in rounds),
               "tasks_s": config["tasks"] * len(round_times)/sum(round_times),
               "tokens_s": config["tasks"] * len(round_times)/sum(round_times)}
    if cutpoint != "none":
        proof = json.loads((directory / "kill_record.json").read_text())
        progress = json.loads((directory / "post_restart_update.json").read_text())
        summary.update(recovery_to_next_update_s=progress["time"]-proof["killed_at"],
                       resumed_update_round=progress["completed_round"], live_old_descendants=proof["live_old_descendants"])
    base.require(summary["resources"]["sampled_tree_peak_rss_bytes"] < 12*1024**3, "RSS ceiling exceeded")
    atomic_json(directory / "summary.json", summary)
    return summary, rounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--workers", nargs="+", type=int, default=[1,2,4,8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2701,2702,2703])
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--eval-tasks", type=int, default=32)
    parser.add_argument("--skip-faults", action="store_true")
    parser.add_argument("--learner", type=Path)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--cutpoint", default="none")
    args = parser.parse_args()
    if args.learner:
        learner(args.learner, args.attempt, args.cutpoint)
        return
    base.require(args.output is not None and args.rounds >= 6 and args.tasks > 0 and args.eval_tasks > 0, "invalid campaign")
    base.require(args.workers and all(w in (1,2,4,8) for w in args.workers) and len(args.workers)==len(set(args.workers)), "invalid workers")
    base.require(args.seeds and len(args.seeds)==len(set(args.seeds)), "invalid seeds")
    base.require(args.skip_faults or 2 in args.workers, "recovery cases require a two-worker reference")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    conformant = (args.workers == protocol["workers"] and args.seeds == protocol["seeds"] and args.rounds == protocol["rounds"]
                  and args.tasks == protocol["rollouts"] and args.eval_tasks == protocol["evaluation_tasks"] and not args.skip_faults)
    if args.model_dir is None:
        from huggingface_hub import snapshot_download
        args.model_dir = Path(snapshot_download(base.MODEL, revision=base.REVISION, allow_patterns=list(protocol["model_files"])))
    args.model_dir = args.model_dir.resolve()
    base.require(all(base.file_sha(args.model_dir/name)==sha for name,sha in protocol["model_files"].items()), "model revision bytes differ")
    template = {"model_dir": str(args.model_dir), "rounds": args.rounds, "tasks": args.tasks,
                "eval_tasks": args.eval_tasks, "kill_round": protocol["kill_round"],
                "protocol_id": protocol["schema"] if conformant else protocol["schema"]+".diagnostic",
                "protocol_conformant": conformant}
    sources = [Path(__file__), Path(base.__file__), PROTOCOL, ROOT/"requirements/e2e-rl-real-model-v1.txt",
               ROOT/"scripts/audit_e2e_rl_continuation.py", ROOT/"tests/test_e2e_rl_continuation.py",
               *[ROOT/"src/vare"/name for name in ("engine.py","lag.py","replay.py","promotion.py","attestation.py",
                                                "reward_evidence.py","runtime.py","checkpoint_artifacts.py","promotion_journal.py")]]
    template["source_sha256"] = {str(p.relative_to(ROOT)):base.file_sha(p) for p in sources}
    for source in sources:
        target = args.output/"sources"/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
    atomic_json(args.output/"manifest.json", {**template, "arguments": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                                             "protocol_sha256":base.file_sha(PROTOCOL), "sources":{str(p.relative_to(ROOT)):base.file_sha(p) for p in sources},
                                             "base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                                             "environment": {k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","TOKENIZERS_PARALLELISM","PYTHONHASHSEED")},
                                             "python":sys.version,"model":base.MODEL,"revision":base.REVISION,
                                             "started_unix":time.time(),"platform":platform.platform(),
                                             "hardware":subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string","hw.memsize","hw.ncpu"],text=True).splitlines() if sys.platform=="darwin" else platform.machine(),
                                             "dependencies":sorted(f"{d.metadata['Name']}=={d.version}" for d in base.import_metadata.distributions())})
    start, rows, references = time.perf_counter(), [], {}
    for workers in args.workers:
        for i, seed in enumerate(args.seeds):
            for mode in (["minimal","guarded"] if i%2==0 else ["guarded","minimal"]):
                name=f"normal-w{workers}-s{seed}-{mode}"
                row, records=run_case(args.output/name,{**template,"workers":workers,"seed":seed,"mode":mode})
                row["directory"]=name
                rows.append(row)
                references[(workers,seed,mode)]=records
                if mode == "guarded":
                    other=references.get((workers,seed,"minimal"))
                else:
                    other=references.get((workers,seed,"guarded"))
                if other:
                    base.require(all(a["active_state_sha256"]==b["active_state_sha256"] and
                                     numeric_update(a["update"])==numeric_update(b["update"])
                                     for a,b in zip(records,other,strict=True)), "normal numerical workload differs")
                atomic_json(args.output/"progress.json",rows)
                print(json.dumps({k:row[k] for k in ("directory","tasks_s","accepted_updates","physical_optimizer_attempts")}),flush=True)
                base.require(time.perf_counter()-start<3600,"campaign deadline exceeded")
    if not args.skip_faults:
        base.require(2 in args.workers, "recovery cases require the matched two-worker reference")
        for seed in args.seeds:
            reference=references[(2,seed,"guarded")]
            for cutpoint in protocol["cutpoints"]:
                name=f"kill-{cutpoint}-s{seed}-guarded"
                row, records=run_case(args.output/name,{**template,"workers":2,"seed":seed,"mode":"guarded"},cutpoint)
                for actual, expected in zip(records,reference,strict=True):
                    base.require(actual["active_state_sha256"]==expected["active_state_sha256"] and
                                 actual["candidate_state_sha256"]==expected["candidate_state_sha256"] and
                                 actual["accepted"]==expected["accepted"] and
                                 numeric_update(actual["update"])==numeric_update(expected["update"]), "resumed update differs from uninterrupted reference")
                row["directory"],row["exact_reference_match"]=name,True
                rows.append(row)
                atomic_json(args.output/"progress.json",rows)
                print(json.dumps({k:row[k] for k in ("directory","recovery_to_next_update_s","physical_optimizer_attempts","exact_reference_match")}),flush=True)
                base.require(time.perf_counter()-start<3600,"campaign deadline exceeded")
    groups=[]
    for workers in args.workers:
        for mode in ("minimal","guarded"):
            group=[r for r in rows if r["cutpoint"]=="none" and r["workers"]==workers and r["mode"]==mode]
            values=[r["tasks_s"] for r in group]
            groups.append({"workers":workers,"mode":mode,"n":len(group),"tasks_s_mean":statistics.fmean(values),
                           "tasks_s_sd":statistics.stdev(values) if len(values)>1 else None})
    atomic_json(args.output/"summary.json", {**template,"cases":len(rows),"completed_rounds":sum(r["completed_rounds"] for r in rows),
                                           "physical_optimizer_attempts":sum(r["physical_optimizer_attempts"] for r in rows),
                                           "exact_recovery_cases":sum(r.get("exact_reference_match",False) for r in rows),
                                           "normal":groups,"campaign_wall_s":time.perf_counter()-start,"pass":True})
    atomic_json(args.output/"SHA256SUMS.json",{str(p.relative_to(args.output)):base.file_sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()})
    print(json.dumps({"pass":True,"protocol_conformant":conformant,"cases":len(rows),"campaign_s":time.perf_counter()-start}),flush=True)


if __name__ == "__main__":
    main()
