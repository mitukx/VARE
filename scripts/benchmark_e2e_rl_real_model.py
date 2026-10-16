"""Measured single-host model/control-plane composition; see the frozen protocol.

No production trainer, scheduler, replay or promotion subsystem is introduced.
The small REINFORCE hook is a deliberately bounded external-backend fixture.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import io
from importlib import metadata as import_metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vare.attestation import EvidenceTrustRoot
from vare.checkpoint_artifacts import TrustedCheckpointStore
from vare.config import EngineConfig, LagConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.evidence import sha256_json
from vare.integrations.rvl import RVLTokenReplayReader
from vare.promotion_journal import PromotionJournal
from vare.reward_evidence import RewardEvidenceManifest, RewardEvidenceRoot
from vare.telemetry import EventLog
from vare.types import Attempt, EvaluationReport, Experience, Task, Verification

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/e2e_rl_real_model_v1/protocol.json"
MODEL = "sshleifer/tiny-gpt2"
REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
CTX = mp.get_context("spawn")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def torch_setup():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def load_model(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True).float().eval()
    # GPT-2's tied token embedding/output matrix remains frozen; autograd and
    # AdamW really update the remaining transformer/position parameters.
    model.get_input_embeddings().weight.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    choices = [tokenizer.encode(s, add_special_tokens=False) for s in (" A", " B")]
    require(all(len(ids) == 1 for ids in choices), "choices must be one token")
    return model, tokenizer, [ids[0] for ids in choices]


def worker_init(model_dir, barrier):
    global WM, WT, WC
    torch_setup()
    WM, WT, WC = load_model(model_dir)
    barrier.wait(timeout=120)


def generate(prompt, seed, crash=False):
    import torch
    if crash:
        os._exit(73)
    started, cpu = time.perf_counter(), time.process_time()
    ids = WT.encode(prompt, add_special_tokens=False)
    with torch.inference_mode():
        probs = WM(torch.tensor([ids])).logits[0, -1, WC].softmax(-1)
        action = int(torch.multinomial(probs, 1, generator=torch.Generator().manual_seed(seed)))
    return {"action": action, "token": WC[action], "prompt_ids": ids,
            "logprob": float(probs[action].log()), "worker_pid": os.getpid(),
            "service_s": time.perf_counter() - started, "cpu_s": time.process_time() - cpu,
            "service_start": started, "service_end": time.perf_counter()}


def ready():
    return generate("Warmup: return A. Answer:", 0)


def pool_start(model_dir, workers):
    start = time.perf_counter()
    pool = ProcessPoolExecutor(workers, mp_context=CTX, initializer=worker_init,
                               initargs=(str(model_dir), CTX.Barrier(workers)))
    # Submission starts all workers. Warmup is outside steady round latency.
    jobs = [pool.submit(ready) for _ in range(workers * 2)]
    for job in jobs:
        job.result(timeout=120)
    return pool, time.perf_counter() - start


async def owned_job(pool, fn, *args):
    """Thread/process work cannot outlive its hook after caller cancellation."""
    future = asyncio.wrap_future(pool.submit(fn, *args))
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout=120)
    except BaseException:
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not future.cancelled():
            future.exception()
        raise


def evaluate_checkpoint(model_dir, run_dir, policy_id, guarded, task_sha):
    """Separate process loads actual bytes and independent untouched task file."""
    import torch
    start, cpu = time.perf_counter(), time.process_time()
    directory = Path(run_dir)
    raw_tasks = (directory / "evaluation_tasks.json").read_bytes()
    require(hashlib.sha256(raw_tasks).hexdigest() == task_sha, "evaluation set changed")
    if guarded:
        store = TrustedCheckpointStore(directory / "checkpoints", read_only=True)
        raw, digest = store.read_verified(policy_id)
    else:
        raw = (directory / f"{policy_id}.pt").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    # Evaluation owns this process-local model. It never calls the learner's
    # verifier or reads its rewards, labels, optimizer or in-memory candidate.
    WM.load_state_dict(state["model"])
    scores, greedy, latencies, prompt_tokens = {}, {}, [], 0
    with torch.inference_mode():
        for row in json.loads(raw_tasks):
            item_start = time.perf_counter()
            require(row["prompt"].endswith("return A. Answer:"), "invalid independent task")
            ids = WT.encode(row["prompt"], add_special_tokens=False)
            probs = WM(torch.tensor([ids])).logits[0, -1, WC].softmax(-1)
            scores[row["id"]] = float(probs[0])
            greedy[row["id"]] = float(int(probs.argmax()) == 0)
            prompt_tokens += len(ids)
            latencies.append(time.perf_counter() - item_start)
    if guarded:
        require(store.digest(policy_id) == digest, "checkpoint changed during evaluation")
    return {"policy_id": policy_id, "scores": scores, "greedy": greedy,
            "checkpoint_sha256": digest, "item_latency_s": latencies,
            "prompt_tokens": prompt_tokens, "worker_pid": os.getpid(),
            "service_s": time.perf_counter() - start, "cpu_s": time.process_time() - cpu}


class Trace:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def record(self, stage, start, **extra):
        row = {"stage": stage, "start": start, "end": time.perf_counter(), **extra}
        row["duration_s"] = row["end"] - start
        self.rows.append(row)
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    @contextmanager
    def span(self, stage):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, start)


class Resources:
    """20ms process-tree RSS/CPU samples (RSS sums shared pages per process)."""
    def __init__(self, path):
        self.path, self.rows, self.seen = path, [], {}
        self.stop = threading.Event()

    def sample(self):
        import psutil
        root = psutil.Process()
        while not self.stop.is_set():
            rss, cpu_delta = 0, 0.0
            for process in [root, *root.children(recursive=True)]:
                try:
                    cpu = process.cpu_times()
                    key = (process.pid, process.create_time())
                    total = cpu.user + cpu.system
                    cpu_delta += max(0.0, total - self.seen.get(key, total))
                    self.seen[key] = total
                    rss += process.memory_info().rss
                except psutil.Error:
                    continue
            self.rows.append({"time": time.perf_counter(), "rss_bytes": rss, "cpu_delta_s": cpu_delta})
            self.stop.wait(0.02)

    def __enter__(self):
        self.start = time.perf_counter()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.wall = time.perf_counter() - self.start
        self.path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in self.rows))

    def metrics(self):
        return {"sampled_tree_peak_rss_bytes": max((r["rss_bytes"] for r in self.rows), default=0),
                "sampled_cpu_core_percent": 100 * sum(r["cpu_delta_s"] for r in self.rows) / self.wall,
                "sample_interval_s": 0.02, "sample_count": len(self.rows)}


def snapshot(model, optimizer, step):
    import torch
    return {"model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(), "step": step}


def tensor_digest(value):
    """Serialization-independent digest includes Adam tensors and RNG bytes."""
    import torch
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(str((str(item.dtype), list(item.shape))).encode())
            h.update(item.detach().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                h.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for member in item:
                visit(member)
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()


def publish_and_die(pointer, policy_id):
    # A real publisher-process exit after fsync+replace, before acknowledgment.
    path = Path(pointer)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump({"policy_id": policy_id}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    os._exit(74)


class ModelHooks:
    def __init__(self, model_dir, directory, pool, eval_pool, guarded, seed, trace, signer, fault="none"):
        import torch
        self.model_dir, self.directory = model_dir, directory
        self.pool, self.eval_pool, self.guarded = pool, eval_pool, guarded
        self.seed, self.trace, self.signer, self.fault = seed, trace, signer, fault
        self.model, self.tokenizer, self.choices = load_model(model_dir)
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=0.001, weight_decay=0)
        self.step, self.active = 0, "policy-0"
        self.train_calls, self.optimizer_steps, self.evaluations = 0, 0, []
        self.evaluation_started = asyncio.Event()
        self.store = TrustedCheckpointStore(directory / "checkpoints") if guarded else None
        self.save("policy-0")
        self.initial_state_digest = tensor_digest(snapshot(self.model, self.optimizer, self.step))
        write_json(directory / "active.json", {"policy_id": self.active})

    def active_policy(self):
        return self.active, 1 if self.active == "policy-0" else 2

    def save(self, policy_id):
        import torch
        with self.trace.span("checkpoint_save"):
            path = self.directory / f"{policy_id}.pt"
            torch.save(snapshot(self.model, self.optimizer, self.step), path)
            if self.store:
                self.store.register_file(policy_id, path)
                path.unlink()

    def read(self, policy_id):
        import torch
        if self.store:
            raw, _ = self.store.read_verified(policy_id)
        else:
            raw = (self.directory / f"{policy_id}.pt").read_bytes()
        return torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")

    def restore(self, policy_id):
        import torch
        state = self.read(policy_id)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        self.step = state["step"]
        return tensor_digest(snapshot(self.model, self.optimizer, self.step))

    def recover_fresh(self, policy_id):
        import torch
        self.model, self.tokenizer, self.choices = load_model(self.model_dir)
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                          lr=0.001, weight_decay=0)
        return self.restore(policy_id)

    def digest(self, policy_id):
        with self.trace.span("checkpoint_hash"):
            return self.store.digest(policy_id) if self.store else file_sha(self.directory / f"{policy_id}.pt")

    async def rollout(self, task, policy_id, policy_version, step):
        start = time.perf_counter()
        data = await owned_job(self.pool, generate, task.prompt, self.seed * 10000 + step,
                               self.fault == "worker_crash" and step == 0)
        group = task.metadata["vare_rollout_group"]
        task = replace(task, metadata={**task.metadata, "rvl_group_id": group})
        version = 0 if self.fault == "policy_stale" else policy_version
        generation = {"prompt_id": task.id, "prompt": task.prompt,
                      "response": "AB"[data["action"]], "logprob": data["logprob"],
                      "metadata": {"prompt_token_ids": data["prompt_ids"],
                                   "response_token_ids": [data["token"]],
                                   "response_token_logprobs": [data["logprob"]],
                                   "sampling_temperature": 1.0}}
        md = {"rvl_generation": generation, "rvl_group_id": group, "rvl_generation_index": 0,
              "vare_rollout_group": group, "vare_rollout_group_size": 1,
              "rvl_generation_behavior_sha256": RVLTokenReplayReader._generation_behavior_digest(generation),
              "rvl_group_behavior_sha256": RVLTokenReplayReader._canonical_digest([generation]),
              "rvl_group_behavior_hash_status": "generation_match", "worker": data}
        attempt = Attempt(task, generation["response"], policy_id, version, step,
                          logprob=data["logprob"], metadata=md)
        with (self.directory / "rollouts.jsonl").open("a") as stream:
            stream.write(json.dumps(asdict(attempt), sort_keys=True) + "\n")
        self.trace.record("rollout", start, **data, policy_lag=policy_version-version)
        return attempt

    async def train_candidate_with_freshness(self, incumbent_id, experiences, assess):
        if self.guarded:
            require(all(assess(e).admitted for e in experiences), "stale at optimizer boundary")
        return await self.train_candidate(incumbent_id, experiences)

    async def train_candidate(self, incumbent_id, experiences):
        import torch
        start = time.perf_counter()
        self.train_calls += 1
        self.restore(incumbent_id)
        before = tensor_digest(self.model.state_dict())
        try:
            losses, rewards, logprobs = [], [], []
            self.optimizer.zero_grad(set_to_none=True)
            for exp in experiences:
                md = exp.attempt.metadata["rvl_generation"]["metadata"]
                ids = torch.tensor([md["prompt_token_ids"]])
                action = self.choices.index(md["response_token_ids"][0])
                lp = self.model(ids).logits[0, -1, self.choices].log_softmax(-1)[action]
                require(abs(float(lp.detach()) - exp.attempt.logprob) < 1e-5, "behavior probability mismatch")
                losses.append(-(exp.verification.score - 0.5) * lp)
                rewards.append(exp.verification.score)
                logprobs.append(float(lp.detach()))
            loss = torch.stack(losses).mean()
            require(bool(torch.isfinite(loss)), "nonfinite loss")
            loss.backward()
            self.optimizer.step()
            self.step += 1
            self.optimizer_steps += 1
            after = tensor_digest(self.model.state_dict())
            require(before != after, "optimizer did not change real model parameters")
            write_json(self.directory / "optimizer_update.json", {
                "loss": float(loss.detach()), "rewards": rewards, "behavior_logprobs": logprobs,
                "parameters_before_sha256": before, "parameters_after_sha256": after,
                "optimizer_state_sha256": tensor_digest(self.optimizer.state_dict()),
                "optimizer_steps": self.optimizer_steps, "batch_size": len(experiences),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in self.model.parameters())})
            if self.fault == "optimizer_exception":
                raise RuntimeError("injected failure after real AdamW step")
            self.save("policy-1")
            self.candidate_state_digest = tensor_digest(snapshot(self.model, self.optimizer, self.step))
            if self.fault == "checkpoint_corrupt":
                path = self.store.artifact_path("policy-1") if self.store else self.directory / "policy-1.pt"
                path.chmod(0o600)
                with path.open("ab") as stream:
                    stream.write(b"injected trailing checkpoint corruption")
            self.restore(incumbent_id)
            return "policy-1"
        except BaseException:
            self.restore(incumbent_id)
            raise
        finally:
            self.trace.record("training", start)

    async def evaluate(self, policy_id):
        start = time.perf_counter()
        self.evaluation_started.set()
        try:
            data = await owned_job(self.eval_pool, evaluate_checkpoint, str(self.model_dir),
                                   str(self.directory), policy_id, self.guarded,
                                   self.signer.evaluation_set_sha256)
            report = EvaluationReport(policy_id, statistics.fmean(data["scores"].values()),
                                      n=len(data["scores"]), metadata={"per_task_scores": data["scores"]})
            if self.guarded:
                report = self.signer.attest(report, checkpoint_sha256=data["checkpoint_sha256"])
            if self.fault == "forged_eval" and policy_id == "policy-1":
                report.primary = 1.0
                report.metadata["per_task_scores"] = dict.fromkeys(data["scores"], 1.0)
            self.evaluations.append({**data, "report": asdict(report)})
            write_json(self.directory / f"evaluation-{policy_id}.json", self.evaluations[-1])
            return report
        finally:
            self.trace.record("evaluation", start, policy_id=policy_id)

    async def promote(self, candidate_id):
        with self.trace.span("promotion"):
            self.restore(candidate_id)
            self.active = candidate_id
            if self.fault == "publication_crash":
                process = CTX.Process(target=publish_and_die, args=(str(self.directory / "active.json"), candidate_id))
                process.start()
                await asyncio.to_thread(process.join, 120)
                require(not process.is_alive(), "publisher failed to terminate")
                write_json(self.directory / "publisher_exit.json", {"exitcode": process.exitcode})
                raise RuntimeError("publisher exited after persisted pointer before acknowledgment")
            write_json(self.directory / "active.json", {"policy_id": candidate_id})

    async def discard(self, candidate_id):
        with self.trace.span("rollback"):
            digest = self.restore("policy-0")
            require(digest == self.initial_state_digest, "rollback model/Adam/RNG mismatch")


class Grader:
    name, version, active_version = "return-A-fixture", 1, 1
    def __init__(self, hooks, reward_root):
        self.hooks, self.root, self.receipts = hooks, reward_root, {}

    async def verify(self, attempt):
        start = time.perf_counter()
        # Optimization grader and independent evaluator have separate code/data.
        passed = attempt.output.strip() == "A"
        group = attempt.metadata["rvl_group_id"]
        verification = Verification(float(passed), passed, 1.0, self.version, self.name, trusted=True,
                                    metadata={"rvl_ready_reward_snapshot": float(passed),
                                              "rvl_replay_status": "ready", "rvl_group_id": group,
                                              "rvl_verification_config_sha256": self.root.verifier_config_sha256})
        exp = Experience(attempt, verification, 0, 0, 0.0)
        if self.hooks.guarded:
            self.receipts[self.root.record_key(exp)] = self.root.attest(exp)
        self.hooks.trace.record("verification", start, step=attempt.created_step)
        return verification


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lo, hi = int(index), min(int(index) + 1, len(values)-1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def stage_metrics(rows):
    output = {}
    for stage in sorted({r["stage"] for r in rows}):
        values = [r["duration_s"] for r in rows if r["stage"] == stage]
        output[stage] = {"n": len(values), "sum_s": sum(values), "p50_s": percentile(values, 0.5),
                         "p95_s": percentile(values, 0.95)}
    return output


def tasks_for(seed, count, evaluation=False):
    # Disjoint namespaces; no retained repository cohort is opened.
    rng = random.Random(seed + (1000000 if evaluation else 0))
    prefix = "Audit" if evaluation else "Job"
    return [Task(f"{prefix}-{seed}-{i}", f"{prefix} {rng.getrandbits(64):016x}: return A. Answer:", "transport")
            for i in range(count)]


async def execute_case(model_dir, directory, pool, eval_pool, mode, seed, workers, count=32, eval_count=32, fault="none"):
    import torch
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    guarded = mode == "guarded"
    trace = Trace(directory / "trace.jsonl")
    train_tasks, eval_tasks = tasks_for(seed, count), tasks_for(seed, eval_count, True)
    require(not ({t.prompt for t in train_tasks} & {t.prompt for t in eval_tasks}), "task overlap")
    write_json(directory / "training_tasks.json", [asdict(t) for t in train_tasks])
    write_json(directory / "evaluation_tasks.json", [asdict(t) for t in eval_tasks])
    signer = EvidenceTrustRoot(os.urandom(32), file_sha(PROTOCOL), file_sha(__file__), uuid.uuid4().hex,
                               file_sha(directory / "evaluation_tasks.json"), uuid.uuid4().hex, True)
    reward_root = RewardEvidenceRoot(os.urandom(32), signer.run_id, file_sha(PROTOCOL), file_sha(__file__), sha256_json({"grader": "return-A", "version": 1}))
    hooks = ModelHooks(model_dir, directory, pool, eval_pool, guarded, seed, trace, signer, fault)
    grader = Grader(hooks, reward_root)
    # Some fail-closed paths legitimately emit no engine event.
    (directory / "engine.jsonl").touch()
    journal = PromotionJournal(directory / "promotion.sqlite", initial_policy_id="policy-0",
                               initial_checkpoint_sha256=hooks.digest("policy-0")) if guarded else None
    loop = CapabilityLoop(hooks=hooks, verifier=grader,
                          config=EngineConfig(rollout_concurrency=workers, replay_batch_size=count,
                                              lag=LagConfig(max_policy_lag=0 if guarded else 999,
                                                            max_verifier_lag=0 if guarded else 999),
                                              promotion=PromotionConfig(min_primary_gain=0.0, min_eval_examples=eval_count)),
                          seed=seed, event_log=EventLog(directory / "engine.jsonl"),
                          evidence_trust_root=signer if guarded else None,
                          promotion_journal=journal, checkpoint_hasher=hooks.digest if guarded else None,
                          training_evidence_authorizer=RewardEvidenceManifest(reward_root, grader.receipts).verified if guarded else None,
                          require_authenticated_training=guarded)
    # Instrument existing admission/replay functions without replacing semantics.
    assess = loop.lag.assess
    def timed_assess(**kwargs):
        with trace.span("freshness"):
            return assess(**kwargs)
    loop.lag.assess = timed_assess
    sample = loop.replay.sample_current
    def timed_sample(*args, **kwargs):
        with trace.span("replay"):
            if fault == "replay_stale":
                grader.version = grader.active_version = 2
            result = sample(*args, **kwargs)
            if fault == "reward_tamper" and result:
                result[0].verification.score = 1.0 - result[0].verification.score
            return result
    loop.replay.sample_current = timed_sample
    result, error, cancellation_start, failure_detected = None, None, None, None
    with Resources(directory / "resources.jsonl") as resources:
        start = time.perf_counter()
        try:
            task = asyncio.create_task(loop.run_round(train_tasks, round_index=0, rollout_count=count))
            if fault == "evaluation_cancel":
                await asyncio.wait_for(hooks.evaluation_started.wait(), timeout=120)
                cancellation_start = time.perf_counter()
                task.cancel()
            result = await task
        except BaseException as exc:
            failure_detected = time.perf_counter()
            error = {"type": type(exc).__name__, "message": str(exc)}
        round_s = time.perf_counter() - start
        recovery_start = time.perf_counter()
        if fault == "publication_crash" and (directory / "publisher_exit.json").exists():
            # Reopen durable state and reload real model+Adam+RNG, independently
            # of the failed publisher's in-memory acknowledgment.
            pid = json.loads((directory / "active.json").read_text())["policy_id"]
            hooks.active = pid
            recovered_digest = hooks.recover_fresh(pid)
            if journal:
                restarted = PromotionJournal(journal.path, initial_policy_id="policy-0",
                                              initial_checkpoint_sha256=hooks.digest("policy-0"))
                status = restarted.status()
                restarted.reconcile(tx_id=status["pending_tx_id"], observed_policy_id=pid,
                                    observed_checkpoint_sha256=hooks.digest(pid))
                restarted.recover_round(owner_id=status["round_owner_id"], observed_policy_id=pid,
                                        observed_checkpoint_sha256=hooks.digest(pid))
                journal = restarted
            require(recovered_digest == hooks.candidate_state_digest, "publication recovery state mismatch")
        else:
            pid = hooks.active
            recovered_digest = hooks.recover_fresh(pid)
            expected = hooks.candidate_state_digest if pid == "policy-1" else hooks.initial_state_digest
            require(recovered_digest == expected, "active checkpoint reload mismatch")
            if journal and journal.status()["round_owner_id"]:
                journal.recover_round(owner_id=journal.status()["round_owner_id"], observed_policy_id=pid,
                                      observed_checkpoint_sha256=hooks.digest(pid))
        recovery_s = time.perf_counter() - recovery_start
        trace.record("checkpoint_recovery", recovery_start)
        if journal:
            journal.assert_clean(policy_id=hooks.active, checkpoint_sha256=hooks.digest(hooks.active))
        wall_s = time.perf_counter() - start
        recovered_at = time.perf_counter()
    summary = {"mode": mode, "seed": seed, "workers": workers, "fault": fault,
               "round_s": round_s, "wall_s": wall_s, "checkpoint_recovery_s": recovery_s,
               "optimizer_steps": hooks.optimizer_steps, "train_calls": hooks.train_calls,
               "active_policy": hooks.active, "error": error, "state_integrity": True,
               "initial_state_sha256": hooks.initial_state_digest,
               "candidate_state_sha256": getattr(hooks, "candidate_state_digest", None),
               "recovered_state_sha256": recovered_digest,
               "result": asdict(result) if result else None,
               "journal": journal.status() if journal else None, "resources": resources.metrics(),
               "stages": stage_metrics(trace.rows),
               "rollout_tokens": sum(r["stage"] == "rollout" for r in trace.rows),
               "rollout_prompt_tokens": sum(len(r["prompt_ids"]) for r in trace.rows if r["stage"] == "rollout"),
               "policy_lag": [r["policy_lag"] for r in trace.rows if r["stage"] == "rollout"],
               "evaluation_greedy_success": {r["policy_id"]: statistics.fmean(r["greedy"].values()) for r in hooks.evaluations}}
    summary["failure_detected_at"] = failure_detected
    summary["failure_to_checkpoint_ready_s"] = recovered_at - failure_detected if failure_detected else None
    summary["cancellation_to_checkpoint_ready_s"] = recovered_at - cancellation_start if cancellation_start else None
    rollouts = [r for r in trace.rows if r["stage"] == "rollout"]
    summary["observed_worker_pids"] = sorted({r["worker_pid"] for r in rollouts})
    concurrency, maximum = 0, 0
    intervals = [(r["service_start"], 1) for r in rollouts] + [(r["service_end"], -1) for r in rollouts]
    for _, delta in sorted(intervals):
        concurrency += delta
        maximum = max(maximum, concurrency)
    summary["max_worker_service_concurrency"] = maximum
    summary["tasks_s"] = summary["tokens_s"] = summary["rollout_tokens"] / wall_s
    screened = loop.events.counters["replay_experiences_screened"]
    summary["replay_screen_event_count"] = screened
    events = [json.loads(s) for s in (directory / "engine.jsonl").read_text().splitlines()]
    summary["replay_rejected_count"] = sum(e.get("count", 0) for e in events if e["event"] == "replay_experiences_screened")
    summary["replay_rejection_rate"] = summary["replay_rejected_count"] / max(1, len(loop.replay))
    if result:
        # Independent arithmetic oracle, not another call to PromotionGate.
        inc, cand = result.incumbent_eval, result.candidate_eval
        metric_ok = cand.n == eval_count and cand.primary >= inc.primary
        trusted = not (guarded and fault == "forged_eval")
        summary["promotion_correct"] = result.decision.accepted == (metric_ok and trusted)
    else:
        summary["promotion_correct"] = hooks.active == "policy-0" or fault == "publication_crash"
    write_json(directory / "summary.json", summary)
    require(summary["resources"]["sampled_tree_peak_rss_bytes"] < 12 * 1024**3, "RSS ceiling exceeded")
    return summary


def fault_oracle(row):
    fault, guarded = row["fault"], row["mode"] == "guarded"
    if fault in {"policy_stale", "replay_stale", "reward_tamper"}:
        return row["optimizer_steps"] == (0 if guarded else 1) and row["active_policy"] == ("policy-0" if guarded else row["active_policy"])
    if fault == "forged_eval":
        return row["optimizer_steps"] == 1 and row["active_policy"] == ("policy-0" if guarded else "policy-1")
    if fault == "checkpoint_corrupt":
        return bool(row["error"]) if guarded else row["error"] is None
    if fault in {"optimizer_exception", "evaluation_cancel"}:
        return row["optimizer_steps"] == 1 and row["active_policy"] == "policy-0" and row["error"] is not None
    if fault == "worker_crash":
        return row["optimizer_steps"] == 0 and row["error"] is not None and row.get("worker_restart_s", 0) > 0
    if fault == "publication_crash":
        return row["active_policy"] == "policy-1" and row["error"] is not None and row["state_integrity"]
    return row["error"] is None and row["optimizer_steps"] == 1 and row["promotion_correct"]


def aggregate(rows):
    output = []
    for mode, workers in sorted({(r["mode"], r["workers"]) for r in rows if r["fault"] == "none"}):
        group = [r for r in rows if (r["mode"], r["workers"], r["fault"]) == (mode, workers, "none")]
        metrics = {}
        for key in ("tasks_s", "tokens_s", "wall_s", "checkpoint_recovery_s"):
            values = [r[key] for r in group]
            metrics[key] = {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values)>1 else None,
                            "min": min(values), "max": max(values)}
        output.append({"mode": mode, "workers": workers, "n": len(group), "metrics": metrics})
    return {"normal": output, "fault_cases": sum(r["fault"] != "none" for r in rows),
            "oracle_passes": sum(fault_oracle(r) for r in rows), "case_count": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1701, 1702, 1703])
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--eval-tasks", type=int, default=32)
    parser.add_argument("--skip-faults", action="store_true")
    args = parser.parse_args()
    require(all(w in (1, 2, 4, 8) for w in args.workers) and args.tasks > 0 and args.eval_tasks > 0, "invalid workload")
    args.output.mkdir(parents=True, exist_ok=False)
    torch_setup()
    campaign_start = time.perf_counter()
    if args.model_dir is None:
        from huggingface_hub import snapshot_download
        args.model_dir = Path(snapshot_download(MODEL, revision=REVISION,
                              allow_patterns=["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]))
    args.model_dir = args.model_dir.resolve()
    expected_files = json.loads(PROTOCOL.read_text())["model_files"]
    require(all(file_sha(args.model_dir / name) == digest for name, digest in expected_files.items()),
            "local model differs from pinned revision bytes")
    sources = [Path(__file__), PROTOCOL, ROOT / "requirements/e2e-rl-real-model-v1.txt",
               ROOT / "scripts/audit_e2e_rl_real_model.py", ROOT / "tests/test_e2e_rl_real_model.py",
               *[ROOT / "src/vare" / name for name in
               ("engine.py", "lag.py", "replay.py", "promotion.py", "attestation.py", "reward_evidence.py", "checkpoint_artifacts.py", "promotion_journal.py")]]
    metadata = {"schema": "vare.e2e-rl-real-model.v1", "started_unix": time.time(), "platform": platform.platform(),
                "python": sys.version, "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"], text=True).splitlines() if sys.platform == "darwin" else platform.machine(),
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "model": MODEL, "revision": REVISION,
                "model_files": {p.name: file_sha(p) for p in args.model_dir.iterdir() if p.is_file()},
                "sources": {str(p.relative_to(ROOT)): file_sha(p) for p in sources},
                "dependencies": sorted(f"{d.metadata['Name']}=={d.version}" for d in import_metadata.distributions()),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "PYTHONHASHSEED")},
                "timing_boundary": "warmed worker pools/model setup excluded; round and exact checkpoint recovery included", "sampling_rss_limit_bytes": 12 * 1024**3}
    write_json(args.output / "manifest.json", metadata)
    for source in sources:
        target = args.output / "sources" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    # Pay first autograd/Adam initialization once before timed comparisons.
    import torch
    warm_model, warm_tokenizer, warm_choices = load_model(args.model_dir)
    warm_optimizer = torch.optim.AdamW([p for p in warm_model.parameters() if p.requires_grad], lr=0.001, weight_decay=0)
    warm_ids = torch.tensor([warm_tokenizer.encode("Warmup: return A. Answer:")])
    (-warm_model(warm_ids).logits[0, -1, warm_choices].log_softmax(-1)[0]).backward()
    warm_optimizer.step()
    del warm_optimizer, warm_model, warm_tokenizer, warm_ids
    rows, cold = [], []
    eval_pool, cold_s = pool_start(args.model_dir, 2)
    cold.append({"pool": "evaluation", "workers": 2, "startup_s": cold_s})
    try:
        for workers in args.workers:
            pool, cold_s = pool_start(args.model_dir, workers)
            cold.append({"pool": "rollout", "workers": workers, "startup_s": cold_s})
            try:
                for index, seed in enumerate(args.seeds):
                    # Alternating order counterbalances warm cache/thermal drift.
                    for mode in (["minimal", "guarded"] if index % 2 == 0 else ["guarded", "minimal"]):
                        directory = args.output / f"w{workers}-s{seed}-{mode}"
                        row = asyncio.run(execute_case(args.model_dir, directory, pool, eval_pool, mode, seed, workers, args.tasks, args.eval_tasks))
                        row["directory"] = directory.name
                        rows.append(row)
                        write_json(args.output / "progress.json", rows)
                        print(json.dumps({k: row[k] for k in ("directory", "tasks_s", "optimizer_steps", "active_policy", "error")}), flush=True)
                        require(time.perf_counter()-campaign_start < 3600, "campaign deadline exceeded")
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        if not args.skip_faults:
            pool, _ = pool_start(args.model_dir, 2)
            try:
                for seed in args.seeds:
                    for fault in json.loads(PROTOCOL.read_text())["faults"]:
                        for mode in ("minimal", "guarded"):
                            directory = args.output / f"fault-{fault}-s{seed}-{mode}"
                            row = asyncio.run(execute_case(args.model_dir, directory, pool, eval_pool, mode, seed, 2, args.tasks, args.eval_tasks, fault))
                            if fault == "worker_crash":
                                restart = time.perf_counter()
                                pool.shutdown(wait=True, cancel_futures=True)
                                pool, _ = pool_start(args.model_dir, 2)
                                pool.submit(ready).result(timeout=120)
                                row["worker_restart_s"] = time.perf_counter() - restart
                                row["failure_to_serving_ready_s"] = time.perf_counter() - row["failure_detected_at"]
                            row["directory"] = directory.name
                            row["fault_oracle_pass"] = fault_oracle(row)
                            write_json(directory / "summary.json", row)
                            rows.append(row)
                            write_json(args.output / "progress.json", rows)
                            print(json.dumps({k: row.get(k) for k in ("directory", "optimizer_steps", "active_policy", "fault_oracle_pass", "error")}), flush=True)
                            require(time.perf_counter()-campaign_start < 3600, "campaign deadline exceeded")
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
    finally:
        eval_pool.shutdown(wait=True, cancel_futures=True)
    write_json(args.output / "cold_start.json", cold)
    write_json(args.output / "summary.json", {**aggregate(rows), "campaign_wall_s": time.perf_counter()-campaign_start})
    files = {str(p.relative_to(args.output)): file_sha(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    write_json(args.output / "SHA256SUMS.json", files)
    require(all(fault_oracle(row) for row in rows), "at least one case failed its predeclared oracle; retain all evidence")
    print(f"PASS {len(rows)} measured cases; {time.perf_counter()-campaign_start:.3f}s campaign", flush=True)


if __name__ == "__main__":
    main()
