"""Five actual learner-process SIGKILL and durable-recovery cutpoints.

The cases reuse one fresh, independently verified TinyStories rollout group
from the frozen matched experiment. Each fault case is an isolated transaction;
the same source group is reused only to control the cutpoint comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/verifier_consistent_async_grpo_v2/protocol.json"
CUTPOINTS = (
    "rollout_complete_before_admission",
    "admission_before_optimizer",
    "optimizer_computed_before_commit",
    "commit_before_acknowledgment",
    "evaluation_before_promotion",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(json.dumps(value, sort_keys=True).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def prepare_pinned_imports(rvl_source: Path) -> None:
    """Make isolated recovery workers import only the frozen source trees."""
    import site
    for entry in site.getsitepackages():
        if entry not in sys.path:
            sys.path.append(entry)
    for finder in list(sys.meta_path):
        mapping = getattr(finder, "MAPPING", None)
        if isinstance(mapping, dict) and ({"vare", "rvl_systems"} & set(mapping)):
            sys.meta_path.remove(finder)
    for name in ("vare", "rvl_systems"):
        module = sys.modules.get(name)
        if module is not None:
            expected = (ROOT / "src/vare/__init__.py" if name == "vare"
                        else rvl_source / "src/rvl_systems/__init__.py")
            require(Path(module.__file__).resolve() == expected.resolve(),
                    f"preloaded {name} came from outside pinned source")
    sys.path.insert(0, str(rvl_source / "src"))
    sys.path.insert(0, str(ROOT / "src"))


def verify_import_origins(rvl_source: Path) -> None:
    from importlib import import_module
    expected = {
        "vare": ROOT / "src/vare/__init__.py",
        "vare.attestation": ROOT / "src/vare/attestation.py",
        "vare.promotion": ROOT / "src/vare/promotion.py",
        "vare.promotion_journal": ROOT / "src/vare/promotion_journal.py",
        "vare.trajectory_store": ROOT / "src/vare/trajectory_store.py",
        "vare.trajectory_admission": ROOT / "src/vare/trajectory_admission.py",
        "rvl_systems": rvl_source / "src/rvl_systems/__init__.py",
        "rvl_systems.hf_backend": rvl_source / "src/rvl_systems/hf_backend.py",
        "rvl_systems.hf_trainer": rvl_source / "src/rvl_systems/hf_trainer.py",
    }
    for name, path in expected.items():
        module = import_module(name)
        require(Path(module.__file__).resolve() == path.resolve(),
                f"import origin mismatch for {name}: {module.__file__}")


def trusted_score(response: str, version: int) -> float:
    """Independent copy of the two frozen task rules."""
    if version == 0:
        return float(any(character in response for character in ".!?"))
    if version == 1:
        return float(re.search(r"(?<![A-Za-z0-9_])ground(?![A-Za-z0-9_])",
                               response, re.IGNORECASE) is not None)
    raise ValueError(f"unknown verifier version: {version}")


def load_initial_model(args, case: Path):
    prepare_pinned_imports(args.rvl_source)
    import torch
    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from vare.promotion_journal import PromotionJournal
    from vare.types import Task
    from validate_grpo_e2e_cpu import tensor_hash
    verify_import_origins(args.rvl_source)

    torch.set_num_threads(1)
    backend = HFLocalBackend(str(args.model_dir), max_new_tokens=3,
                             device="cpu", precision="fp32")
    model = backend.model
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("transformer.wpe.")
                                 or name.startswith("transformer.ln_f."))
    trainer = HFCausalLMGRPOTrainer(model,
        config=HFTTrainerConfig(learning_rate=1e-4))
    initial_path = case / "checkpoints/policy-0.pt"
    state = torch.load(initial_path, weights_only=True)
    trainer.restore_training_state(state)
    eval_tasks = [Task(
        f"heldout-{args.seed}-{index}",
        f"Held-out landing rule {index}: a blue bird landed on the",
        "landing",
    ) for index in range(4)]
    evaluation_model = copy.deepcopy(model)
    return {
        "torch": torch,
        "trainer": trainer,
        "model": model,
        "backend": backend,
        "journal_class": PromotionJournal,
        "tensor_hash": tensor_hash,
        "eval_tasks": eval_tasks,
        "evaluation_model": evaluation_model,
    }


def load_rows(run: Path) -> tuple[list[dict], dict]:
    raw = json.loads((run / "sync/rollouts.json").read_text())["fresh"]
    from vare.types import Attempt, Task
    rows = []
    for item in raw["members"]:
        attempt_raw = item["attempt"]
        task = Task(**attempt_raw["task"])
        attempt = Attempt(
            task=task, output=attempt_raw["output"],
            policy_id=attempt_raw["policy_id"],
            policy_version=attempt_raw["policy_version"],
            created_step=attempt_raw["created_step"],
            logprob=attempt_raw["logprob"], latency_ms=attempt_raw["latency_ms"],
            metadata=attempt_raw["metadata"],
        )
        rows.append({"attempt": attempt, "score": float(re.search(
            r"(?<![A-Za-z0-9_])ground(?![A-Za-z0-9_])",
            attempt.output, re.IGNORECASE) is not None),
                     "verifier_version": 1})
    require(len(rows) == 8, "source rollout group does not contain eight members")
    return rows, raw


def ensure_store(case: Path, run: Path, seed: int):
    sys.path.insert(0, str(ROOT / "src"))
    from vare.evidence import sha256_file, sha256_json
    from vare.trajectory_admission import TrajectoryRewardRoot, admit_terminal_grpo_group
    from vare.trajectory_store import PolicyCheckpoint, TrajectoryStore
    from benchmark_verifier_consistent_async_grpo import persist_group

    model_p0 = case / "checkpoints/policy-0.pt"
    policy = PolicyCheckpoint("policy-0", 0, sha256_file(model_p0))
    run_id = f"sigkill-{seed}-{case.name}"
    store_path = case / "trajectories.sqlite"
    reward_key_path = case / "reward-key.bin"
    protocol_sha = sha(PROTOCOL)
    v0_sha = sha256_json({"name": "landing", "version": 0,
                          "rule": "punctuation completeness"})
    v1_sha = sha256_json({"name": "landing", "version": 1,
                          "rule": "canonical ground semantics"})
    verifier_shas = {0: v0_sha, 1: v1_sha}
    rows, _ = load_rows(run)
    task = rows[0]["attempt"].task
    if not store_path.exists():
        store = TrajectoryStore.create(store_path, run_id=run_id, max_inflight=16)
        store.register_policy(policy)
        store.register_verifier("landing", 0, v0_sha)
        store.register_verifier("landing", 1, v1_sha)
        reward_key = os.urandom(32)
        fd = os.open(reward_key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(reward_key)
            stream.flush()
            os.fsync(stream.fileno())
        root = TrajectoryRewardRoot(reward_key, run_id, protocol_sha)
        persist_group(store, "crash-group", rows, policy=policy,
                      reward_root=root, verifier_shas=verifier_shas, task=task)
    else:
        store = TrajectoryStore(store_path)
        root = TrajectoryRewardRoot(reward_key_path.read_bytes(), run_id, protocol_sha)
    result = admit_terminal_grpo_group(
        store, "crash-group", active_policy=policy, verifier_name="landing",
        verifier_version=1, verifier_config_sha256=v1_sha, trust_root=root,
    )
    require(result.admitted, f"crash recovery source group admission failed: {result.reasons}")
    atomic_json(case / "admission.json", {"group_id": result.group_id,
        "admitted": result.admitted, "heads": result.heads,
        "verifier_versions": result.verifier_lags, "members": len(result.experiences)})
    return store, result, root, verifier_shas


def verified_samples(rows: list[dict]):
    from rvl_systems.types import Generation, VerifiedGeneration
    result = []
    for item in rows:
        raw = item["attempt"].metadata["rvl_generation"]
        generation = Generation(**raw)
        result.append(VerifiedGeneration(
            generation=generation, reward=float(item["score"]),
            verifier_latency_s=0.0, verifier_version=1,
            metadata={"verifier_name": "landing", "trusted": True}))
    return result


def save_state(torch, path: Path, state) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    started = time.perf_counter()
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return time.perf_counter() - started


def independent_eval(model, tokenizer, state, tasks, version: int):
    import torch
    from vare.types import EvaluationReport
    model.load_state_dict(state["model"])
    model.eval()
    scores, outputs = {}, {}
    with torch.no_grad():
        for task in tasks:
            prompt_ids = tokenizer.encode(task.prompt)
            generated = []
            for _ in range(3):
                token = int(model(torch.tensor([prompt_ids + generated])).logits[0, -1].argmax())
                generated.append(token)
                if token == tokenizer.eos_token_id:
                    break
            text = tokenizer.decode(generated, skip_special_tokens=True)
            outputs[task.id] = {"token_ids": generated, "text": text}
        scores[task.id] = trusted_score(text, version)
    from statistics import fmean
    primary = fmean(scores.values())
    report = EvaluationReport("policy-1", primary, {"landing": primary}, 1.0,
                              0.0, len(scores), {"per_task_scores": scores,
                                                "verifier_version": version})
    return report, {"outputs": outputs, "scores": scores}


def pause_at(case: Path, cutpoint: str, *, resume: bool) -> None:
    if resume:
        return
    atomic_json(case / "cutpoint.json", {"cutpoint": cutpoint,
        "pid": os.getpid(), "pgid": os.getpgrp(), "time": time.perf_counter()})
    os.kill(os.getpid(), signal.SIGSTOP)
    raise RuntimeError("learner resumed after supervisor stop instead of SIGKILL")


def worker(args) -> None:
    case = args.case
    case.mkdir(parents=True, exist_ok=True)
    progress_path = case / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"phase": "new"}
    source_checkpoint = args.source_run / "sync/checkpoints/policy-0.pt"
    checkpoint0 = case / "checkpoints/policy-0.pt"
    checkpoint0.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint0.exists():
        temporary = checkpoint0.with_suffix(checkpoint0.suffix + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(source_checkpoint.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, checkpoint0)
    source_rollouts = args.source_run / "sync/rollouts.json"
    rollout_path = case / "rollouts.json"
    if not rollout_path.exists():
        temporary = rollout_path.with_suffix(rollout_path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(source_rollouts.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, rollout_path)
    if progress["phase"] == "new":
        progress = {"phase": "rollouts_complete", "rollout_source_sha256": sha(rollout_path)}
        atomic_json(progress_path, progress)
    if args.cutpoint == CUTPOINTS[0] and progress["phase"] == "rollouts_complete":
        pause_at(case, CUTPOINTS[0], resume=args.resume)

    store, admission, reward_root, verifier_shas = ensure_store(case, args.source_run, args.seed)
    if progress["phase"] in {"new", "rollouts_complete"}:
        progress = {**progress, "phase": "admitted", "admission_heads": admission.heads}
        atomic_json(progress_path, progress)
    if args.cutpoint == CUTPOINTS[1] and progress["phase"] == "admitted":
        pause_at(case, CUTPOINTS[1], resume=args.resume)

    runtime = load_initial_model(args, case)
    torch, trainer, model, backend = runtime["torch"], runtime["trainer"], runtime["model"], runtime["backend"]
    rows, _ = load_rows(args.source_run)
    samples = verified_samples(rows)
    journal_path = case / "promotion.sqlite"
    initial_sha = sha(checkpoint0)
    journal = runtime["journal_class"](journal_path, initial_policy_id="policy-0",
                                       initial_checkpoint_sha256=initial_sha)
    candidate_path = case / "checkpoints/policy-1.pt"

    if progress["phase"] == "committed" and args.cutpoint == CUTPOINTS[3]:
        committed = json.loads((case / "candidate-state.json").read_text())
        state = torch.load(candidate_path, weights_only=True)
        trainer.restore_training_state(state)
        require(sha(candidate_path) == committed["candidate_sha256"]
                and journal.status()["active_checkpoint_sha256"] == committed["candidate_sha256"],
                "committed checkpoint and journal differ after restart")
        require(runtime["tensor_hash"](trainer.snapshot_training_state()["model"])
                == committed["model_tensor_sha256"],
                "model parameters differ after committed-process restart")
        store.close_group("crash-group", admission.heads, reason="consumed",
                          decision={"recovered_commit": committed["transaction_id"]})
        atomic_json(progress_path, {**progress, "phase": "acknowledged"})
        call_count = {"value": 0}
        io_s = 0.0
    else:
        if progress["phase"] == "evaluated_unpromoted" and args.resume:
            unpromoted = case / "checkpoints/policy-1-attempt1-unpromoted.pt"
            if candidate_path.exists() and not unpromoted.exists():
                os.replace(candidate_path, unpromoted)
            atomic_json(case / "unpromoted-candidate.json", {
                "checkpoint": unpromoted.name, "sha256": sha(unpromoted),
                "reason": "process died after evaluation and before promotion; restore incumbent and retry"})
            progress = {**progress, "phase": "admitted"}
            atomic_json(progress_path, progress)

        attempt_number = sum(1 for line in (case / "optimizer-attempts.jsonl").read_text().splitlines()
                             if line.strip()) if (case / "optimizer-attempts.jsonl").exists() else 0
        # Restore the frozen incumbent for every uncommitted retry.
        initial = torch.load(checkpoint0, weights_only=True)
        trainer.restore_training_state(initial)
        call_count = {"value": 0}
        original_step = trainer.optimizer.step
        def count_step(*a, **kw):
            call_count["value"] += 1
            return original_step(*a, **kw)
        trainer.optimizer.step = count_step
        started = time.perf_counter()
        trainer.train_step(samples)
        state = trainer.snapshot_training_state()
        attempt_number += 1
        append_jsonl(case / "optimizer-attempts.jsonl", {
            "attempt": attempt_number, "optimizer_calls": call_count["value"],
            "model_tensor_sha256": runtime["tensor_hash"](state["model"]),
            "optimizer_state": "volatile_until_checkpoint_save",
            "elapsed_s": time.perf_counter() - started})
        if args.cutpoint == CUTPOINTS[2] and not args.resume:
            atomic_json(case / "volatile-candidate.json", {
                "attempt": attempt_number,
                "model_tensor_sha256": runtime["tensor_hash"](state["model"]),
                "checkpoint_saved": False, "journal_committed": False})
            pause_at(case, CUTPOINTS[2], resume=False)
        io_s = save_state(torch, candidate_path, state)
        tasks = runtime["eval_tasks"]
        eval_model = runtime["evaluation_model"]
        report_old, detail_old = independent_eval(eval_model, backend.tokenizer,
            initial, tasks, version=1)
        report_new, detail_new = independent_eval(eval_model, backend.tokenizer,
            state, tasks, version=1)
        from vare.attestation import EvidenceTrustRoot
        from vare.config import PromotionConfig
        from vare.evidence import sha256_json
        from vare.promotion import PromotionGate
        evaluation_key_path = case / "evaluation-key.bin"
        if not evaluation_key_path.exists():
            fd = os.open(evaluation_key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(os.urandom(32))
        evaluator_sha = sha256_json({"verifier": "landing-v1-independent-eval"})
        root = EvidenceTrustRoot(
            evaluation_key_path.read_bytes(), sha(PROTOCOL), evaluator_sha,
            f"sigkill-{args.seed}-{case.name}",
            sha256_json([asdict(task) for task in tasks]),
            f"promotion-nonce-{args.seed}-{case.name}-v1", True)
        report_old.policy_id = "policy-0"
        report_new.policy_id = "policy-1"
        signed_old = root.attest(report_old, checkpoint_sha256=initial_sha)
        candidate_sha = sha(candidate_path)
        signed_new = root.attest(report_new, checkpoint_sha256=candidate_sha)
        gate = PromotionGate(PromotionConfig(
            min_primary_gain=-1.0, max_slice_regression=1.0,
            max_verifier_disagreement=1.0, min_eval_examples=len(tasks),
            max_relative_cost_increase=1e6), trust_root=root)
        decision = gate.decide(signed_old, signed_new,
                               expected_verifier_version=1)
        require(decision.accepted, f"crash-case promotion gate rejected: {decision.reasons}")
        eval_record = {"incumbent": asdict(signed_old), "candidate": asdict(signed_new),
                       "incumbent_detail": detail_old, "candidate_detail": detail_new,
                       "decision": asdict(decision)}
        atomic_json(case / "candidate-evaluation.json", eval_record)
        if args.cutpoint == CUTPOINTS[4] and not args.resume:
            progress = {**progress, "phase": "evaluated_unpromoted",
                        "candidate_sha256": candidate_sha}
            atomic_json(progress_path, progress)
            pause_at(case, CUTPOINTS[4], resume=False)

        evaluation_sha = sha256_json(eval_record)
        tx_id = f"promotion-tx-{args.seed}-{case.name}"
        journal.prepare(tx_id=tx_id, incumbent_id="policy-0",
            incumbent_sha256=initial_sha, candidate_id="policy-1",
            candidate_sha256=candidate_sha, evaluation_sha256=evaluation_sha)
        atomic_json(case / "active.json", {"policy_id": "policy-1",
                                            "checkpoint_sha256": candidate_sha})
        journal.commit(tx_id)
        committed = {"transaction_id": tx_id, "candidate_sha256": candidate_sha,
                     "model_tensor_sha256": runtime["tensor_hash"](state["model"]),
                     "checkpoint_io_s": io_s}
        atomic_json(case / "candidate-state.json", committed)
        progress = {**progress, "phase": "committed", **committed}
        atomic_json(progress_path, progress)
        if args.cutpoint == CUTPOINTS[3] and not args.resume:
            pause_at(case, CUTPOINTS[3], resume=False)
        store.close_group("crash-group", admission.heads, reason="consumed",
                          decision={"transaction_id": tx_id})
        atomic_json(progress_path, {**progress, "phase": "acknowledged"})

    snapshot = case / "trajectory-snapshot.sqlite"
    store.backup(snapshot)
    from vare.trajectory_store import audit_trajectory_store
    audit = audit_trajectory_store(snapshot)
    status = journal.status()
    require(status["active_policy_id"] == "policy-1"
            and status["pending_tx_id"] is None,
            "post-recovery promotion journal is not clean")
    require(store.counts().get("consumed", 0) == 8,
            "post-recovery trajectory group was not acknowledged exactly once")
    atomic_json(case / "worker-summary.json", {
        "cutpoint": args.cutpoint, "seed": args.seed,
        "phase": json.loads(progress_path.read_text())["phase"],
        "optimizer_attempts": sum(1 for line in (case / "optimizer-attempts.jsonl").read_text().splitlines() if line.strip()),
        "committed_updates": 1, "effective_consumed_trajectories": 8,
        "store_counts": store.counts(), "store_audit": audit,
        "promotion_journal": status, "candidate_sha256": sha(candidate_path),
        "candidate_model_tensor_sha256": runtime["tensor_hash"](
            torch.load(candidate_path, weights_only=True)["model"]),
        "main_sync_candidate_sha256": sha(args.source_run / "sync/checkpoints/policy-1.pt"),
        "main_sync_candidate_model_matches": runtime["tensor_hash"](
            torch.load(candidate_path, weights_only=True)["model"])
            == runtime["tensor_hash"](torch.load(
                args.source_run / "sync/checkpoints/policy-1.pt", weights_only=True)["model"]),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "actual_optimizer_step_calls_this_process": call_count["value"] if "call_count" in locals() else 0,
    })


def parent(args) -> None:
    require(args.source_run.is_dir(), "source matched run does not exist")
    results = []
    harness_started = time.perf_counter()
    for cutpoint in CUTPOINTS:
        atomic_json(args.output / "progress.json", {
            "status": "running", "completed_cutpoints": [row["cutpoint"] for row in results],
            "active_cutpoint": cutpoint, "elapsed_s": time.perf_counter() - harness_started,
        })
        case = args.output / cutpoint
        require(not case.exists(), f"crash case already exists: {case}")
        case.mkdir(parents=True)
        command = [sys.executable, "-S", str(Path(__file__).resolve()), "--worker",
                   "--case", str(case), "--source-run", str(args.source_run),
                   "--model-dir", str(args.model_dir), "--rvl-source", str(args.rvl_source),
                   "--output", str(args.output), "--cutpoint", cutpoint,
                   "--seed", str(args.seed)]
        stdout_path, stderr_path = case / "learner.stdout.log", case / "learner.stderr.log"
        started = time.perf_counter()
        process = None
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            try:
                deadline = time.monotonic() + args.timeout
                marker = case / "cutpoint.json"
                while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                require(marker.exists(), f"learner failed to reach cutpoint {cutpoint}; inspect {stderr_path}")
                proof = json.loads(marker.read_text())
                require(proof["pid"] == process.pid and proof["pgid"] == process.pid
                        and proof["cutpoint"] == cutpoint, "cutpoint process identity mismatch")
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=30)
                require(returncode == -signal.SIGKILL, "learner did not exit from SIGKILL")
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
                raise
        restart_started = time.perf_counter()
        with (case / "restart.stdout.log").open("wb") as stdout, (case / "restart.stderr.log").open("wb") as stderr:
            resumed = subprocess.run(command + ["--resume"], stdout=stdout,
                                      stderr=stderr, timeout=args.timeout, check=False)
        require(resumed.returncode == 0,
                f"recovery process failed for {cutpoint}; inspect {case / 'restart.stderr.log'}")
        summary = json.loads((case / "worker-summary.json").read_text())
        optimizer_attempts = summary["optimizer_attempts"]
        expected_attempts = 2 if cutpoint in {CUTPOINTS[2], CUTPOINTS[4]} else 1
        require(optimizer_attempts == expected_attempts,
                f"unexpected optimizer recomputation count for {cutpoint}")
        require(summary["peak_rss_bytes"] <= 8 * 1024**3,
                f"learner exceeded 8 GiB RSS at {cutpoint}")
        require(summary["main_sync_candidate_model_matches"],
                f"recovered checkpoint differs from independent matched candidate: {cutpoint}")
        control = {"cutpoint": cutpoint, "pid": process.pid,
                   "process_group_id": proof["pgid"], "signal_returncode": returncode,
                   "marker": proof, "restart_returncode": resumed.returncode,
                   "recovery_wall_s": time.perf_counter() - restart_started,
                   "case_wall_s": time.perf_counter() - started,
                   "old_process_alive": _alive(process.pid),
                   "worker_summary": summary}
        require(not control["old_process_alive"], "old SIGKILLed learner remains alive")
        atomic_json(case / "supervisor-proof.json", control)
        results.append(control)
    report = {
        "schema": "vare.verifier-consistent-async-grpo-sigkill.v1",
        "source_run": str(args.source_run), "seed": args.seed,
        "cutpoints": results,
        "process_kills": len(results),
        "harness_wall_s": time.perf_counter() - harness_started,
        "all_sigkill_verified": all(row["signal_returncode"] == -signal.SIGKILL for row in results),
        "all_recoveries_exact": all(row["worker_summary"]["main_sync_candidate_model_matches"] for row in results),
        "scope": "Five isolated actual Python learner-process SIGKILL cases reuse one independently verified TinyStories group. This measures transaction recovery, not throughput parallelism or independent task generalization.",
    }
    atomic_json(args.output / "summary.json", report)
    atomic_json(args.output / "progress.json", {
        "status": "complete", "completed_cutpoints": [row["cutpoint"] for row in results],
        "active_cutpoint": None, "elapsed_s": report["harness_wall_s"],
    })
    inventory = {
        str(path.relative_to(args.output)): sha(path)
        for path in args.output.rglob("*")
        if path.is_file() and path.name != "SHA256.json"
        and not path.name.endswith(("-wal", "-shm", ".tmp"))
    }
    atomic_json(args.output / "SHA256.json", inventory)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    require(sys.flags.no_site == 1,
            "launch the crash runner with python -S to disable site startup hooks")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rvl-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=82101)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case", type=Path)
    parser.add_argument("--cutpoint", choices=CUTPOINTS)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    require(sha(Path(__file__)) == protocol["crash_runner_sha256"],
            "crash runner differs from frozen protocol")
    require(subprocess.check_output(["git", "-C", str(ROOT), "merge-base", "HEAD",
                                     protocol["vare_base_commit"]], text=True).strip()
            == protocol["vare_base_commit"],
            "VARE base commit differs from frozen protocol")
    for relative, digest in protocol["source_sha256"].items():
        require(sha(ROOT / relative) == digest,
                f"VARE source differs from frozen protocol: {relative}")
    require(args.seed in protocol["budgets"]["seeds"], "undeclared crash seed")
    for filename, digest in protocol["budgets"]["model_files"].items():
        require(sha(args.model_dir / filename) == digest,
                f"model checksum mismatch: {filename}")
    require(subprocess.check_output(
        ["git", "-C", str(args.rvl_source), "rev-parse", "HEAD"], text=True).strip()
        == protocol["budgets"]["rvl_revision"], "RVL revision mismatch")
    require(subprocess.check_output(
        ["git", "-C", str(args.rvl_source), "status", "--porcelain"], text=True).strip()
        == "M src/rvl_systems/hf_backend.py", "unexpected RVL source modifications")
    require(sha(args.rvl_source / "src/rvl_systems/hf_backend.py")
            == protocol["budgets"]["rvl_patched_backend_sha256"],
            "patched RVL backend mismatch")
    if args.worker:
        require(args.case is not None and args.cutpoint is not None,
                "worker requires case and cutpoint")
        worker(args)
    else:
        require(not args.output.exists(), "crash output already exists")
        args.output.mkdir(parents=True)
        parent(args)
    if args.worker:
        print(json.dumps({"case": str(args.case), "cutpoint": args.cutpoint}, indent=2))
    else:
        print(json.dumps(json.loads((args.output / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
