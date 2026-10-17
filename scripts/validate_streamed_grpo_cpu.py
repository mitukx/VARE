"""Real-model streamed-ingestion correctness fixture; no task-success claim."""
from __future__ import annotations
import argparse
import asyncio
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from validate_grpo_e2e_cpu import equal, reference_update, tensor_hash, require, write, sha

ROOT = Path(__file__).resolve().parents[1]


def reward(text):
    return int(hashlib.sha256(("training:" + text).encode()).hexdigest()[:8], 16) / 4294967295


def independent_score(text):
    return (int(hashlib.sha256(("audit:" + text).encode()).hexdigest()[:8], 16) % 7) / 6


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    args.output_created = True
    protocol = json.loads((ROOT / "benchmarks/streamed_grpo_cpu_v3/protocol-v3.1.json").read_text())
    write(args.output / "protocol.json", protocol)
    require(args.seed in protocol["seeds"] and sha(__file__) == protocol["runner_sha256"], "undeclared seed or runner")
    for name, digest in protocol["model_files"].items():
        require(sha(args.model_dir / name) == digest, "model hash mismatch")
    require(subprocess.check_output(["git", "-C", str(args.rvl_source), "rev-parse", "HEAD"], text=True).strip() == protocol["rvl_revision"], "RVL revision mismatch")
    require(subprocess.check_output(["git", "-C", str(args.rvl_source), "status", "--porcelain"], text=True).strip() == "M src/rvl_systems/hf_backend.py", "unexpected RVL edits")
    require(sha(args.rvl_source / "src/rvl_systems/hf_backend.py") == protocol["rvl_patched_backend_sha256"], "RVL patch mismatch")
    source_files = [Path(__file__), ROOT / "scripts/validate_grpo_e2e_cpu.py", *[ROOT / "src/vare" / name for name in ("trajectory_store.py", "trajectory_workers.py", "trajectory_admission.py", "trajectory_learner.py", "engine.py", "integrations/rvl_grpo.py")]]
    for path in source_files:
        target = args.output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(args.rvl_source / "src"))
    import torch
    import transformers
    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from rvl_systems.types import Generation, VerifiedGeneration
    from vare.attestation import EvidenceTrustRoot
    from vare.config import EngineConfig, LagConfig, PromotionConfig, ReplayConfig
    from vare.engine import CapabilityLoop
    from vare.evidence import sha256_file, sha256_json
    from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
    from vare.promotion import PromotionGate
    from vare.promotion_journal import PromotionJournal
    from vare.telemetry import EventLog
    from vare.trajectory_admission import TrajectoryRewardRoot
    from vare.trajectory_learner import TrajectoryLearnerAdapter
    from vare.trajectory_store import AgentStep, PolicyCheckpoint, TrajectoryStore, audit_trajectory_store
    from vare.trajectory_workers import execute_trajectories, verify_trajectories, owned_io
    from vare.types import Task, Verification
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    backend = HFLocalBackend(str(args.model_dir), max_new_tokens=3, device="cpu", precision="fp32")
    model = backend.model
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("transformer.wpe.") or name.startswith("transformer.ln_f."))
    config = HFTTrainerConfig(learning_rate=1e-4)
    trainer = HFCausalLMGRPOTrainer(model, config=config)
    reference_model = copy.deepcopy(model)
    eval_tasks = [Task(f"stream-audit-{args.seed}-{i}", f"Stream checkpoint audit {args.seed}/{i}: A blue bird landed", "audit") for i in range(4)]
    hooks = RVLGRPOHooks(backend=backend, trainer=trainer, eval_tasks=eval_tasks,
        score_fn=lambda t, text: independent_score(text), config=RVLGRPOConfig(seed=args.seed),
        current_verifier_version=0, freshness_lag_config=LagConfig(max_policy_lag=0, max_verifier_lag=0),
        generation_factory=Generation, verified_generation_factory=VerifiedGeneration)
    checkpoint_paths = {}
    def save(policy_id):
        path = args.output / (policy_id + ".pt")
        state = hooks._states[policy_id].training_state
        torch.save(state, path)
        require(equal(state, torch.load(path, weights_only=True)), "checkpoint round trip failed")
        checkpoint_paths[policy_id] = path
        return state
    save("policy-0")
    def policy():
        return PolicyCheckpoint(*hooks.active_policy(), sha256_file(checkpoint_paths[hooks.active_policy()[0]]))
    run_id = f"stream-real-{args.seed}"
    store = TrajectoryStore.create(args.output / "trajectories.db", run_id=run_id, max_inflight=16)
    store.register_policy(policy())
    verifier_sha = sha256_json({"reward_source": sha(__file__), "kind": "synthetic-checksum"})
    store.register_verifier("checksum", 0, verifier_sha)
    root = TrajectoryRewardRoot(os.urandom(32), run_id, sha256_json(protocol))
    evaluation_key = os.urandom(32)
    adapter = TrajectoryLearnerAdapter(store, trust_root=root, current_policy=policy,
        verifier_identity=("checksum", 0, verifier_sha),
        reference_families=protocol["shift_reference_families"], max_members=4)
    hooks.training_evidence_authorizer = adapter.authorizes
    hooks.require_authenticated_training = True
    journal = PromotionJournal(args.output / "promotion.db", initial_policy_id="policy-0", initial_checkpoint_sha256=policy().checkpoint_sha256)
    loop = CapabilityLoop(hooks=hooks, verifier=type("VersionOnly", (), {"active_version": 0})(),
        config=EngineConfig(replay_batch_size=4, replay=ReplayConfig(capacity=4), lag=LagConfig(max_policy_lag=0, max_verifier_lag=0),
                            promotion=PromotionConfig(min_primary_gain=0, min_eval_examples=4, max_relative_cost_increase=1e6)),
        promotion_journal=journal, checkpoint_hasher=lambda pid: sha256_file(checkpoint_paths[pid]),
        training_evidence_authorizer=adapter.authorizes, require_authenticated_training=True,
        # A fresh per-round signer is installed below, with disjoint fixed tasks.
        evidence_trust_root=EvidenceTrustRoot(evaluation_key, sha256_json(protocol), sha(__file__), run_id,
            sha256_json([asdict(t) for t in eval_tasks]), f"stream-{args.seed}-initial", True),
        event_log=EventLog(args.output / "engine.jsonl"), seed=args.seed)
    raw_train, raw_eval = hooks.train_candidate_with_freshness, hooks.evaluate
    records, evaluations, rounds, acknowledgments = [], [], [], []
    optimizer_calls = 0
    raw_opt_step = trainer.optimizer.step
    def counted(*a, **kw):
        nonlocal optimizer_calls
        optimizer_calls += 1
        return raw_opt_step(*a, **kw)
    trainer.optimizer.step = counted
    async def train(incumbent, batch, assessor):
        start = time.perf_counter()
        before = trainer.snapshot_training_state()
        oracle_batch = copy.deepcopy(batch)
        for i, exp in enumerate(oracle_batch):
            exp.attempt.created_step = i  # Numerical oracle uses a unique local locator.
        oracle_start = time.perf_counter()
        ref = reference_update(reference_model, before, oracle_batch, config)
        oracle_s = time.perf_counter()-oracle_start
        captured = {}
        old_step, old_train_step = trainer.optimizer.step, trainer.train_step
        def capture(*a, **kw):
            captured["gradients"] = {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
            return old_step(*a, **kw)
        def train_step(*a, **kw):
            captured["metrics"] = old_train_step(*a, **kw)
            return captured["metrics"]
        trainer.optimizer.step, trainer.train_step = capture, train_step
        try:
            candidate_start = time.perf_counter()
            candidate = await raw_train(incumbent, batch, assessor)
            candidate_s = time.perf_counter()-candidate_start
        finally:
            trainer.optimizer.step, trainer.train_step = old_step, old_train_step
        require(equal(before, trainer.snapshot_training_state()), "candidate did not restore incumbent model/Adam/RNG/step")
        state = save(candidate)
        errors = {"loss": abs(captured["metrics"]["loss"]-ref["loss"]),
                  "gradient": max(float((v-ref["grads"][n]).abs().max()) for n,v in captured["gradients"].items()),
                  "parameter": max(float((v-ref["model"][n]).abs().max()) for n,v in state["model"].items())}
        errors["optimizer"] = max(float((v-ref["optimizer"]["state"][k][n]).abs().max()) for k,values in state["optimizer"]["state"].items() for n,v in values.items() if isinstance(v, torch.Tensor))
        require(max(errors.values()) < 2e-6 and ref["max_behavior_alignment_error"] < 1e-4, "numerical reference mismatch")
        require(torch.equal(before["cpu_rng"], state["cpu_rng"]), "GRPO RNG changed")
        changed = sum(not torch.equal(v,before["model"][n]) for n,v in state["model"].items())
        require(changed > 0, "no actual model update")
        torch.save({"actual_gradients":captured["gradients"], "reference_gradients":ref["grads"]}, args.output/(candidate+"-gradients.pt"))
        records.append({"candidate":candidate,"errors":errors,"behavior_alignment_error":ref["max_behavior_alignment_error"],"changed_tensors":changed,"batch":[asdict(x) for x in batch],"metrics":captured["metrics"],"reference_s":oracle_s,"actual_candidate_s":candidate_s,"wall_s":time.perf_counter()-start})
        return candidate
    hooks.train_candidate_with_freshness = train
    async def evaluate(pid):
        start = time.perf_counter()
        report = await raw_eval(pid)
        saved = torch.load(checkpoint_paths[pid], weights_only=True)
        reference_model.load_state_dict(saved["model"])
        reference_model.eval()
        scores, greedy = {}, {}
        with torch.no_grad():
            for task in eval_tasks:
                ids, output = backend.tokenizer.encode(task.prompt), []
                for _ in range(3):
                    token = int(reference_model(torch.tensor([ids+output])).logits[0,-1].argmax())
                    output.append(token)
                    if token == backend.tokenizer.eos_token_id:
                        break
                greedy[task.id] = {"tokens":output, "text":backend.tokenizer.decode(output, skip_special_tokens=True)}
                scores[task.id] = independent_score(greedy[task.id]["text"])
        require(scores == report.metadata["per_task_scores"], "independent checkpoint decoder differs")
        signed = signer.attest(report, checkpoint_sha256=sha256_file(checkpoint_paths[pid]))
        evaluations.append({"report":asdict(signed),"independent_greedy":greedy,"wall_s":time.perf_counter()-start})
        return signed
    hooks.evaluate = evaluate
    stop = asyncio.Event()
    generation_serial = 0
    async def build(resumed, ident):
        index, state = resumed["next_step"], resumed["state"]
        task = Task(**resumed["spec"]["task"])
        nonlocal generation_serial
        p = policy()
        await owned_io(store.register_policy, p)
        if index == 0:
            value = int(task.metadata["lookup_value"])
            return AgentStep(0, "lookup", p, {"kind":"tool","operation_id":ident+":lookup","semantics":"at_least_once"}, {"lookup_value":value}, {"lookup_value":value})
        terminal = replace(task, prompt=task.prompt+f" Observed lookup value: {state['lookup_value']}.")
        generation_serial += 1
        attempt = await hooks.rollout(terminal,p.policy_id,p.version,generation_serial)
        require(tensor_hash(model.state_dict()) == tensor_hash(hooks._states[p.policy_id].training_state["model"]), "worker loaded wrong checkpoint")
        acknowledgments.append({"trajectory_id":ident,"policy":asdict(p),"tokens":attempt.metadata["token_count"]})
        return AgentStep(1, "answer", p, {"kind":"answer"}, {}, state, attempt.metadata["rvl_generation"])
    async def verify(source, ident):
        row = await owned_io(source.trajectory, ident)
        final = list(store.iter_steps(row["id"]))[-1]
        value = Verification(reward(final["generation"]["response"]), True, 1., 0, "checksum", trusted=True,
            metadata={"trajectory_sha256":row["head"],"verifier_config_sha256":verifier_sha})
        return root.attest(row["id"],row["head"],value)
    workers = [asyncio.create_task(execute_trajectories(store,f"real-worker-{i}",build,stop=stop,lease_seconds=10)) for i in range(2)]
    workers.append(asyncio.create_task(verify_trajectories(store,"reward-worker",verify,stop=stop,lease_seconds=10)))
    started, cpu = time.perf_counter(), time.process_time()
    try:
        for round_index in range(4):
            signer = EvidenceTrustRoot(evaluation_key, sha256_json(protocol), sha(__file__), run_id,
                sha256_json([asdict(t) for t in eval_tasks]), f"stream-{args.seed}-round-{round_index}", True)
            loop.evidence_trust_root = signer
            loop.promotion = PromotionGate(replace(loop.config.promotion, min_primary_gain=1.1 if round_index==1 else 0.), trust_root=signer)
            task = Task(f"stream-train-{args.seed}-{round_index}", f"Stream tool trace {args.seed}/{round_index}: A red fox saw", "train",
                {"lookup_value":(args.seed+round_index)%997,"learning_rule":"terminal_action_on_policy_grpo_v1"})
            for member in range(4):
                await owned_io(store.submit,f"r{round_index}-{member}",task,expected_steps=2,group_id=f"r{round_index}",member=member,group_size=4)
            if round_index==2:
                terminal = adapter._terminal
                def lost_ack(*a, **kw):
                    raise OSError("deterministic ingestion acknowledgment interruption")
                adapter._terminal = lost_ack
            result = None
            start = time.perf_counter()
            while result is None:
                require(time.perf_counter()-start < 120, "streamed round timeout")
                for worker in workers:
                    if worker.done():
                        worker.result()
                try:
                    result = await adapter.advance(loop,round_index=round_index)
                except OSError as exc:
                    recovery_start = time.perf_counter()
                    require(round_index==2 and "acknowledgment interruption" in str(exc), "unexpected failure")
                    require(adapter.pending() is not None, "lost delivery disappeared")
                    active = hooks.active_policy()[0]
                    saved = torch.load(checkpoint_paths[active],weights_only=True)
                    trainer.restore_training_state(saved)
                    require(equal(saved,trainer.snapshot_training_state()), "checkpoint recovery model/optimizer/RNG/step mismatch")
                    adapter._terminal = terminal
                    status = journal.status()
                    require(status["round_owner_id"] is not None, "lost ACK released runtime ownership")
                    journal.recover_round(owner_id=status["round_owner_id"], observed_policy_id=active, observed_checkpoint_sha256=policy().checkpoint_sha256)
                    loop.runtime_coordinator.recover()
                    # A new ingestion owner, same trusted backend after exact restore.
                    adapter = TrajectoryLearnerAdapter(store, trust_root=root,current_policy=policy,
                        verifier_identity=("checksum",0,verifier_sha),
                        reference_families=protocol["shift_reference_families"],max_members=4)
                    loop.training_evidence_authorizer = hooks.training_evidence_authorizer = adapter.authorizes
                    require(adapter.reconcile_after_backend_recovery(loop,observed_policy=policy()), "delivery not reconciled")
                    rounds.append({"round":round_index,"ack_interrupted":True,"active":asdict(policy()),"recovery_s":time.perf_counter()-recovery_start,"round_wall_s":time.perf_counter()-start})
                    break
                if result is None:
                    await asyncio.sleep(.01)
            if result is not None:
                require(round_index!=1 or not result.decision.accepted, "forced reject promoted")
                rounds.append({"round":round_index,"result":asdict(result),"wall_s":time.perf_counter()-start})
            require(equal(hooks._states[hooks.active_policy()[0]].training_state,trainer.snapshot_training_state()), "active state inconsistency")
        require(optimizer_calls==4 and len(records)==4, "not four actual optimizer calls")
        require(adapter.pending() is None and len(loop.replay)==0, "delivery remained unresolved")
    finally:
        stop.set()
        await asyncio.gather(*workers)
        write(args.output/"updates.json",records)
        write(args.output/"evaluations.json",evaluations)
        write(args.output/"rounds.json",rounds)
        write(args.output/"worker-policy-acknowledgments.json",acknowledgments)
    store.backup(args.output/"snapshot.db")
    audit = audit_trajectory_store(args.output/"snapshot.db")
    wall = time.perf_counter()-started
    summary = {"pass":True,"seed":args.seed,"scope":"Real pretrained model updates on synthetic terminal rewards; independent checkpoint decoder; no semantic task success or capability claim. Lost ingestion ACK and exact state reload are same-process logical restart, not learner SIGKILL.",
        "optimizer_calls":optimizer_calls,"rounds":len(rounds),"trajectory_states":store.counts(),"audit":audit,"wall_s":wall,
        "parent_cpu_core_percent":100*(time.process_time()-cpu)/wall,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"torch":torch.__version__,"transformers":transformers.__version__,
        "revision":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),"source_files":{str(p.relative_to(ROOT)):sha(p) for p in source_files}}
    require(summary["peak_rss_bytes"] <= 8*1024**3, "RSS budget exceeded")
    write(args.output/"summary.json",summary)
    write(args.output/"SHA256.json",{str(p.relative_to(args.output)):sha(p) for p in args.output.rglob('*') if p.is_file() and p.name!='trajectories.db' and not p.name.startswith('trajectories.db-')})
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--rvl-source',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args = parser.parse_args()
    try:
        asyncio.run(asyncio.wait_for(run(args), timeout=1200))
    except BaseException as exc:
        if getattr(args, "output_created", False):
            write(args.output/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
