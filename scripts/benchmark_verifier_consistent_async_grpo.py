"""Matched TinyStories GRPO experiment for verifier drift and admission.

This is a controlled systems experiment, not a coding-capability evaluation.
The pretrained model, real sampled tokens, and RVL GRPO optimizer are real;
the small verifier task is synthetic and its policy score is not a capability
metric. See the frozen protocol in benchmarks/verifier_consistent_async_grpo_v1.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks/verifier_consistent_async_grpo_v5/protocol.json"
PROMPT = "Choose one label for a story: a blue bird landed on the"
GROUP_SIZE = 8


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def prepare_pinned_imports(rvl_source: Path) -> None:
    """Avoid editable-install finders that can silently route VARE elsewhere."""
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
                    f"preloaded {name} module came from outside the pinned source")
    sys.path.insert(0, str(rvl_source / "src"))
    sys.path.insert(0, str(ROOT / "src"))


def verify_import_origins(rvl_source: Path) -> None:
    from importlib import import_module
    expected = {
        "vare": ROOT / "src/vare/__init__.py",
        "vare.attestation": ROOT / "src/vare/attestation.py",
        "vare.config": ROOT / "src/vare/config.py",
        "vare.engine": ROOT / "src/vare/engine.py",
        "vare.evidence": ROOT / "src/vare/evidence.py",
        "vare.promotion": ROOT / "src/vare/promotion.py",
        "vare.promotion_journal": ROOT / "src/vare/promotion_journal.py",
        "vare.trajectory_admission": ROOT / "src/vare/trajectory_admission.py",
        "vare.trajectory_learner": ROOT / "src/vare/trajectory_learner.py",
        "vare.trajectory_store": ROOT / "src/vare/trajectory_store.py",
        "vare.types": ROOT / "src/vare/types.py",
        "vare.telemetry": ROOT / "src/vare/telemetry.py",
        "vare.integrations.rvl_grpo": ROOT / "src/vare/integrations/rvl_grpo.py",
        "rvl_systems": rvl_source / "src/rvl_systems/__init__.py",
        "rvl_systems.hf_backend": rvl_source / "src/rvl_systems/hf_backend.py",
        "rvl_systems.hf_trainer": rvl_source / "src/rvl_systems/hf_trainer.py",
        "rvl_systems.grpo": rvl_source / "src/rvl_systems/grpo.py",
        "rvl_systems.types": rvl_source / "src/rvl_systems/types.py",
    }
    for name, path in expected.items():
        module = import_module(name)
        require(Path(module.__file__).resolve() == path.resolve(),
                f"import origin mismatch for {name}: {module.__file__}")


def swap_used_bytes() -> int | None:
    if sys.platform != "darwin":
        return None
    raw = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
    match = re.search(r"used\s*=\s*([0-9.]+)([MG])", raw)
    require(match is not None, f"cannot parse macOS swap telemetry: {raw}")
    scale = 1024**2 if match.group(2) == "M" else 1024**3
    return int(float(match.group(1)) * scale)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       int((len(ordered) - 1) * probability + 0.5)))
    return float(ordered[index])


def versioned_score(version: int, response: str) -> float:
    if version == 0:
        return float(bool(re.search(r"[.!?]", response)))
    if version == 1:
        return float(bool(re.search(
            r"(?<![A-Za-z0-9_])ground(?![A-Za-z0-9_])",
            response, flags=re.IGNORECASE)))
    raise ValueError("unknown verifier version")


def reward_disagreements(group: dict) -> dict[str, int]:
    false_accepts = false_rejects = 0
    for member in group["members"]:
        authoritative = versioned_score(1, member["attempt"].output)
        if member["score"] == 1 and authoritative == 0:
            false_accepts += 1
        if member["score"] == 0 and authoritative == 1:
            false_rejects += 1
    return {"false_accepts": false_accepts, "false_rejects": false_rejects,
            "total_disagreements": false_accepts + false_rejects}


class VersionState:
    def __init__(self, version: int):
        self.version = version

    @property
    def active_version(self) -> int:
        return self.version


async def collect_group(hooks, task, *, start_step: int, drift: bool,
                        state: VersionState) -> dict:
    """One producer and one verifier; a bounded queue overlaps the drift arm."""
    rows: list[dict | None] = [None] * GROUP_SIZE
    queue_wait = [0.0] * GROUP_SIZE
    started = time.perf_counter()
    if not drift:
        for member in range(GROUP_SIZE):
            attempt = await hooks.rollout(task, "policy-0", 0, start_step + member)
            rows[member] = {
                "attempt": attempt,
                "verifier_version": state.version,
                "score": versioned_score(state.version, attempt.output),
                "queue_wait_s": 0.0,
            }
    else:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        generated = {"count": 0}

        async def producer() -> None:
            for member in range(GROUP_SIZE):
                attempt = await hooks.rollout(task, "policy-0", 0, start_step + member)
                generated["count"] = member + 1
                await queue.put((member, attempt, time.perf_counter()))
                await asyncio.sleep(0)
            await queue.put(None)

        transition = {}

        async def verifier() -> None:
            seen = 0
            while True:
                item = await queue.get()
                if item is None:
                    return
                member, attempt, enqueued = item
                queue_wait[member] = time.perf_counter() - enqueued
                version = state.version
                rows[member] = {
                    "attempt": attempt,
                    "verifier_version": version,
                    "score": versioned_score(version, attempt.output),
                    "queue_wait_s": queue_wait[member],
                }
                seen += 1
                if seen == GROUP_SIZE // 2:
                    state.version = 1
                    transition.update({"verified_members": seen,
                                       "generated_members": generated["count"],
                                       "rollouts_still_active": generated["count"] < GROUP_SIZE})
                await asyncio.sleep(0)

        await asyncio.gather(producer(), verifier())
        require(transition.get("rollouts_still_active") is True,
                "verifier transition did not overlap active rollout collection")
    completed = [x for x in rows if x is not None]
    require(len(completed) == GROUP_SIZE, "incomplete rollout group")
    versions = [int(x["verifier_version"]) for x in completed]
    scores = [float(x["score"]) for x in completed]
    return {"members": completed, "versions": versions, "scores": scores,
            "wall_s": time.perf_counter() - started,
            "queue_wait_s": queue_wait,
            "transition": transition if drift else None}


def make_experiences(group: dict, group_id: str, *, verifier_shas: dict,
                     trusted: bool = False):
    from vare.types import Experience, Verification

    result = []
    for member, row in enumerate(group["members"]):
        source = row["attempt"]
        metadata = {
            **source.metadata,
            "vare_rollout_group": group_id,
            "vare_rollout_group_size": GROUP_SIZE,
            "rvl_generation": copy.deepcopy(source.metadata["rvl_generation"]),
        }
        source.metadata = metadata
        value = Verification(
            score=float(row["score"]), passed=bool(row["score"]), confidence=1.0,
            verifier_version=int(row["verifier_version"]), verifier_name="landing",
            trusted=trusted,
            metadata={"verifier_config_sha256": verifier_shas[int(row["verifier_version"])]},
        )
        result.append(Experience(source, value, 0, 1 - value.verifier_version, 0.0))
    return result


def install_naive_rvl_reward_path(hooks) -> None:
    """Match RVL's default prompt grouping while intentionally ignoring verifier lineage."""
    from rvl_systems.grpo import compute_group_advantages

    def prompt_only_advantages(experiences, verified, *, eps, clip):
        require(len(experiences) == len(verified),
                "naive reward path changed sample cardinality")
        prompt_ids = {sample.generation.prompt_id for sample in verified}
        require(len(prompt_ids) == 1,
                "naive fixture must be one RVL prompt-relative group")
        return [record.advantage for record in compute_group_advantages(
            list(verified), eps=eps, clip=clip)]

    # This is the deliberate negative control: vanilla RVL normalizes reward
    # by prompt ID and does not bind that reward to a verifier version. VARE's
    # normal provenance-aware group guard remains unchanged in the other arms.
    hooks._group_relative_advantages = prompt_only_advantages


def max_tensor_error(actual, expected) -> float:
    import torch
    errors = []
    def walk(a, b):
        if isinstance(a, torch.Tensor):
            require(isinstance(b, torch.Tensor) and a.shape == b.shape, "tensor reference shape mismatch")
            errors.append(float((a.detach().cpu() - b.detach().cpu()).abs().max()) if a.numel() else 0.0)
        elif isinstance(a, dict):
            require(isinstance(b, dict) and a.keys() == b.keys(), "state reference keys mismatch")
            for key in a:
                walk(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            require(isinstance(b, type(a)) and len(a) == len(b), "state reference sequence mismatch")
            for x, y in zip(a, b):
                walk(x, y)
        else:
            require(a == b, "scalar state reference mismatch")
    walk(actual, expected)
    return max(errors, default=0.0)


async def build_runtime(args, out: Path, seed: int, *, naive: bool,
                        eval_version: int):
    prepare_pinned_imports(args.rvl_source)
    import torch
    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from rvl_systems.types import Generation, VerifiedGeneration
    from vare.config import LagConfig
    from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
    verify_import_origins(args.rvl_source)

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    state = VersionState(1 if not naive else 0)
    backend = HFLocalBackend(str(args.model_dir), max_new_tokens=3,
                             device="cpu", precision="fp32")
    model = backend.model
    from validate_grpo_e2e_cpu import tensor_hash
    initial_model_tensor_sha256 = tensor_hash(model.state_dict())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("transformer.wpe.")
                                 or name.startswith("transformer.ln_f."))
    train_config = HFTTrainerConfig(learning_rate=1e-4)
    trainer = HFCausalLMGRPOTrainer(model, config=train_config)
    eval_tasks = [
        __import__("vare.types", fromlist=["Task"]).Task(
            f"heldout-{seed}-{i}", f"Held-out landing rule {i}: a blue bird landed on the", "landing")
        for i in range(4)
    ]
    hooks = RVLGRPOHooks(
        backend=backend, trainer=trainer, eval_tasks=eval_tasks,
        score_fn=lambda _task, text: versioned_score(eval_version, text),
        config=RVLGRPOConfig(seed=seed), current_verifier_version=lambda: state.version,
        freshness_lag_config=LagConfig(max_policy_lag=0,
                                       max_verifier_lag=1 if naive else 0),
        generation_factory=Generation, verified_generation_factory=VerifiedGeneration,
    )
    if naive:
        install_naive_rvl_reward_path(hooks)
    checkpoint_paths: dict[str, Path] = {}
    checkpoint_io: list[dict] = []
    def save(policy_id: str):
        box = hooks._states[policy_id]
        state_dict = box.training_state if hasattr(box, "training_state") else box
        path = out / "checkpoints" / f"{policy_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        io_started = time.perf_counter()
        torch.save(state_dict, path)
        write_s = time.perf_counter() - io_started
        loaded = torch.load(path, weights_only=True)
        require(max_tensor_error(state_dict, loaded) == 0.0, "checkpoint round-trip differs")
        checkpoint_paths[policy_id] = path
        checkpoint_io.append({"policy_id": policy_id, "write_s": write_s,
                              "bytes": path.stat().st_size, "sha256": sha(path)})
        return state_dict
    save("policy-0")
    original_train = hooks.train_candidate_with_freshness
    updates: list[dict] = []
    optimizer_calls = {"count": 0}
    original_step = trainer.optimizer.step
    captured: dict = {}
    def counted_step(*a, **kw):
        optimizer_calls["count"] += 1
        captured["gradients"] = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters() if param.grad is not None
        }
        return original_step(*a, **kw)
    trainer.optimizer.step = counted_step
    reference_model = copy.deepcopy(model)
    evaluation_model = copy.deepcopy(model)
    async def checked_train(incumbent_id, batch, assessor):
        before = trainer.snapshot_training_state()
        reference_batch = copy.deepcopy(list(batch))
        for index, experience in enumerate(reference_batch):
            experience.attempt.created_step = index
        from validate_grpo_e2e_cpu import reference_update
        oracle = reference_update(reference_model, before, reference_batch, train_config)
        candidate_id = await original_train(incumbent_id, batch, assessor)
        box = hooks._states[candidate_id]
        candidate_state = box.training_state if hasattr(box, "training_state") else box
        require(max_tensor_error(before, trainer.snapshot_training_state()) == 0.0,
                "candidate training failed to restore incumbent state")
        errors = {
            "gradient_max_abs": max_tensor_error(captured["gradients"], oracle["grads"]),
            "model_max_abs": max_tensor_error(candidate_state["model"], oracle["model"]),
            "optimizer_max_abs": max_tensor_error(candidate_state["optimizer"], oracle["optimizer"]),
        }
        require(max(errors.values()) < 2e-6, "actual GRPO update differs from independent replay")
        require(oracle["max_behavior_alignment_error"] < 1e-4,
                "policy behavior log-probability alignment failed")
        save(candidate_id)
        updates.append({"candidate_id": candidate_id, "batch_size": len(batch),
                        "optimizer_calls": optimizer_calls["count"], "errors": errors,
                        "loss": oracle["loss"], "advantages": oracle["advantages"],
                        "behavior_alignment_error": oracle["max_behavior_alignment_error"]})
        return candidate_id
    hooks.train_candidate_with_freshness = checked_train
    return locals()


async def independent_evaluate(runtime: dict, policy_id: str, verifier_version: int):
    """Greedy-decode a saved checkpoint with a separate model object."""
    import torch
    from vare.types import EvaluationReport

    started = time.perf_counter()
    hooks, tokenizer = runtime["hooks"], runtime["backend"].tokenizer
    box = hooks._states[policy_id]
    state = box.training_state if hasattr(box, "training_state") else box
    evaluator = runtime["evaluation_model"]
    evaluator.load_state_dict(state["model"])
    evaluator.eval()
    outputs, scores = {}, {}
    with torch.no_grad():
        for task in hooks.eval_tasks:
            prompt_ids = tokenizer.encode(task.prompt)
            generated = []
            for _ in range(3):
                logits = evaluator(torch.tensor([prompt_ids + generated])).logits[0, -1]
                token = int(logits.argmax())
                generated.append(token)
                if token == tokenizer.eos_token_id:
                    break
            text = tokenizer.decode(generated, skip_special_tokens=True)
            outputs[task.id] = {"token_ids": generated, "text": text}
            scores[task.id] = versioned_score(verifier_version, text)
    primary = fmean(scores.values())
    report = EvaluationReport(
        policy_id=policy_id, primary=primary, slices={"landing": primary},
        cost=time.perf_counter() - started, verifier_disagreement=0.0,
        n=len(scores), metadata={"per_task_scores": scores,
                                 "verifier_version": verifier_version},
    )
    return report, {"outputs": outputs, "scores": scores,
                    "wall_s": time.perf_counter() - started}


def persist_group(store, group_id: str, rows: list[dict], *, policy,
                  reward_root, verifier_shas: dict, task):
    from vare.trajectory_store import AgentStep
    from vare.types import Verification
    ids = [f"{group_id}-{i:02d}" for i in range(GROUP_SIZE)]
    for member, ident in enumerate(ids):
        store.submit(ident, task, expected_steps=1, group_id=group_id,
                     member=member, group_size=GROUP_SIZE)
    attempts_by_id = dict(zip(ids, rows, strict=True))
    seen = set()
    while len(seen) < GROUP_SIZE:
        lease = store.claim("execute", "grpo-rollout")
        require(lease is not None, "execute queue ended before complete group")
        attempt = attempts_by_id[lease.trajectory_id]["attempt"]
        step = AgentStep(
            index=0, event_id="terminal-answer", policy=policy,
            action={"kind": "answer"}, observation={}, state={},
            generation=copy.deepcopy(attempt.metadata["rvl_generation"]),
        )
        store.append_step(lease, step)
        store.finish(lease)
        seen.add(lease.trajectory_id)
    verified = set()
    while len(verified) < GROUP_SIZE:
        lease = store.claim("verify", "grpo-verifier")
        require(lease is not None, "verify queue ended before complete group")
        row = store.trajectory(lease.trajectory_id)
        generated = attempts_by_id[lease.trajectory_id]
        ver = int(generated["verifier_version"])
        value = Verification(
            score=float(generated["score"]), passed=bool(generated["score"]),
            confidence=1.0, verifier_version=ver, verifier_name="landing",
            trusted=True,
            metadata={"trajectory_sha256": row["head"],
                      "verifier_config_sha256": verifier_shas[ver]},
        )
        store.record_verification(
            lease, reward_root.attest(row["id"], row["head"], value)
        )
        verified.add(lease.trajectory_id)


def serializable_group(group: dict) -> dict:
    return {"versions": group["versions"], "scores": group["scores"],
            "wall_s": group["wall_s"], "transition": group["transition"],
            "queue_wait_s": group["queue_wait_s"],
            "members": [
                {"attempt": asdict(row["attempt"]),
                 "verifier_version": row["verifier_version"], "score": row["score"],
                 "queue_wait_s": row["queue_wait_s"]}
                for row in group["members"]
            ]}


async def run(args) -> None:
    require(sys.flags.no_site == 1,
            "launch the measured runner with python -S to disable site startup hooks")
    require(not args.output.exists(), "output directory already exists")
    args.output.mkdir(parents=True)
    protocol = json.loads(PROTOCOL_PATH.read_text())
    require(sha(Path(__file__)) == protocol["runner_sha256"], "runner hash differs from frozen protocol")
    require(sha(ROOT / "scripts/audit_verifier_consistent_async_grpo.py")
            == protocol["auditor_sha256"], "independent auditor hash differs from frozen protocol")
    require(sha(ROOT / "scripts/benchmark_verifier_consistent_grpo_sigkill.py")
            == protocol["crash_runner_sha256"], "crash runner hash differs from frozen protocol")
    require(subprocess.check_output(["git", "-C", str(ROOT), "merge-base", "HEAD",
                                     protocol["vare_base_commit"]], text=True).strip()
            == protocol["vare_base_commit"],
            "VARE base commit differs from frozen protocol")
    for relative, digest in protocol["source_sha256"].items():
        require(sha(ROOT / relative) == digest,
                f"VARE source differs from frozen protocol: {relative}")
    # Do this before importing any VARE module. The host Python environment can
    # contain editable-install finders that shadow this exact worktree.
    prepare_pinned_imports(args.rvl_source)
    verify_import_origins(args.rvl_source)
    budgets = protocol["budgets"]
    require(args.seed in budgets["seeds"], "undeclared seed")
    for filename, digest in budgets["model_files"].items():
        require(sha(args.model_dir / filename) == digest, f"model checksum mismatch: {filename}")
    require(subprocess.check_output(["git", "-C", str(args.rvl_source), "rev-parse", "HEAD"], text=True).strip()
            == budgets["rvl_revision"], "RVL revision mismatch")
    require(subprocess.check_output(["git", "-C", str(args.rvl_source), "status", "--porcelain"], text=True).strip()
            == "M src/rvl_systems/hf_backend.py", "unexpected RVL source modifications")
    require(sha(args.rvl_source / "src/rvl_systems/hf_backend.py")
            == budgets["rvl_patched_backend_sha256"], "RVL backend patch mismatch")
    require(sha(ROOT / "patches/rvl-contiguous-transition-logprobs-v1.patch")
            == budgets["rvl_patch_sha256"], "RVL patch provenance mismatch")
    source_files = [Path(__file__), ROOT / "scripts/audit_verifier_consistent_async_grpo.py",
                    ROOT / "scripts/benchmark_verifier_consistent_grpo_sigkill.py",
                    ROOT / "scripts/validate_grpo_e2e_cpu.py",
                    ROOT / "src/vare/engine.py", ROOT / "src/vare/promotion.py",
                    ROOT / "src/vare/promotion_journal.py",
                    ROOT / "src/vare/config.py", ROOT / "src/vare/attestation.py",
                    ROOT / "src/vare/evidence.py", ROOT / "src/vare/types.py",
                    ROOT / "src/vare/telemetry.py",
                    ROOT / "src/vare/trajectory_learner.py",
                    ROOT / "src/vare/trajectory_store.py",
                    ROOT / "src/vare/trajectory_admission.py",
                    ROOT / "src/vare/integrations/rvl_grpo.py",
                    ROOT / "tests/test_checkpoint_bound_promotion.py",
                    args.rvl_source / "src/rvl_systems/hf_backend.py",
                    args.rvl_source / "src/rvl_systems/hf_trainer.py"]
    for path in source_files:
        if path.is_file():
            target = args.output / "source" / (path.name + "." + sha(path)[:12])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (args.output / "protocol.json").write_bytes(PROTOCOL_PATH.read_bytes())
    write(args.output / "progress.json", {"status": "running", "seed": args.seed,
                                            "completed_arms": [], "active_arm": "sync"})
    from vare.attestation import EvidenceTrustRoot
    from vare.config import EngineConfig, LagConfig, PromotionConfig, ReplayConfig
    from vare.engine import CapabilityLoop
    from vare.evidence import sha256_file, sha256_json
    from vare.promotion import PromotionGate
    from vare.promotion_journal import PromotionJournal
    from vare.trajectory_admission import TrajectoryRewardRoot
    from vare.trajectory_learner import TrajectoryLearnerAdapter
    from vare.trajectory_store import PolicyCheckpoint, TrajectoryStore, audit_trajectory_store
    from vare.telemetry import EventLog
    from vare.types import Task
    import torch
    import transformers
    verify_import_origins(args.rvl_source)

    started = time.perf_counter()
    result = {"seed": args.seed, "arms": {}, "environment": {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": sys.version, "device": "cpu", "threads": 1,
        "model_path": str(args.model_dir),
        "peak_rss_bytes": None,
    }}
    swap_before = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip() if sys.platform == "darwin" else "unavailable"
    swap_before_bytes = swap_used_bytes()
    protocol_sha = sha(PROTOCOL_PATH)
    v0_sha = sha256_json({"name": "landing", "version": 0,
                          "rule": "punctuation completeness"})
    v1_sha = sha256_json({"name": "landing", "version": 1,
                          "rule": "canonical ground semantics"})
    verifier_shas = {0: v0_sha, 1: v1_sha}

    for arm in ("sync", "naive_async", "vare_async"):
        write(args.output / "progress.json", {
            "status": "running", "seed": args.seed,
            "completed_arms": [*result["arms"].keys()], "active_arm": arm,
            "elapsed_s": time.perf_counter() - started,
        })
        arm_started = time.perf_counter()
        arm_cpu_started = time.process_time()
        arm_out = args.output / arm
        arm_out.mkdir()
        naive = arm == "naive_async"
        # All arms start from the same pretrained checkpoint and optimizer
        # initialization. The narrow in-memory run is sequential on CPU.
        runtime = await build_runtime(args, arm_out, args.seed,
                                      naive=naive, eval_version=1 if arm != "naive_async" else 0)
        hooks, trainer, checkpoints = runtime["hooks"], runtime["trainer"], runtime["checkpoint_paths"]
        state = runtime["state"]
        groups = {}
        if arm == "sync":
            state.version = 1
            groups["audit"] = await collect_group(hooks, Task(
                f"sync-{args.seed}-audit", PROMPT, "landing",
                {"learning_rule": "terminal_action_on_policy_grpo_v1"}),
                start_step=0, drift=False, state=state)
            groups["fresh"] = await collect_group(hooks, Task(
                f"sync-{args.seed}-fresh", PROMPT, "landing",
                {"learning_rule": "terminal_action_on_policy_grpo_v1"}),
                start_step=8, drift=False, state=state)
        else:
            state.version = 0
            groups["stale"] = await collect_group(hooks, Task(
                f"{arm}-{args.seed}-stale", PROMPT, "landing",
                {"learning_rule": "terminal_action_on_policy_grpo_v1"}),
                start_step=0, drift=True, state=state)
            groups["fresh"] = await collect_group(hooks, Task(
                f"{arm}-{args.seed}-fresh", PROMPT, "landing",
                {"learning_rule": "terminal_action_on_policy_grpo_v1"}),
                start_step=8, drift=False, state=state)
        all_records = {key: serializable_group(value) for key, value in groups.items()}
        write(arm_out / "rollouts.json", all_records)
        require(all(len(set(g["scores"])) > 1 for g in groups.values()),
                "a prespecified verifier group is degenerate; stop without resampling")

        if arm == "naive_async":
            batch = make_experiences(
                groups["stale"], "naive-stale", verifier_shas=verifier_shas)
            candidate_id = await hooks.train_candidate_with_freshness(
                "policy-0", batch, hooks._training_freshness_assessor)
            # The naive control deliberately ignores verifier provenance at
            # policy selection and promotion. It trains once on the mixed group.
            eval_key = os.urandom(32)
            eval_set = [asdict(task) for task in hooks.eval_tasks]
            root = EvidenceTrustRoot(
                eval_key, protocol_sha, v0_sha, f"naive-run-{args.seed}",
                sha256_json(eval_set), f"naive-nonce-{args.seed}-long", True)
            eval_records = []
            async def signed_eval(policy_id, _runtime=runtime, _root=root,
                                  _files=checkpoints, _records=eval_records):
                report, detail = await independent_evaluate(_runtime, policy_id, 0)
                signed = _root.attest(report, checkpoint_sha256=sha256_file(_files[policy_id]))
                _records.append({"report": asdict(signed), **detail})
                return signed
            old_eval, new_eval = await signed_eval("policy-0"), await signed_eval(candidate_id)
            write(arm_out / "evaluations.json", eval_records)
            cfg = PromotionConfig(min_primary_gain=-1.0, max_slice_regression=1.0,
                                  max_verifier_disagreement=1.0, min_eval_examples=4,
                                  max_relative_cost_increase=1e6)
            naive_decision = PromotionGate(cfg, trust_root=root).decide(old_eval, new_eval)
            strict_decision = PromotionGate(cfg, trust_root=root).decide(
                old_eval, new_eval, expected_verifier_version=1)
            require(naive_decision.accepted, "naive promotion control did not accept its prespecified stale report")
            require(not strict_decision.accepted
                    and strict_decision.reasons == ("evaluation_verifier_version_mismatch",),
                    "strict verifier-version gate did not reject stale signed evaluation")
            journal = PromotionJournal(arm_out / "promotion.sqlite",
                initial_policy_id="policy-0", initial_checkpoint_sha256=sha256_file(checkpoints["policy-0"]))
            evaluation_sha = sha256_json({"incumbent": asdict(old_eval), "candidate": asdict(new_eval)})
            journal.prepare(tx_id=f"naive-promotion-nonce-{args.seed}",
                incumbent_id="policy-0", incumbent_sha256=sha256_file(checkpoints["policy-0"]),
                candidate_id=candidate_id, candidate_sha256=sha256_file(checkpoints[candidate_id]),
                evaluation_sha256=evaluation_sha)
            await hooks.promote(candidate_id)
            journal.commit(f"naive-promotion-nonce-{args.seed}")
            result["arms"][arm] = {
                "rollouts": GROUP_SIZE * 2, "verifier_calls": GROUP_SIZE * 2,
                "training_group": "stale", "training_versions": groups["stale"]["versions"],
                "training_path": protocol["naive_control"]["training_path"],
                "stale_reward_disagreements": reward_disagreements(groups["stale"]),
                "optimizer_calls": runtime["optimizer_calls"]["count"],
                "naive_promotion": asdict(naive_decision),
                "strict_stale_promotion": asdict(strict_decision),
                "journal": journal.status(), "initial_state_sha256": sha256_file(checkpoints["policy-0"]),
                "candidate_state_sha256": sha256_file(checkpoints[candidate_id]),
                "wall_s": sum(g["wall_s"] for g in groups.values()),
                "updates": runtime["updates"],
                "unused_group": ["fresh"],
            }
        else:
            arm_run_id = f"varegrpo-v5-{arm}-{args.seed}"
            require(len(arm_run_id) >= 16, "operator run ID is shorter than VARE trust-root minimum")
            store = TrajectoryStore.create(arm_out / "trajectories.sqlite",
                run_id=arm_run_id, max_inflight=32)
            def current_policy(_hooks=hooks, _files=checkpoints):
                active_id, active_version = _hooks.active_policy()
                return PolicyCheckpoint(active_id, active_version,
                                        sha256_file(_files[active_id]))
            initial_policy = current_policy()
            store.register_policy(initial_policy)
            store.register_verifier("landing", 0, v0_sha)
            store.register_verifier("landing", 1, v1_sha)
            reward_key = os.urandom(32)
            reward_root = TrajectoryRewardRoot(reward_key, arm_run_id, protocol_sha)
            groups_to_store = (["fresh"] if arm == "sync" else ["stale", "fresh"])
            for name in groups_to_store:
                group_id = "g-01-fresh" if name == "fresh" else "g-00-stale"
                task = groups[name]["members"][0]["attempt"].task
                persist_group(store, group_id, groups[name]["members"],
                              policy=initial_policy, reward_root=reward_root,
                              verifier_shas=verifier_shas, task=task)
            eval_key = os.urandom(32)
            eval_tasks = hooks.eval_tasks
            eval_root = EvidenceTrustRoot(
                eval_key, protocol_sha, v1_sha, arm_run_id,
                sha256_json([asdict(task) for task in eval_tasks]),
                f"promotion-nonce-{arm}-{args.seed}-v1", True)
            hooks.training_evidence_authorizer = None
            journal = PromotionJournal(arm_out / "promotion.sqlite",
                initial_policy_id="policy-0", initial_checkpoint_sha256=initial_policy.checkpoint_sha256)
            adapter = TrajectoryLearnerAdapter(
                store, trust_root=reward_root, current_policy=current_policy,
                verifier_identity=("landing", 1, v1_sha),
                reference_families=("landing",), max_groups=1, max_members=GROUP_SIZE)
            hooks.training_evidence_authorizer = adapter.authorizes
            hooks.require_authenticated_training = True
            loop = CapabilityLoop(
                hooks=hooks, verifier=state, require_authenticated_training=True,
                training_evidence_authorizer=adapter.authorizes,
                require_current_evaluation_verifier=True,
                evidence_trust_root=eval_root, promotion_journal=journal,
                checkpoint_hasher=lambda pid: sha256_file(checkpoints[pid]),
                config=EngineConfig(
                    samples_per_task=GROUP_SIZE, replay_batch_size=GROUP_SIZE,
                    replay=ReplayConfig(capacity=GROUP_SIZE),
                    lag=LagConfig(max_policy_lag=0, max_verifier_lag=0),
                    promotion=PromotionConfig(min_primary_gain=-1.0,
                        max_slice_regression=1.0, max_verifier_disagreement=1.0,
                        min_eval_examples=4, max_relative_cost_increase=1e6)),
                event_log=EventLog(arm_out / "events.jsonl"), seed=args.seed)
            active_signer = {"root": eval_root}
            evaluations = []
            async def versioned_eval(policy_id, _runtime=runtime, _signer=active_signer,
                                     _files=checkpoints, _records=evaluations):
                report, detail = await independent_evaluate(_runtime, policy_id, 1)
                signed = _signer["root"].attest(
                    report, checkpoint_sha256=sha256_file(_files[policy_id]))
                _records.append({"report": asdict(signed), **detail})
                return signed
            hooks.evaluate = versioned_eval
            # checked_train was installed by build_runtime and performs a full
            # independent numerical replay of the actual optimizer update.
            if arm == "vare_async":
                first = await adapter.advance(loop, round_index=0)
                require(first is None, "mixed stale group unexpectedly trained")
                require(store.counts().get("quarantined", 0) == GROUP_SIZE,
                        "stale VARE group was not explicitly quarantined")
            trained = await adapter.advance(loop, round_index=1 if arm == "vare_async" else 0)
            require(trained is not None, "fresh verifier group was not trained")
            write(arm_out / "evaluations.json", evaluations)
            store.backup(arm_out / "snapshot.sqlite")
            audit = audit_trajectory_store(arm_out / "snapshot.sqlite")
            result["arms"][arm] = {
                "rollouts": GROUP_SIZE * 2, "verifier_calls": GROUP_SIZE * 2,
                "stored_groups": groups_to_store,
                "stale_group_versions": groups["stale"]["versions"] if arm == "vare_async" else None,
                "admission_decisions": [
                    {"group": row["group_id"], "state": row["state"], "reason": row["reason"]}
                    for row in (store.group("g-00-stale") if arm == "vare_async" else [])],
                "input_group": "fresh", "optimizer_calls": runtime["optimizer_calls"]["count"],
                "journal": journal.status(), "trajectory_counts": store.counts(),
                "audit": audit, "update_records": runtime["updates"],
                "initial_state_sha256": sha256_file(checkpoints["policy-0"]),
                "active_checkpoint_sha256": journal.status()["active_checkpoint_sha256"],
                "wall_s": sum(g["wall_s"] for g in groups.values()),
                "updates": runtime["updates"],
                "unused_group": (["audit"] if arm == "sync" else []),
                "correctly_scored_fresh_group": sum(groups["fresh"]["scores"]),
            }
            if arm == "vare_async":
                result["arms"][arm]["stale_reward_disagreements"] = reward_disagreements(
                    groups["stale"])
        result["arms"][arm]["model_hashes"] = {
            key: sha(path) for key, path in checkpoints.items()}
        result["arms"][arm]["total_wall_s"] = time.perf_counter() - arm_started
        result["arms"][arm]["initial_model_tensor_sha256"] = runtime["initial_model_tensor_sha256"]
        result["arms"][arm]["checkpoint_io"] = runtime["checkpoint_io"]
        result["arms"][arm]["rollouts_per_hour"] = (
            result["arms"][arm]["rollouts"] * 3600 /
            max(result["arms"][arm]["total_wall_s"], 1e-9))
        result["arms"][arm]["verifier_calls_per_hour"] = (
            result["arms"][arm]["verifier_calls"] * 3600 /
            max(result["arms"][arm]["total_wall_s"], 1e-9))
        result["arms"][arm]["optimizer_updates_per_hour"] = (
            result["arms"][arm]["optimizer_calls"] * 3600 /
            max(result["arms"][arm]["total_wall_s"], 1e-9))
        result["arms"][arm]["cpu_core_percent"] = (
            100 * (time.process_time() - arm_cpu_started) /
            max(result["arms"][arm]["total_wall_s"], 1e-9))
        result["arms"][arm]["generated_tokens"] = sum(
            int(member["attempt"].metadata["rvl_generation"]["token_count"])
            for group in groups.values() for member in group["members"])
        result["arms"][arm]["committed_updates"] = int(
            result["arms"][arm]["journal"]["active_policy_id"] != "policy-0")
        disagreements = result["arms"][arm].get(
            "stale_reward_disagreements", {"total_disagreements": 0})["total_disagreements"]
        result["arms"][arm]["valid_committed_updates"] = int(
            result["arms"][arm]["committed_updates"] == 1
            and (arm != "naive_async" or disagreements == 0))
        result["arms"][arm]["invalid_committed_updates"] = int(
            result["arms"][arm]["committed_updates"] == 1 and arm == "naive_async"
            and disagreements > 0)
        result["arms"][arm]["extra_verifier_calls_for_safety"] = 0
        result["arms"][arm]["quarantined_rollout_fraction"] = (
            result["arms"][arm].get("trajectory_counts", {}).get("quarantined", 0)
            / result["arms"][arm]["rollouts"])
        result["arms"][arm]["useful_verified_groups_per_hour"] = (
            3600 / max(result["arms"][arm]["total_wall_s"], 1e-9)
            if arm != "naive_async" else 0.0)
        sample_latencies = [float(member["attempt"].latency_ms) / 1000
                            for group in groups.values()
                            for member in group["members"]]
        queue_waits = [float(member["queue_wait_s"])
                       for group in groups.values()
                       for member in group["members"]]
        result["arms"][arm]["latency_s"] = {
            "sample_p50": percentile(sample_latencies, 0.5),
            "sample_p95": percentile(sample_latencies, 0.95),
            "queue_inclusive_group_p50": percentile(
                [float(group["wall_s"]) for group in groups.values()], 0.5),
            "queue_inclusive_group_p95": percentile(
                [float(group["wall_s"]) for group in groups.values()], 0.95),
            "queue_wait_p50": percentile(queue_waits, 0.5),
            "queue_wait_p95": percentile(queue_waits, 0.95),
        }
        write(args.output / "progress.json", {
            "status": "running", "seed": args.seed,
            "completed_arms": [*result["arms"].keys()],
            "active_arm": None,
            "elapsed_s": time.perf_counter() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if sys.platform == "darwin" else 1024),
        })
        require(time.perf_counter() - started <= protocol["budgets"]["max_wall_seconds_per_seed"],
                "seed exceeded the frozen 1200-second wall budget")
        current_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024)
        require(current_peak_rss <= protocol["budgets"]["max_rss_bytes_per_seed"],
                "seed exceeded the frozen 8 GiB RSS budget")
        current_swap = swap_used_bytes()
        swap_growth = max(0, current_swap - swap_before_bytes) if (
            current_swap is not None and swap_before_bytes is not None) else 0
        require(swap_growth <= protocol["budgets"]["max_swap_growth_bytes"],
                "seed exceeded the frozen swap-growth budget")
        # Do not keep multiple model instances resident on the 32 GB host.
        del runtime, hooks, trainer
        import gc
        gc.collect()
    initial_hashes = {arm["initial_model_tensor_sha256"]
                      for arm in result["arms"].values()}
    require(len(initial_hashes) == 1, "arms did not start from identical model tensors")
    result["environment"]["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    result["environment"]["swap_usage_before"] = swap_before
    result["environment"]["swap_usage_after"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip() if sys.platform == "darwin" else "unavailable"
    swap_after_bytes = swap_used_bytes()
    result["environment"]["swap_growth_bytes"] = (
        max(0, swap_after_bytes - swap_before_bytes)
        if swap_after_bytes is not None and swap_before_bytes is not None else None)
    result["wall_s"] = time.perf_counter() - started
    write(args.output / "summary.json", result)
    write(args.output / "progress.json", {"status": "complete", "seed": args.seed,
                                            "completed_arms": [*result["arms"].keys()],
                                            "elapsed_s": result["wall_s"],
                                            "peak_rss_bytes": result["environment"]["peak_rss_bytes"]})
    files = {str(path.relative_to(args.output)): sha(path)
             for path in args.output.rglob("*") if path.is_file()
             and path.name not in {"SHA256.json"} and not path.name.endswith("-wal")
             and not path.name.endswith("-shm")}
    write(args.output / "SHA256.json", files)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rvl-source", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    cli = parser.parse_args()
    try:
        asyncio.run(asyncio.wait_for(run(cli), timeout=1200))
    except BaseException as error:
        if cli.output.exists():
            write(cli.output / "failure.json", {"type": type(error).__name__,
                "message": str(error), "time": time.time()})
        raise
