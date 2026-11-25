#!/usr/bin/env python3
"""Prospective Qwen2.5-0.5B VARE/RVL update and process-recovery experiment.

The base language model is frozen. A trainable vocabulary-logit bias is the
policy adapter. This makes the real-model systems path bounded on CPU while
keeping optimizer, replay, provenance, and recovery operations real. Synthetic
choice rewards are strictly a systems workload, not a capability benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
from typing import Any


DEFAULT_PROTOCOL = Path(__file__).resolve().parents[2] / "results/sprint_xi_20261028/model_coupled_qwen025_v1/protocol.lock.json"
ANSWER_RE = re.compile(r"^\s*([A-D])(?=$|[\s).,:;])")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_bytes_atomic(path, json.dumps(value, sort_keys=True, indent=2).encode() + b"\n")


def _append_journal(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = "0" * 64
    if path.exists():
        rows = [json.loads(row) for row in path.read_text().splitlines() if row]
        if rows:
            previous = rows[-1]["event_sha256"]
    payload = {"sequence": len(rows) if path.exists() else 0, "previous_sha256": previous, **event}
    payload["event_sha256"] = _sha_bytes(_canonical(payload))
    with path.open("ab") as stream:
        stream.write(_canonical(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_protocol(path: Path, *, check_runner: bool = True) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("frozen_before_run") is not True:
        raise RuntimeError("protocol is not frozen")
    if check_runner and _sha_file(Path(__file__).resolve()) != protocol["runner_sha256"]:
        raise RuntimeError("runner SHA does not match frozen protocol")
    return protocol


def _validate_sources(protocol: dict[str, Any], vare_root: Path, rvl_root: Path) -> None:
    if subprocess.check_output(["git", "-C", str(vare_root), "rev-parse", "HEAD"], text=True).strip() != protocol["source"]["vare_revision"]:
        raise RuntimeError("VARE revision mismatch")
    if subprocess.run(["git", "-C", str(vare_root), "diff", "--quiet", "HEAD", "--"]).returncode != 0:
        raise RuntimeError("tracked VARE source modifications detected")
    if subprocess.check_output(["git", "-C", str(rvl_root), "rev-parse", "HEAD"], text=True).strip() != protocol["source"]["rvl_revision"]:
        raise RuntimeError("RVL revision mismatch")
    if subprocess.check_output(["git", "-C", str(rvl_root), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("RVL checkout is not clean")
    for label, root in (("vare", vare_root), ("rvl", rvl_root)):
        for rel, expected in protocol["source_sha256"][label].items():
            actual = _sha_file(root / rel)
            if actual != expected:
                raise RuntimeError(f"{label} source hash mismatch: {rel}")


def _check_runtime(protocol: dict[str, Any], model_path: Path) -> dict[str, Any]:
    import psutil

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("protocol requires macOS arm64")
    if sys.version_info[:3] != tuple(protocol["runtime"]["python_version"]):
        raise RuntimeError("Python version mismatch")
    import torch
    import transformers

    if torch.__version__ != protocol["runtime"]["torch_version"]:
        raise RuntimeError("PyTorch version mismatch")
    if transformers.__version__ != protocol["runtime"]["transformers_version"]:
        raise RuntimeError("Transformers version mismatch")
    if not model_path.is_dir() or protocol["model"]["revision"] not in str(model_path):
        raise RuntimeError("model snapshot path/revision mismatch")
    for name, expected in protocol["model"]["files_sha256"].items():
        if _sha_file(model_path / name) != expected:
            raise RuntimeError(f"model file hash mismatch: {name}")
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage(str(model_path))
    observed = {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "ram_total_bytes": vm.total,
        "ram_available_bytes": vm.available,
        "swap_total_bytes": swap.total,
        "swap_used_bytes": swap.used,
        "swap_free_bytes": swap.free,
        "disk_free_bytes": disk.free,
        "device": "cpu",
    }
    limits = protocol["resource_limits"]
    if vm.available < limits["minimum_available_ram_bytes"]:
        raise RuntimeError("resource preflight: available RAM below protocol minimum")
    if swap.free < limits["minimum_free_swap_bytes"]:
        raise RuntimeError("resource preflight: free swap below protocol minimum")
    if disk.free < limits["minimum_free_disk_bytes"]:
        raise RuntimeError("resource preflight: free disk below protocol minimum")
    return observed


def _render_prompt(question: str, system: str) -> str:
    return (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\n" + question + "\n"
        "Reply with exactly one uppercase option letter (A, B, C, or D)."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def _extract_answer(text: str) -> str | None:
    match = ANSWER_RE.match(text)
    return match.group(1) if match else None


def _load_backend(protocol: dict[str, Any], model_path: Path, checkpoint: Path | None):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(protocol["runtime"]["torch_threads"])
    os.environ["MKL_NUM_THREADS"] = str(protocol["runtime"]["torch_threads"])
    import numpy as np
    import torch
    from rvl_systems.hf_backend import HFLocalBackend

    torch.set_num_threads(int(protocol["runtime"]["torch_threads"]))
    backend = HFLocalBackend(
        model_name=str(model_path),
        max_new_tokens=int(protocol["sampling"]["max_new_tokens"]),
        device="cpu",
        precision="fp32",
    )
    backend.ensure_loaded()
    model = backend.model
    tokenizer = backend.tokenizer
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.register_parameter(
        "vare_action_bias",
        torch.nn.Parameter(torch.zeros(int(model.config.vocab_size), dtype=torch.float32)),
    )

    def add_action_bias(_module, _inputs, logits):
        return logits + model.vare_action_bias.to(dtype=logits.dtype)

    hook = model.lm_head.register_forward_hook(add_action_bias)
    if checkpoint is not None:
        manifest = _restore_policy_state(checkpoint, model, None, torch, np, load_optimizer=False)
        if int(manifest["metadata"]["policy_version"]) < 1:
            raise RuntimeError("checkpoint has invalid policy version")
    return backend, model, tokenizer, hook, torch, np


def _checkpoint_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_hashes = manifest["files_sha256"]
    for name, expected in file_hashes.items():
        if _sha_file(path / name) != expected:
            raise RuntimeError(f"checkpoint file hash mismatch: {name}")
    expected_checkpoint_sha = _sha_bytes(_canonical({"metadata": manifest["metadata"], "files_sha256": file_hashes}))
    if expected_checkpoint_sha != manifest["checkpoint_sha256"]:
        raise RuntimeError("checkpoint manifest hash mismatch")
    return manifest


def _restore_policy_state(path: Path, model: Any, trainer: Any, torch: Any, np: Any, *, load_optimizer: bool) -> dict[str, Any]:
    manifest = _checkpoint_manifest(path)
    bias = np.load(path / "adapter_bias.npy", allow_pickle=False)
    if tuple(bias.shape) != tuple(model.vare_action_bias.shape):
        raise RuntimeError("adapter shape mismatch")
    with torch.no_grad():
        model.vare_action_bias.copy_(torch.from_numpy(bias).to(dtype=model.vare_action_bias.dtype))
    if load_optimizer:
        if trainer is None:
            raise ValueError("trainer required for optimizer restore")
        trainer.optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False))
    rng = torch.load(path / "cpu_rng.pt", map_location="cpu", weights_only=False)
    torch.set_rng_state(rng)
    return manifest


def _tensor_sha(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return _sha_bytes(value.numpy().tobytes())


def _save_checkpoint(
    target: Path,
    *,
    model: Any,
    trainer: Any,
    torch: Any,
    np: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(target.name + f".stage-{os.getpid()}")
    stage.mkdir()
    with (stage / "adapter_bias.npy").open("xb") as stream:
        np.save(stream, model.vare_action_bias.detach().cpu().numpy(), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    with (stage / "optimizer.pt").open("xb") as stream:
        torch.save(trainer.optimizer.state_dict(), stream)
        stream.flush()
        os.fsync(stream.fileno())
    with (stage / "cpu_rng.pt").open("xb") as stream:
        torch.save(torch.get_rng_state().cpu(), stream)
        stream.flush()
        os.fsync(stream.fileno())
    file_hashes = {name: _sha_file(stage / name) for name in ("adapter_bias.npy", "optimizer.pt", "cpu_rng.pt")}
    metadata = {
        **metadata,
        "adapter_sha256": file_hashes["adapter_bias.npy"],
        "optimizer_serialization_sha256": file_hashes["optimizer.pt"],
        "rng_serialization_sha256": file_hashes["cpu_rng.pt"],
    }
    checkpoint_sha = _sha_bytes(_canonical({"metadata": metadata, "files_sha256": file_hashes}))
    manifest = {"metadata": metadata, "files_sha256": file_hashes, "checkpoint_sha256": checkpoint_sha}
    _write_json_atomic(stage / "manifest.json", manifest)
    fd = os.open(stage, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(stage, target)
    fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return manifest


def _write_rollout(protocol: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    model_path = Path(protocol["model"]["local_snapshot"])
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    backend, model, tokenizer, hook, torch, _np = _load_backend(protocol, model_path, checkpoint)
    policy_version = int(args.policy_version)
    if checkpoint is None:
        adapter_sha = _tensor_sha(model.vare_action_bias)
        policy_sha = _sha_bytes(_canonical({
            "model_revision": protocol["model"]["revision"],
            "model_weights_sha256": protocol["model"]["files_sha256"]["model.safetensors"],
            "adapter_sha256": adapter_sha,
        }))
    else:
        checkpoint_manifest = _checkpoint_manifest(checkpoint)
        if int(checkpoint_manifest["metadata"]["policy_version"]) != policy_version:
            raise RuntimeError("worker policy version differs from checkpoint version")
        policy_sha = checkpoint_manifest["checkpoint_sha256"]
    cohort = protocol["rollout_cohorts"][args.cohort]
    records: list[dict[str, Any]] = []
    selected = None
    started = time.monotonic()
    total_generated_tokens = 0
    for task_index, spec in enumerate(cohort):
        prompt = _render_prompt(spec["question"], protocol["task"]["system_prompt"])
        generations = asyncio.run(backend.generate(
            spec["prompt_id"], prompt,
            n=int(protocol["sampling"]["group_size"]),
            temperature=float(protocol["sampling"]["temperature"]),
            seed=int(protocol["sampling"]["base_seed"]) + int(args.seed_offset) + task_index,
        ))
        rewards = [float(_extract_answer(item.response) == spec["answer_key"]) for item in generations]
        group = {
            "prompt_id": spec["prompt_id"],
            "prompt": prompt,
            "question": spec["question"],
            "answer_key": spec["answer_key"],
            "group_id": f"sprint-xi-v1:{args.cohort}:{spec['prompt_id']}",
            "policy_id": f"policy-v{policy_version}",
            "policy_version": policy_version,
            "policy_checkpoint_sha256": policy_sha,
            "verifier_name": protocol["task"]["verifier_name"],
            "verifier_version": int(protocol["task"]["verifier_version"]),
            "verifier_config_sha256": protocol["task"]["verifier_config_sha256"],
            "declared_group_size": int(protocol["sampling"]["group_size"]),
            "rewards": rewards,
            "generations": [],
        }
        for generation, reward in zip(generations, rewards, strict=True):
            generation_record = {
                "prompt_id": generation.prompt_id,
                "prompt": generation.prompt,
                "response": generation.response,
                "logprob": generation.logprob,
                "token_count": generation.token_count,
                "latency_s": generation.latency_s,
                "reward": reward,
                "metadata": generation.metadata,
            }
            group["generations"].append(generation_record)
            total_generated_tokens += int(generation.token_count)
        records.append(group)
        if len(set(rewards)) > 1:
            selected = group
            break
    elapsed = time.monotonic() - started
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "cohort": args.cohort,
        "status": "mixed_reward_group_found" if selected else "no_mixed_reward_group",
        "model_revision": protocol["model"]["revision"],
        "model_file_sha256": protocol["model"]["files_sha256"]["model.safetensors"],
        "policy_version": policy_version,
        "policy_checkpoint_sha256": policy_sha,
        "task_records": records,
        "selected_group_id": selected["group_id"] if selected else None,
        "selected_prompt_id": selected["prompt_id"] if selected else None,
        "total_generated_tokens": total_generated_tokens,
        "wall_seconds": elapsed,
        "fault_after_persist": bool(args.fault_after_persist),
        "seed_offset": int(args.seed_offset),
    }
    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists() or artifact.with_suffix(artifact.suffix + ".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite rollout artifact {artifact}")
    _write_json_atomic(artifact, result)
    manifest = {
        "artifact_sha256": _sha_file(artifact),
        "protocol_sha256": _sha_file(Path(args.protocol)),
        "artifact_bytes": artifact.stat().st_size,
        "persisted_unix": time.time(),
    }
    _write_json_atomic(artifact.with_suffix(artifact.suffix + ".manifest.json"), manifest)
    if args.fault_after_persist:
        os.kill(os.getpid(), signal.SIGKILL)
    hook.remove()
    return {"status": result["status"], "artifact_sha256": manifest["artifact_sha256"], "wall_seconds": elapsed}


def _ack_rollout(args: argparse.Namespace) -> dict[str, Any]:
    artifact = Path(args.artifact)
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = _sha_file(artifact)
    if actual != manifest["artifact_sha256"]:
        raise RuntimeError("persisted rollout artifact hash mismatch")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("status") != "mixed_reward_group_found" or not payload.get("selected_group_id"):
        raise RuntimeError("persisted rollout is not a complete nonconstant group")
    ack = {
        "status": "recovered_existing_rollout",
        "artifact_sha256": actual,
        "group_id": payload["selected_group_id"],
        "generation_replayed": False,
        "effective_group_deliveries": 1,
        "ack_unix": time.time(),
    }
    _write_json_atomic(Path(args.ack_output), ack)
    return ack


def _build_experiences(protocol: dict[str, Any], rollout: dict[str, Any]):
    vare_root = Path(rollout["_vare_root"])
    sys.path.insert(0, str(vare_root / "src"))
    from vare.types import Attempt, Experience, Task, Verification

    group = next(row for row in rollout["task_records"] if row["group_id"] == rollout["selected_group_id"])
    task = Task(
        id=group["prompt_id"],
        prompt=group["prompt"],
        family="synthetic-choice-systems",
        metadata={
            "vare_rollout_group": group["group_id"],
            "vare_rollout_group_size": group["declared_group_size"],
            "policy_checkpoint_sha256": group["policy_checkpoint_sha256"],
        },
    )
    experiences = []
    for index, generation in enumerate(group["generations"]):
        metadata = {
            **generation["metadata"],
            "vare_rollout_group": group["group_id"],
            "vare_rollout_group_size": group["declared_group_size"],
            "policy_checkpoint_sha256": group["policy_checkpoint_sha256"],
            "token_count": generation["token_count"],
            "rvl_generation": generation,
        }
        attempt = Attempt(
            task=task,
            output=generation["response"],
            policy_id=group["policy_id"],
            policy_version=group["policy_version"],
            created_step=group["policy_version"],
            logprob=float(generation["logprob"]),
            latency_ms=float(generation["latency_s"]) * 1000.0,
            metadata=metadata,
        )
        verification = Verification(
            score=float(generation["reward"]),
            passed=bool(generation["reward"]),
            confidence=1.0,
            verifier_version=int(group["verifier_version"]),
            verifier_name=group["verifier_name"],
            trusted=True,
            metadata={
                "rvl_verification_config_sha256": group["verifier_config_sha256"],
                "latency_s": 0.0,
                "source": "independent_sprint_xi_exact_choice_verifier",
            },
        )
        experiences.append(Experience(attempt, verification, 0, 0, 0.0))
    return experiences


def _learn(protocol: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig

    model_path = Path(protocol["model"]["local_snapshot"])
    initial = Path(args.checkpoint) if args.checkpoint else None
    backend, model, tokenizer, hook, torch, np = _load_backend(protocol, model_path, None)
    if initial is None:
        adapter_sha = _tensor_sha(model.vare_action_bias)
        active_policy_sha = _sha_bytes(_canonical({
            "model_revision": protocol["model"]["revision"],
            "model_weights_sha256": protocol["model"]["files_sha256"]["model.safetensors"],
            "adapter_sha256": adapter_sha,
        }))
        parent_version = 0
    else:
        initial_manifest = _checkpoint_manifest(initial)
        active_policy_sha = initial_manifest["checkpoint_sha256"]
        parent_version = int(initial_manifest["metadata"]["policy_version"])
    trainer = HFCausalLMGRPOTrainer(
        model,
        config=HFTTrainerConfig(
            learning_rate=float(protocol["optimizer"]["learning_rate"]),
            clip_eps=float(protocol["optimizer"]["clip_epsilon"]),
            max_grad_norm=float(protocol["optimizer"]["max_grad_norm"]),
            advantage_eps=float(protocol["optimizer"]["advantage_epsilon"]),
            clip_advantage=float(protocol["optimizer"]["clip_advantage"]),
            objective_backend="torch",
        ),
    )
    torch.manual_seed(int(protocol["optimizer"]["torch_seed"]))
    if initial is not None:
        _restore_policy_state(initial, model, trainer, torch, np, load_optimizer=True)
    rollout = json.loads(Path(args.rollout).read_text(encoding="utf-8"))
    rollout["_vare_root"] = str(Path(args.vare_root))
    group = next(row for row in rollout["task_records"] if row["group_id"] == rollout["selected_group_id"])
    if group["policy_checkpoint_sha256"] != rollout["policy_checkpoint_sha256"]:
        raise RuntimeError("rollout policy checkpoint identity mismatch")
    if args.admission == "vaware" and (
        group["policy_version"] != parent_version
        or group["policy_checkpoint_sha256"] != active_policy_sha
    ):
        raise RuntimeError("VARE freshness gate rejected stale policy group")
    experiences = _build_experiences(protocol, rollout)
    from vare.config import LagConfig, ReplayConfig
    from vare.lag import LagController, make_experience_freshness_assessor
    from vare.replay import PrioritizedReplay
    from vare.integrations.rvl_grpo import RVLGRPOHooks
    from rvl_systems.types import Generation, VerifiedGeneration

    replay = PrioritizedReplay(ReplayConfig(capacity=32), seed=int(protocol["optimizer"]["torch_seed"]))
    accepted = replay.add_group(experiences, [1.0] * len(experiences))
    if not accepted:
        raise RuntimeError("VARE group-aware replay rejected complete rollout group")
    active_version = parent_version
    lag_controller = LagController(LagConfig(max_policy_lag=0, max_verifier_lag=0, max_shift=0.0))
    assessor = make_experience_freshness_assessor(
        lag_controller,
        active_policy_version=active_version,
        active_verifier_version=int(protocol["task"]["verifier_version"]),
    )
    if args.admission == "vaware":
        def freshness(experience):
            decision = assessor(experience)
            if not decision.admitted:
                return None
            if experience.attempt.metadata.get("policy_checkpoint_sha256") != active_policy_sha:
                return None
            if experience.verification.metadata.get("rvl_verification_config_sha256") != protocol["task"]["verifier_config_sha256"]:
                return None
            return decision.freshness
    else:
        freshness = lambda _experience: 1.0
    samples_exp = replay.sample_current(len(experiences), freshness_fn=freshness, grouped=True)
    if len(samples_exp) != len(experiences):
        raise RuntimeError("freshness admission did not return the complete GRPO group")
    verified = []
    for experience in samples_exp:
        raw = experience.attempt.metadata["rvl_generation"]
        generation = Generation(
            prompt_id=raw["prompt_id"],
            prompt=raw["prompt"],
            response=raw["response"],
            logprob=float(raw["logprob"]),
            token_count=int(raw["token_count"]),
            latency_s=float(raw["latency_s"]),
            metadata=dict(raw["metadata"]),
        )
        verified.append(VerifiedGeneration(
            generation=generation,
            reward=float(experience.verification.score),
            verifier_latency_s=0.0,
            verifier_version=experience.verification.verifier_version,
            metadata=dict(experience.verification.metadata),
        ))
    advantages = RVLGRPOHooks._group_relative_advantages(
        samples_exp,
        verified,
        eps=float(protocol["optimizer"]["advantage_epsilon"]),
        clip=float(protocol["optimizer"]["clip_advantage"]),
    )
    if advantages is None or len(set(advantages)) < 2:
        raise RuntimeError("nonconstant reward group did not yield nonconstant advantages")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    journal = run_dir / "learner.journal.jsonl"
    update_id = str(args.update_id)
    before_bias = _tensor_sha(model.vare_action_bias)
    _append_journal(journal, {
        "kind": "update_prepare",
        "update_id": update_id,
        "parent_policy_version": parent_version,
        "parent_checkpoint_sha256": active_policy_sha,
        "group_id": group["group_id"],
        "group_sha256": _sha_file(Path(args.rollout)),
        "freshness_policy": args.admission,
        "optimizer_attempt": 1,
        "unix": time.time(),
    })
    metrics = trainer.train_step(verified, advantages=advantages)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
        raise RuntimeError("optimizer produced missing or non-finite gradients")
    if not any(torch.count_nonzero(g).item() for g in gradients):
        raise RuntimeError("optimizer produced only zero gradients")
    after_bias = _tensor_sha(model.vare_action_bias)
    _append_journal(journal, {
        "kind": "optimizer_applied_uncommitted" if args.fault_after_optimizer else "optimizer_applied",
        "update_id": update_id,
        "attempt_id": f"{update_id}-attempt-1",
        "optimizer_step_count": sum(int(state.get("step", 0)) for state in trainer.optimizer.state.values()),
        "bias_before_sha256": before_bias,
        "bias_after_sha256": after_bias,
        "metrics": metrics,
        "unix": time.time(),
    })
    if args.fault_after_optimizer:
        os.kill(os.getpid(), signal.SIGKILL)
    candidate_version = parent_version + 1
    manifest = _save_checkpoint(
        Path(args.checkpoint_out),
        model=model,
        trainer=trainer,
        torch=torch,
        np=np,
        metadata={
            "protocol_id": protocol["protocol_id"],
            "model_revision": protocol["model"]["revision"],
            "policy_version": candidate_version,
            "parent_policy_version": parent_version,
            "parent_checkpoint_sha256": active_policy_sha,
            "update_id": update_id,
            "rollout_group_id": group["group_id"],
            "rollout_group_sha256": _sha_file(Path(args.rollout)),
            "admission_policy": args.admission,
            "optimizer_step_count": sum(int(state.get("step", 0)) for state in trainer.optimizer.state.values()),
            "mean_reward": metrics["mean_reward"],
            "optimizer_metrics": metrics,
        },
    )
    _append_journal(journal, {
        "kind": "checkpoint_committed",
        "update_id": update_id,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "policy_version": candidate_version,
        "unix": time.time(),
    })
    result = {
        "status": "committed",
        "update_id": update_id,
        "parent_policy_version": parent_version,
        "policy_version": candidate_version,
        "parent_checkpoint_sha256": active_policy_sha,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_path": str(args.checkpoint_out),
        "bias_before_sha256": before_bias,
        "bias_after_sha256": after_bias,
        "bias_changed": before_bias != after_bias,
        "group_id": group["group_id"],
        "group_rewards": group["rewards"],
        "advantages": advantages,
        "metrics": metrics,
        "replay_admitted": len(samples_exp),
        "replay_complete_group": len(samples_exp) == int(group["declared_group_size"]),
        "optimizer_step_count": sum(int(state.get("step", 0)) for state in trainer.optimizer.state.values()),
        "peak_rss_bytes": int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss),
        "run_dir": str(run_dir),
    }
    _write_json_atomic(run_dir / "summary.json", result)
    hook.remove()
    return result


def _evaluate(protocol: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    model_path = Path(protocol["model"]["local_snapshot"])
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    backend, model, tokenizer, hook, torch, _np = _load_backend(protocol, model_path, checkpoint)
    model.eval()
    if args.bias_override is not None:
        token_id = int(tokenizer.encode(args.override_token, add_special_tokens=False)[0])
        with torch.no_grad():
            model.vare_action_bias[token_id] = float(args.bias_override)
    scores = []
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"]
    if any(len(ids) != 1 for ids in letter_ids) or len({ids[0] for ids in letter_ids}) != 4:
        raise RuntimeError("evaluation requires four distinct single-token choice labels")
    ids = [int(row[0]) for row in letter_ids]
    for spec in protocol["evaluation"]["tasks"]:
        prompt = _render_prompt(spec["question"], protocol["task"]["system_prompt"])
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            logits = model(input_ids=encoded["input_ids"]).logits[0, -1].float()
            subset = torch.softmax(logits[ids], dim=0)
        correct_index = "ABCD".index(spec["answer_key"])
        scores.append({
            "task_id": spec["prompt_id"],
            "answer_key": spec["answer_key"],
            "choice_probabilities": {letter: float(subset[i]) for i, letter in enumerate("ABCD")},
            "correct_choice_probability": float(subset[correct_index]),
        })
    mean = sum(row["correct_choice_probability"] for row in scores) / len(scores)
    if checkpoint is None:
        adapter_sha = _tensor_sha(model.vare_action_bias)
        policy_sha = _sha_bytes(_canonical({
            "model_revision": protocol["model"]["revision"],
            "model_weights_sha256": protocol["model"]["files_sha256"]["model.safetensors"],
            "adapter_sha256": adapter_sha,
        }))
    else:
        policy_sha = _checkpoint_manifest(checkpoint)["checkpoint_sha256"]
    result = {
        "status": "evaluated",
        "checkpoint_sha256": _checkpoint_manifest(checkpoint)["checkpoint_sha256"] if checkpoint else None,
        "policy_sha256": policy_sha,
        "adapter_sha256": _tensor_sha(model.vare_action_bias),
        "bias_override": args.bias_override,
        "tasks": scores,
        "mean_correct_choice_probability": mean,
        "scope": "synthetic verifier evaluation; no coding or generalization claim",
    }
    _write_json_atomic(Path(args.output), result)
    hook.remove()
    return result


def _system_preflight(protocol: dict[str, Any], model_path: Path) -> dict[str, Any]:
    import psutil
    import shutil

    state = _check_runtime(protocol, model_path)
    state["disk_free_bytes"] = shutil.disk_usage(str(model_path)).free
    return state


def _child_environment(vare_root: Path, rvl_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": os.pathsep.join([str(vare_root / "src"), str(rvl_root / "src"), env.get("PYTHONPATH", "")]),
    })
    return env


def _command_base(script: Path, mode: str, protocol_path: Path, vare_root: Path, rvl_root: Path) -> list[str]:
    return [sys.executable, str(script), mode, "--protocol", str(protocol_path), "--vare-root", str(vare_root), "--rvl-root", str(rvl_root)]


def _sample_process_tree(proc: subprocess.Popen, psutil: Any) -> int:
    try:
        root = psutil.Process(proc.pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    total = 0
    for item in processes:
        try:
            total += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _run_processes(commands: list[tuple[str, list[str]]], *, output_root: Path, env: dict[str, str], limits: dict[str, Any], baseline_swap: int) -> list[dict[str, Any]]:
    import psutil

    running = []
    started = time.monotonic()
    for name, command in commands:
        log_path = output_root / f"process-{name}.log"
        stream = log_path.open("wb")
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        running.append({"name": name, "proc": proc, "stream": stream, "log": str(log_path), "peak_tree_rss": 0})
    while any(item["proc"].poll() is None for item in running):
        current_swap = psutil.swap_memory().used
        ram_available = psutil.virtual_memory().available
        total_rss = 0
        for item in running:
            if item["proc"].poll() is None:
                tree_rss = _sample_process_tree(item["proc"], psutil)
                item["peak_tree_rss"] = max(item["peak_tree_rss"], tree_rss)
                total_rss += tree_rss
        abort_reason = None
        if current_swap - baseline_swap > limits["max_swap_growth_bytes"]:
            abort_reason = "host swap growth cap exceeded"
        elif ram_available < limits["hard_minimum_available_ram_bytes"]:
            abort_reason = "host available RAM emergency floor crossed"
        elif total_rss > limits["max_process_tree_rss_bytes"]:
            abort_reason = "model process tree RSS cap exceeded"
        elif time.monotonic() - started > limits["max_wall_seconds"]:
            abort_reason = "campaign wall time cap exceeded"
        if abort_reason:
            for item in running:
                if item["proc"].poll() is None:
                    item["proc"].terminate()
            for item in running:
                try:
                    item["proc"].wait(timeout=3)
                except subprocess.TimeoutExpired:
                    item["proc"].kill()
            raise RuntimeError(abort_reason)
        time.sleep(0.2)
    results = []
    for item in running:
        rc = item["proc"].wait()
        item["stream"].close()
        results.append({
            "name": item["name"],
            "returncode": rc,
            "peak_process_tree_rss_bytes": item["peak_tree_rss"],
            "wall_seconds": time.monotonic() - started,
            "log": item["log"],
        })
    return results


def _write_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "sha256-manifest.json":
            continue
        rows.append({"path": path.relative_to(root).as_posix(), "sha256": _sha_file(path), "bytes": path.stat().st_size})
    manifest = {"schema_version": 1, "files": rows}
    _write_json_atomic(root / "sha256-manifest.json", manifest)
    return manifest


def _run_campaign(protocol_path: Path, vare_root: Path, rvl_root: Path, output: Path) -> dict[str, Any]:
    import psutil

    protocol = _read_protocol(protocol_path)
    _validate_sources(protocol, vare_root, rvl_root)
    model_path = Path(protocol["model"]["local_snapshot"])
    preflight = _system_preflight(protocol, model_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite campaign output: {output}")
    output.mkdir(parents=True)
    _write_json_atomic(output / "preflight.json", preflight)
    started = time.monotonic()
    initial_swap = psutil.swap_memory().used
    env = _child_environment(vare_root, rvl_root)
    script = Path(__file__).resolve()
    base = _command_base(script, "worker", protocol_path, vare_root, rvl_root)
    process_records: list[dict[str, Any]] = []

    def run_one(name: str, mode: str, arguments: list[str], expected: int = 0) -> dict[str, Any]:
        command = _command_base(script, mode, protocol_path, vare_root, rvl_root) + arguments
        rows = _run_processes([(name, command)], output_root=output, env=env, limits=protocol["resource_limits"], baseline_swap=initial_swap)
        process_records.extend(rows)
        row = rows[0]
        if row["returncode"] != expected:
            raise RuntimeError(f"child {name} exited {row['returncode']}, expected {expected}; see {row['log']}")
        return row

    rollouts = output / "rollouts"
    checkpoints = output / "checkpoints"
    runs = output / "learner-runs"
    g0 = rollouts / "fresh-v0.json"
    run_one("worker-fresh-v0-sigkill", "worker", ["--cohort", "fresh_v0", "--policy-version", "0", "--seed-offset", "0", "--artifact", str(g0), "--fault-after-persist"], expected=-signal.SIGKILL)
    ack = output / "worker-recovery-ack.json"
    run_one("worker-recovery-ack", "ack", ["--artifact", str(g0), "--ack-output", str(ack)])

    # A stale rollout is deliberately in flight while the learner commits v1.
    stale_out = rollouts / "queued-v0-stale.json"
    cp1 = checkpoints / "candidate-v1"
    run1_dir = runs / "update-v1"
    worker_cmd = _command_base(script, "worker", protocol_path, vare_root, rvl_root) + [
        "--cohort", "queued_stale_v0", "--policy-version", "0", "--seed-offset", "17", "--artifact", str(stale_out)
    ]
    learner_cmd = _command_base(script, "learn", protocol_path, vare_root, rvl_root) + [
        "--rollout", str(g0), "--checkpoint-out", str(cp1), "--run-dir", str(run1_dir),
        "--update-id", "update-v1", "--admission", "vaware"
    ]
    process_records.extend(_run_processes(
        [("async-stale-rollout-worker", worker_cmd), ("learner-update-v1", learner_cmd)],
        output_root=output, env=env, limits=protocol["resource_limits"], baseline_swap=initial_swap,
    ))
    for row in process_records[-2:]:
        if row["returncode"] != 0:
            raise RuntimeError(f"parallel child {row['name']} exited {row['returncode']}; see {row['log']}")
    cp1_sha = _checkpoint_manifest(cp1)["checkpoint_sha256"]

    # Replay receives the queued v0 group after v1 is active. VARE rejects it;
    # the explicitly naive comparator admits the same group for one branch step.
    stale_payload = json.loads(stale_out.read_text(encoding="utf-8"))
    stale_payload["_vare_root"] = str(vare_root)
    stale_experiences = _build_experiences(protocol, stale_payload)
    from vare.config import LagConfig, ReplayConfig
    from vare.lag import LagController, make_experience_freshness_assessor
    from vare.replay import PrioritizedReplay

    stale_replay = PrioritizedReplay(ReplayConfig(capacity=32), seed=int(protocol["optimizer"]["torch_seed"]))
    stale_added = stale_replay.add_group(stale_experiences, [1.0] * len(stale_experiences))
    lag_controller = LagController(LagConfig(max_policy_lag=0, max_verifier_lag=0, max_shift=0.0))
    stale_assessor = make_experience_freshness_assessor(lag_controller, active_policy_version=1, active_verifier_version=0)
    stale_freshness = lambda exp: stale_assessor(exp).freshness if stale_assessor(exp).admitted and exp.attempt.metadata.get("policy_checkpoint_sha256") == stale_payload["policy_checkpoint_sha256"] else None
    vare_stale_batch = stale_replay.sample_current(4, freshness_fn=stale_freshness, grouped=True)
    naive_stale_batch = stale_replay.sample_current(4, freshness_fn=lambda _exp: 1.0, grouped=True)
    stale_sidecar = json.loads(stale_out.with_suffix(stale_out.suffix + ".manifest.json").read_text())
    stale_gate = {
        "group_id": stale_payload["selected_group_id"],
        "rollout_policy_version": 0,
        "active_policy_version": 1,
        "policy_lag_limit": 0,
        "replay_group_added": stale_added,
        "vare_admitted_members": len(vare_stale_batch),
        "naive_admitted_members": len(naive_stale_batch),
        "vare_rejected_stale_group": stale_added and len(vare_stale_batch) == 0,
        "naive_accepted_stale_group": len(naive_stale_batch) == 4,
        "queue_seconds": max(0.0, time.time() - float(stale_sidecar["persisted_unix"])),
    }
    # persisted_unix lives in the sidecar, and the protocol claim is only the
    # admitted/rejected identity; queue latency is captured from worker logs.
    _write_json_atomic(output / "staleness-admission.json", stale_gate)
    if not stale_gate["vare_rejected_stale_group"] or not stale_gate["naive_accepted_stale_group"]:
        raise RuntimeError("staleness admission contrast did not behave as preregistered")

    naive_cp = checkpoints / "naive-stale-v2"
    run_one("naive-stale-learner", "learn", [
        "--rollout", str(stale_out), "--checkpoint", str(cp1), "--checkpoint-out", str(naive_cp),
        "--run-dir", str(runs / "naive-stale-v2"), "--update-id", "naive-stale-v2", "--admission", "naive"
    ])

    fresh_v1 = rollouts / "fresh-v1.json"
    run_one("worker-fresh-v1", "worker", [
        "--cohort", "fresh_v1", "--policy-version", "1", "--seed-offset", "31", "--checkpoint", str(cp1), "--artifact", str(fresh_v1)
    ])
    ref_cp = checkpoints / "reference-v2"
    run_one("synchronous-reference-v2", "learn", [
        "--rollout", str(fresh_v1), "--checkpoint", str(cp1), "--checkpoint-out", str(ref_cp),
        "--run-dir", str(runs / "reference-v2"), "--update-id", "update-v2", "--admission", "vaware"
    ])

    fault_dir = runs / "faulted-v2-attempt"
    run_one("learner-sigkill-after-optimizer", "learn", [
        "--rollout", str(fresh_v1), "--checkpoint", str(cp1), "--checkpoint-out", str(checkpoints / "uncommitted-v2"),
        "--run-dir", str(fault_dir), "--update-id", "update-v2", "--admission", "vaware", "--fault-after-optimizer"
    ], expected=-signal.SIGKILL)
    recovery_cp = checkpoints / "recovered-v2"
    run_one("learner-recovery-v2", "learn", [
        "--rollout", str(fresh_v1), "--checkpoint", str(cp1), "--checkpoint-out", str(recovery_cp),
        "--run-dir", str(runs / "recovery-v2"), "--update-id", "update-v2", "--admission", "vaware"
    ])

    evals = output / "evaluations"
    evals.mkdir()
    run_one("evaluate-base", "evaluate", ["--output", str(evals / "base.json")])
    run_one("evaluate-v1", "evaluate", ["--checkpoint", str(cp1), "--output", str(evals / "v1.json")])
    run_one("evaluate-reference-v2", "evaluate", ["--checkpoint", str(ref_cp), "--output", str(evals / "reference-v2.json")])
    run_one("evaluate-recovered-v2", "evaluate", ["--checkpoint", str(recovery_cp), "--output", str(evals / "recovered-v2.json")])
    run_one("evaluate-naive-stale-v2", "evaluate", ["--checkpoint", str(naive_cp), "--output", str(evals / "naive-stale-v2.json")])

    base_score = json.loads((evals / "base.json").read_text())["mean_correct_choice_probability"]
    v1_score = json.loads((evals / "v1.json").read_text())["mean_correct_choice_probability"]
    v2_score = json.loads((evals / "recovered-v2.json").read_text())["mean_correct_choice_probability"]
    active = cp1 if v1_score >= base_score else None
    active_score = v1_score if active is not None else base_score
    if v2_score >= active_score:
        active = recovery_cp
        active_score = v2_score
    active_sha = (
        _checkpoint_manifest(active)["checkpoint_sha256"]
        if active is not None
        else json.loads((evals / "base.json").read_text())["policy_sha256"]
    )
    pointer = output / "promoted-policy.json"
    _write_json_atomic(pointer, {
        "policy_version": int(_checkpoint_manifest(active)["metadata"]["policy_version"]) if active is not None else 0,
        "checkpoint_path": str(active) if active is not None else None,
        "checkpoint_sha256": active_sha,
        "promotion_rule": "mean synthetic correct-choice probability is nondecreasing",
        "base_score": base_score,
        "v1_score": v1_score,
        "v2_score": v2_score,
        "selected_checkpoint": active.name if active is not None else "base",
    })
    # Negative-control candidate: large adverse logit bias must be rejected;
    # the durable active pointer and checkpoint identity must remain unchanged.
    run_one("evaluate-adverse-candidate", "evaluate", [
        *( ["--checkpoint", str(active)] if active is not None else [] ),
        "--bias-override", "-20.0", "--override-token", "B", "--output", str(evals / "adverse-candidate.json")
    ])
    adverse_score = json.loads((evals / "adverse-candidate.json").read_text())["mean_correct_choice_probability"]
    rollback_rejected = adverse_score < active_score
    rollback = {
        "incumbent_policy_sha256": active_sha,
        "adverse_candidate_score": adverse_score,
        "incumbent_score": base_score if active is None else (v1_score if active == cp1 else v2_score),
        "adverse_candidate_rejected": rollback_rejected,
        "active_pointer_sha256_after_rejection": _sha_file(pointer),
        "active_checkpoint_sha256_after_rejection": _checkpoint_manifest(active)["checkpoint_sha256"] if active is not None else active_sha,
        "active_checkpoint_unchanged": (_checkpoint_manifest(active)["checkpoint_sha256"] if active is not None else active_sha) == active_sha,
    }
    _write_json_atomic(output / "rollback-negative-control.json", rollback)

    final = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha_file(protocol_path),
        "runner_sha256": _sha_file(Path(__file__).resolve()),
        "vare_revision": protocol["source"]["vare_revision"],
        "rvl_revision": protocol["source"]["rvl_revision"],
        "model_revision": protocol["model"]["revision"],
        "model_file_sha256": protocol["model"]["files_sha256"]["model.safetensors"],
        "status": "completed",
        "worker_sigkill_recovered": json.loads(ack.read_text())["generation_replayed"] is False,
        "learner_sigkill_returncode": -signal.SIGKILL,
        "version_aware_stale_rejection": stale_gate,
        "optimizer_attempts_for_v2_recovery": 2,
        "effective_v2_commits": 1,
        "reference_checkpoint_sha256": _checkpoint_manifest(ref_cp)["checkpoint_sha256"],
        "recovered_checkpoint_sha256": _checkpoint_manifest(recovery_cp)["checkpoint_sha256"],
        "naive_stale_checkpoint_sha256": _checkpoint_manifest(naive_cp)["checkpoint_sha256"],
        "evaluation": {"base": base_score, "v1": v1_score, "reference_v2": json.loads((evals / "reference-v2.json").read_text())["mean_correct_choice_probability"], "recovered_v2": v2_score, "naive_stale_v2": json.loads((evals / "naive-stale-v2.json").read_text())["mean_correct_choice_probability"]},
        "promotion": json.loads(pointer.read_text()),
        "rollback_negative_control": rollback,
        "processes": process_records,
        "campaign_wall_seconds": time.monotonic() - started,
        "host_swap_growth_bytes": psutil.swap_memory().used - initial_swap,
        "host_ram_available_bytes_after": psutil.virtual_memory().available,
        "scope": "real Qwen inference and real GRPO optimizer updates on a trainable output-bias adapter with synthetic exact-choice rewards; systems/recovery feasibility only",
        "capability_claim": False,
    }
    _write_json_atomic(output / "campaign-summary.json", final)
    _write_manifest(output)
    return final


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("run", "worker", "ack", "learn", "evaluate"):
        p = sub.add_parser(mode)
        p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        p.add_argument("--vare-root", type=Path, required=True)
        p.add_argument("--rvl-root", type=Path, required=True)
        if mode == "run":
            p.add_argument("--output", type=Path, required=True)
        elif mode == "worker":
            p.add_argument("--cohort", required=True)
            p.add_argument("--policy-version", type=int, required=True)
            p.add_argument("--seed-offset", type=int, default=0)
            p.add_argument("--checkpoint", type=Path)
            p.add_argument("--artifact", type=Path, required=True)
            p.add_argument("--fault-after-persist", action="store_true")
        elif mode == "ack":
            p.add_argument("--artifact", type=Path, required=True)
            p.add_argument("--ack-output", type=Path, required=True)
        elif mode == "learn":
            p.add_argument("--rollout", type=Path, required=True)
            p.add_argument("--checkpoint", type=Path)
            p.add_argument("--checkpoint-out", type=Path, required=True)
            p.add_argument("--run-dir", type=Path, required=True)
            p.add_argument("--update-id", required=True)
            p.add_argument("--admission", choices=("vaware", "naive"), required=True)
            p.add_argument("--fault-after-optimizer", action="store_true")
        elif mode == "evaluate":
            p.add_argument("--checkpoint", type=Path)
            p.add_argument("--bias-override", type=float)
            p.add_argument("--override-token", default="B")
            p.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    protocol = _read_protocol(args.protocol)
    vare_root, rvl_root = args.vare_root.resolve(), args.rvl_root.resolve()
    _validate_sources(protocol, vare_root, rvl_root)
    sys.path.insert(0, str(vare_root / "src"))
    sys.path.insert(0, str(rvl_root / "src"))
    if args.mode == "run":
        result = _run_campaign(args.protocol.resolve(), vare_root, rvl_root, args.output.resolve())
    elif args.mode == "worker":
        _check_runtime(protocol, Path(protocol["model"]["local_snapshot"]))
        result = _write_rollout(protocol, args)
    elif args.mode == "ack":
        result = _ack_rollout(args)
    elif args.mode == "learn":
        _check_runtime(protocol, Path(protocol["model"]["local_snapshot"]))
        result = _learn(protocol, args)
    else:
        _check_runtime(protocol, Path(protocol["model"]["local_snapshot"]))
        result = _evaluate(protocol, args)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
