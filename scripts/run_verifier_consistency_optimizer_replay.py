"""Run a bounded fixed-data verifier admission/optimizer reference replay.

The inputs are the four immutable reward-calibration groups. The script samples
no outputs and makes no active-policy promotion or capability claim.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

PROTOCOL = ROOT / "benchmarks/verifier_consistent_async_grpo_v7/optimizer-replay-v1.json"
CALIBRATION_PROTOCOL = ROOT / "benchmarks/verifier_consistent_async_grpo_v7/reward-calibration-v1.json"
CALIBRATION_RUN = ROOT / "results/verifier_consistent_async_grpo_v7/reward-calibration-v1/pilot-1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def hash_tree(value: Any, digest) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: str(item)):
            hash_tree(str(key), digest)
            hash_tree(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            hash_tree(item, digest)
    elif isinstance(value, bytes):
        digest.update(b"bytes\0" + value)
    else:
        digest.update(type(value).__name__.encode() + b"\0")
        digest.update(repr(value).encode() + b"\0")


def sha_tree(value: Any) -> str:
    digest = hashlib.sha256()
    hash_tree(value, digest)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        temp.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    if path.read_bytes() != payload:
        raise RuntimeError(f"atomic read-back failed: {path}")
    return hashlib.sha256(payload).hexdigest()


def write_group(store, root, policy, task, rows: list[dict[str, Any]], *,
                group_id: str, version: int, verifier_name: str,
                verifier_sha: str, score_key: str, generation_factory) -> None:
    from vare.trajectory_store import AgentStep
    from vare.types import Verification

    ids = [f"{group_id}-{index:02d}" for index in range(len(rows))]
    for index, trajectory_id in enumerate(ids):
        store.submit(trajectory_id, task, expected_steps=1, group_id=group_id,
                     member=index, group_size=len(rows))
    for index, (trajectory_id, row) in enumerate(zip(ids, rows, strict=True)):
        lease = store.claim("execute", "fixed-replay-loader")
        generation = {
            "prompt_id": task.id,
            "prompt": row["prompt"],
            "response": row["output_text"],
            "logprob": float(row["behavior_logprob"]),
            "token_count": int(row["token_count"]),
            "latency_s": float(row["generation_latency_s"]),
            "metadata": row["generation_metadata"],
        }
        if generation_factory is not None:
            generation_factory(**generation)
        store.append_step(lease, AgentStep(
            index=0,
            event_id=f"{trajectory_id}-terminal",
            policy=policy,
            action={"kind": "answer"},
            observation={"source": "immutable-calibration-replay"},
            state={"member": index},
            generation=generation,
        ))
        store.finish(lease)
    for trajectory_id, row in zip(ids, rows, strict=True):
        lease = store.claim("verify", "independent-reward-oracle")
        trajectory = store.trajectory(trajectory_id)
        score = float(row[score_key])
        verification = Verification(
            score=score,
            passed=score > 0.0,
            confidence=1.0,
            verifier_version=version,
            verifier_name=verifier_name,
            trusted=True,
            metadata={
                "trajectory_sha256": trajectory["head"],
                "verifier_config_sha256": verifier_sha,
                "replay_source_sha256": sha256_file(
                    CALIBRATION_RUN / "groups" / f"{task.metadata['source_group_id']}.json"
                ),
            },
        )
        store.record_verification(
            lease, root.attest(trajectory_id, trajectory["head"], verification)
        )


def experiences_from_store(store, root, group_id: str):
    from vare.types import Attempt, Experience, Task, Verification

    experiences = []
    for row in store.group(group_id):
        steps = list(store.iter_steps(row["id"]))
        step = steps[-1]
        verification = Verification(**json.loads(row["verification"]))
        if not root.verifies(row["id"], row["head"], verification):
            raise RuntimeError("fixed replay reward receipt failed authentication")
        task = Task(**json.loads(row["task"])["task"])
        generation = step["generation"]
        metadata = {
            "vare_rollout_group": group_id,
            "vare_rollout_group_size": len(store.group(group_id)),
            "trajectory_id": row["id"],
            "trajectory_sha256": row["head"],
            "policy_checkpoint_sha256": step["policy"]["checkpoint_sha256"],
            "rvl_generation": generation,
        }
        attempt = Attempt(
            task=task,
            output=generation["response"],
            policy_id=step["policy"]["policy_id"],
            policy_version=step["policy"]["version"],
            created_step=0,
            logprob=float(generation["logprob"]),
            metadata=metadata,
        )
        experiences.append(Experience(
            attempt=attempt,
            verification=verification,
            policy_lag=0,
            verifier_lag=0,
            shift_score=0.0,
        ))
    return experiences


def parameter_vector(model):
    import torch

    return torch.nn.utils.parameters_to_vector(
        [parameter.detach() for parameter in model.parameters()]
    ).to(dtype=torch.float64, device="cpu")


def optimizer_steps(snapshot: dict[str, Any]) -> dict[str, Any]:
    states = snapshot["optimizer"]["state"].values()
    steps = [int(value["step"].item()) for value in states if "step" in value]
    return {
        "parameters_with_adam_state": len(steps),
        "distinct_step_counts": sorted(set(steps)),
        "maximum_step": max(steps, default=0),
    }


def train_candidate_branch(trainer, initial_state, initial_vector, experiences,
                           *, active_verifier_version: int,
                           reward_root, generation_types):
    from vare.config import LagConfig
    from vare.integrations.rvl_grpo import RVLGRPOHooks

    # Wrap the method only to retain its diagnostics. The pinned implementation
    # remains the sole code that computes gradients and calls optimizer.step().
    original_train_step = trainer.train_step
    replay_metrics = []
    def recording_train_step(samples, *, advantages=None):
        metrics = original_train_step(samples, advantages=advantages)
        replay_metrics.append(metrics)
        return metrics
    trainer.train_step = recording_train_step
    trainer.restore_training_state(initial_state)

    def authorize(experience) -> bool:
        metadata = experience.attempt.metadata
        return reward_root.verifies(
            metadata["trajectory_id"],
            metadata["trajectory_sha256"],
            experience.verification,
        )

    hooks = RVLGRPOHooks(
        backend=object(),
        trainer=trainer,
        eval_tasks=[],
        score_fn=lambda _task, _response: 0.0,
        generation_factory=generation_types.Generation,
        verified_generation_factory=generation_types.VerifiedGeneration,
        training_evidence_authorizer=authorize,
        require_authenticated_training=True,
        current_policy_version=0,
        current_verifier_version=active_verifier_version,
        freshness_lag_config=LagConfig(max_policy_lag=0, max_verifier_lag=0),
    )
    candidate_id = asyncio.run(hooks.train_candidate("policy-0", experiences))
    if len(replay_metrics) != 1:
        raise RuntimeError("expected exactly one real optimizer call")
    hooks._restore(candidate_id)
    candidate_state = trainer.snapshot_training_state()
    candidate_vector = parameter_vector(trainer.model)
    delta = candidate_vector - initial_vector
    metrics = {
        "candidate_id": candidate_id,
        "train_step_metrics": replay_metrics[0],
        "optimizer_state": optimizer_steps(candidate_state),
        "initial_model_sha256": sha_tree(initial_state["model"]),
        "candidate_model_sha256": sha_tree(candidate_state["model"]),
        "candidate_optimizer_sha256": sha_tree(candidate_state["optimizer"]),
        "parameter_delta_l2": float(delta.norm()),
        "candidate_state_restored_for_audit": sha_tree(trainer.snapshot_training_state())
            == sha_tree(candidate_state),
    }
    trainer.restore_training_state(initial_state)
    if sha_tree(trainer.snapshot_training_state()) != sha_tree(initial_state):
        raise RuntimeError("checkpoint reset failed between replay branches")
    trainer.train_step = original_train_step
    return metrics, delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT.parent / "tinystories-1m-pinned")
    parser.add_argument("--rvl-source", type=Path, default=ROOT.parent / "rvl-pinned")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()

    model_dir = args.model_dir.resolve()
    rvl_source = args.rvl_source.resolve()
    frozen = json.loads(CALIBRATION_PROTOCOL.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    if sha256_file(CALIBRATION_PROTOCOL) != protocol["source_data"]["calibration_protocol_sha256"]:
        raise RuntimeError("frozen calibration protocol hash mismatch")
    atomic_json(output / "protocol.json", protocol)
    rvl_revision = subprocess.check_output(
        ["git", "-C", str(rvl_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if rvl_revision != protocol["training"]["rvl_revision"]:
        raise RuntimeError("RVL source revision differs from replay protocol")
    model_files_ok = {
        name: sha256_file(model_dir / name) == expected
        for name, expected in frozen["model"]["files"].items()
    }
    if not all(model_files_ok.values()):
        raise RuntimeError("TinyStories checkpoint differs from frozen model manifest")
    relevant_rvl_files = [
        "src/rvl_systems/types.py",
        "src/rvl_systems/grpo.py",
        "src/rvl_systems/hf_trainer.py",
        "src/rvl_systems/triton_grpo.py",
    ]
    dirty_relevant = subprocess.run(
        ["git", "-C", str(rvl_source), "diff", "--quiet", "HEAD", "--", *relevant_rvl_files]
    ).returncode != 0
    if dirty_relevant:
        raise RuntimeError("a pinned RVL training source file has local modifications")

    sys.path.insert(0, str(rvl_source / "src"))
    from scripts.audit_verifier_reward_calibration import audit, independent_reward
    from scripts.audit_verifier_reward_calibration import digest as audit_digest
    from scripts.audit_verifier_reward_calibration import PROTOCOL as frozen_protocol_path
    from rvl_systems import types as generation_types
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from transformers import AutoModelForCausalLM
    import torch
    from vare.evidence import sha256_json
    from vare.trajectory_admission import TrajectoryRewardRoot, admit_terminal_grpo_group
    from vare.trajectory_store import PolicyCheckpoint, TrajectoryStore, audit_trajectory_store
    from vare.types import Task

    audit_result = audit(CALIBRATION_RUN)
    if audit_result["status"] != "PASS_INDEPENDENT_RECONSTRUCTION":
        raise RuntimeError("independent calibration reward audit failed")
    if audit_digest(CALIBRATION_PROTOCOL) != protocol["source_data"]["calibration_protocol_sha256"]:
        raise RuntimeError("independent audit protocol identity mismatch")
    groups = []
    for task in frozen["sampling"]["tasks"]:
        group = json.loads((CALIBRATION_RUN / "groups" / f"{task['id']}.json").read_text())
        expected_sha = protocol["source_data"]["groups"][task["id"]]
        if sha256_file(CALIBRATION_RUN / "groups" / f"{task['id']}.json") != expected_sha:
            raise RuntimeError(f"fixed source group hash mismatch: {task['id']}")
        for member in group["members"]:
            category, score_v0, score_v1 = independent_reward(
                member["output_text"], task["target_class"], frozen["reward_definition"]
            )
            if (category != member["surface_class"] or score_v0 != member["v0"]
                    or score_v1 != member["v1"]):
                raise RuntimeError("independent verifier recomputation differs from frozen scores")
        groups.append(group)

    verifier_shas = {}
    for version in ("v0", "v1"):
        verifier_shas[version] = sha256_json({
            "calibration_protocol_sha256": protocol["source_data"]["calibration_protocol_sha256"],
            "version": version,
            "reward_definition": frozen["reward_definition"],
        })
    run_id = "verifier-consistency-optimizer-replay-v1"
    protocol_sha = sha256_file(PROTOCOL)
    reward_root = TrajectoryRewardRoot(
        key=hashlib.sha256(b"VARE fixed replay fixture key; not a deployment secret").digest(),
        run_id=run_id,
        protocol_sha256=protocol_sha,
    )
    checkpoint_sha = canonical_sha(frozen["model"]["files"])
    policy = PolicyCheckpoint("policy-0", 0, checkpoint_sha)
    store = TrajectoryStore.create(output / "trajectories.sqlite", run_id=run_id,
                                   max_inflight=128)
    decisions = []
    try:
        store.register_policy(policy)
        for version in (0, 1):
            store.register_verifier("surface-compatibility", version,
                                    verifier_shas[f"v{version}"])
        for group in groups:
            task = Task(
                id=group["task_id"],
                prompt=group["prompt"],
                family=group["target_class"],
                metadata={
                    "learning_rule": "terminal_action_on_policy_grpo_v1",
                    "source_group_id": group["task_id"],
                    "target_class": group["target_class"],
                },
            )
            for version in (0, 1):
                group_id = f"{group['task_id']}-v{version}"
                write_group(
                    store, reward_root, policy, task, group["members"],
                    group_id=group_id, version=version,
                    verifier_name="surface-compatibility",
                    verifier_sha=verifier_shas[f"v{version}"],
                    score_key=f"v{version}", generation_factory=generation_types.Generation,
                )
                admission = admit_terminal_grpo_group(
                    store, group_id, active_policy=policy,
                    verifier_name="surface-compatibility",
                    verifier_version=version,
                    verifier_config_sha256=verifier_shas[f"v{version}"],
                    trust_root=reward_root,
                )
                if not admission.admitted or len(admission.experiences) != len(group["members"]):
                    raise RuntimeError(f"fresh {group_id} group was not admitted: {admission.reasons}")
                decisions.append({
                    "task_id": group["task_id"],
                    "group_id": group_id,
                    "verifier_version": version,
                    "admitted": admission.admitted,
                    "members": len(admission.experiences),
                    "authenticated_heads": admission.heads,
                })

        store_audit = audit_trajectory_store(output / "trajectories.sqlite")
        if store_audit.get("pass") is not True:
            raise RuntimeError("trajectory-store independent audit failed")

        torch.set_num_threads(1)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.float32
        )
        model.config.use_cache = False
        trainer = HFCausalLMGRPOTrainer(
            model,
            config=HFTTrainerConfig(
                learning_rate=1e-6,
                clip_eps=0.2,
                max_grad_norm=1.0,
                advantage_eps=1e-6,
                clip_advantage=5.0,
                disable_dropout=True,
                objective_backend="torch",
            ),
        )
        initial_state = trainer.snapshot_training_state()
        initial_vector = parameter_vector(model)
        initial_model_sha = sha_tree(initial_state["model"])
        initial_optimizer_sha = sha_tree(initial_state["optimizer"])
        replay_pairs = []
        for group in groups:
            task_id = group["task_id"]
            stale_v0_id = f"{task_id}-v0"
            fresh_v1_id = f"{task_id}-v1"
            stale_aware = admit_terminal_grpo_group(
                store, stale_v0_id, active_policy=policy,
                verifier_name="surface-compatibility", verifier_version=1,
                verifier_config_sha256=verifier_shas["v1"], trust_root=reward_root,
            )
            if stale_aware.admitted or "verifier_stale_or_mismatched" not in stale_aware.reasons:
                raise RuntimeError("version-aware admission failed to reject stale v0 group")
            stale_experiences = experiences_from_store(store, reward_root, stale_v0_id)
            fresh_experiences = admit_terminal_grpo_group(
                store, fresh_v1_id, active_policy=policy,
                verifier_name="surface-compatibility", verifier_version=1,
                verifier_config_sha256=verifier_shas["v1"], trust_root=reward_root,
            ).experiences
            if not fresh_experiences:
                raise RuntimeError("version-aware admission rejected freshly verified v1 group")

            aware_hook = None
            from vare.config import LagConfig
            from vare.integrations.rvl_grpo import RVLGRPOHooks
            def authorize(experience) -> bool:
                metadata = experience.attempt.metadata
                return reward_root.verifies(
                    metadata["trajectory_id"], metadata["trajectory_sha256"],
                    experience.verification,
                )
            aware_hook = RVLGRPOHooks(
                backend=object(), trainer=trainer, eval_tasks=[],
                score_fn=lambda _task, _response: 0.0,
                generation_factory=generation_types.Generation,
                verified_generation_factory=generation_types.VerifiedGeneration,
                training_evidence_authorizer=authorize,
                require_authenticated_training=True,
                current_policy_version=0, current_verifier_version=1,
                freshness_lag_config=LagConfig(max_policy_lag=0, max_verifier_lag=0),
            )
            before_reject = sha_tree(trainer.snapshot_training_state())
            rejected = False
            try:
                asyncio.run(aware_hook.train_candidate("policy-0", stale_experiences))
            except RuntimeError as exc:
                rejected = str(exc) == "stale_training_experience"
                if not rejected:
                    raise
            if not rejected or sha_tree(trainer.snapshot_training_state()) != before_reject:
                raise RuntimeError("stale group changed real trainer state")
            if getattr(trainer, "replay_metrics", None):
                raise RuntimeError("stale group reached the optimizer boundary")

            aware_update, aware_vector = train_candidate_branch(
                trainer, initial_state, initial_vector, list(fresh_experiences),
                active_verifier_version=1,
                reward_root=reward_root,
                generation_types=generation_types,
            )
            naive_experiences = admit_terminal_grpo_group(
                store, stale_v0_id, active_policy=policy,
                verifier_name="surface-compatibility", verifier_version=0,
                verifier_config_sha256=verifier_shas["v0"], trust_root=reward_root,
            ).experiences
            if not naive_experiences:
                raise RuntimeError("version-unaware control failed to admit its current v0 group")
            naive_update, naive_vector = train_candidate_branch(
                trainer, initial_state, initial_vector, list(naive_experiences),
                active_verifier_version=0,
                reward_root=reward_root,
                generation_types=generation_types,
            )
            vector_diff = naive_vector - aware_vector
            cosine_denominator = float(naive_vector.norm() * aware_vector.norm())
            preference_reversals = sum(
                (float(left["v0"]) - float(right["v0"]))
                * (float(left["v1"]) - float(right["v1"])) < 0.0
                for index, left in enumerate(group["members"])
                for right in group["members"][index + 1:]
            )
            replay_pairs.append({
                "task_id": task_id,
                "reward_values_changed": sum(
                    float(member["v0"]) != float(member["v1"])
                    for member in group["members"]
                ),
                "strict_pairwise_preference_reversals": preference_reversals,
                "version_aware": {
                    "stale_v0_rejected_at_admission": True,
                    "stale_v0_rejected_at_trainer_boundary": True,
                    "stale_rejection_optimizer_calls": 0,
                    "fresh_v1_update": aware_update,
                },
                "version_unaware": {
                    "v0_group_admitted": True,
                    "v0_update": naive_update,
                },
                "candidate_update_comparison": {
                    "parameter_delta_cosine_similarity": float(
                        torch.dot(naive_vector, aware_vector) / cosine_denominator
                    ) if cosine_denominator else None,
                    "parameter_delta_difference_l2": float(vector_diff.norm()),
                    "candidate_model_hashes_differ": (
                        naive_update["candidate_model_sha256"] != aware_update["candidate_model_sha256"]
                    ),
                    "candidate_optimizer_hashes_differ": (
                        naive_update["candidate_optimizer_sha256"] != aware_update["candidate_optimizer_sha256"]
                    ),
                },
            })
            del aware_vector, naive_vector, vector_diff

        summary = {
            "schema": "vare.verifier-consistency-optimizer-replay-result.v1",
            "status": "PASS_FIXED_REPLAY_SYSTEMS_REFERENCE",
            "interpretation_limit": "Replay-only systems result. No fresh rollouts, active-policy promotions, capability claim, or async comparison.",
            "audit": {
                "frozen_calibration_reward_audit": audit_result,
                "trajectory_store_audit": store_audit,
                "all_receipts_authenticated": True,
                "all_rewards_independently_recomputed": True,
            },
            "admission_decisions": decisions,
            "optimizer_replay_pairs": replay_pairs,
            "totals": {
                "fixed_groups": len(groups),
                "members_replayed": sum(len(group["members"]) for group in groups),
                "stale_groups_rejected": len(groups),
                "version_unaware_optimizer_steps": len(groups),
                "version_aware_optimizer_steps": len(groups),
                "stale_rejection_optimizer_steps": 0,
                "model_promotions": 0,
                "model_checkpoint_sha256": checkpoint_sha,
                "initial_model_sha256": initial_model_sha,
                "initial_optimizer_sha256": initial_optimizer_sha,
            },
            "identities": {
                "protocol_sha256": protocol_sha,
                "calibration_protocol_sha256": sha256_file(CALIBRATION_PROTOCOL),
                "model_revision": frozen["model"]["revision"],
                "model_files_sha256": model_files_ok,
                "rvl_revision": rvl_revision,
                "rvl_source_sha256": {path: sha256_file(rvl_source / path) for path in relevant_rvl_files},
                "vare_source_sha256": {
                    path: sha256_file(ROOT / path)
                    for path in (
                        "src/vare/integrations/rvl_grpo.py",
                        "src/vare/trajectory_admission.py",
                        "src/vare/trajectory_store.py",
                    )
                },
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "independent_reward_auditor_sha256": sha256_file(frozen_protocol_path.parent.parent.parent / "scripts/audit_verifier_reward_calibration.py"),
            },
            "resource_use": {
                "device": "cpu",
                "precision": "fp32",
                "torch_threads": 1,
                "wall_seconds": round(time.monotonic() - start, 6),
            },
        }
        atomic_json(output / "summary.json", summary)
    except BaseException as exc:
        failure = {
            "schema": "vare.verifier-consistency-optimizer-replay-failure.v1",
            "status": "FAILED",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "protocol_sha256": sha256_file(PROTOCOL),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "partial_summary_present": (output / "summary.json").is_file(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            atomic_json(output / "failure.json", failure)
        except FileExistsError:
            pass
        raise

    inventory = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    atomic_json(output / "SHA256.json", inventory)
    if json.loads((output / "SHA256.json").read_text()) != inventory:
        raise RuntimeError("run inventory read-back mismatch")
    print(json.dumps({
        "output": str(output),
        "status": summary["status"],
        "summary_sha256": sha256_file(output / "summary.json"),
        "inventory_sha256": sha256_file(output / "SHA256.json"),
        "optimizer_steps": summary["totals"]["version_unaware_optimizer_steps"]
            + summary["totals"]["version_aware_optimizer_steps"],
    }, indent=2))


if __name__ == "__main__":
    main()
