"""Bridge parent-graded model completions through VARE's durable RL path.

The store records one generation and its trusted verification per trajectory.
Only a complete, homogeneous group is admitted to replay, and only replayed
fresh experiences are converted back to GRPO samples. Consumer acknowledgment
is intentionally separate from the learner's durable update acknowledgment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from vare.config import ReplayConfig
from vare.grpo_mlx import Sample, group_samples
from vare.replay import PrioritizedReplay
from vare.trajectory_store import AgentStep, PolicyCheckpoint, TrajectoryStore
from vare.types import Attempt, Experience, Task, Verification


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to replace trajectory acknowledgment: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class VerifiedGroupReplay:
    """Persist, verify, admit, freshness-check, and reconstruct whole groups."""

    def __init__(self, run_dir: Path, *, run_id: str, group_budget: int,
                 group_size: int, replay_seed: int) -> None:
        if type(group_budget) is not int or group_budget <= 0:
            raise ValueError("group_budget must be positive")
        if type(group_size) is not int or group_size <= 0:
            raise ValueError("group_size must be positive")
        self.path = run_dir / "trajectory-store.sqlite3"
        if self.path.exists():
            self.store = TrajectoryStore(self.path)
            config = self.store.config()
            if config.get("run_id") != run_id:
                raise RuntimeError("trajectory store belongs to another experiment")
        else:
            self.store = TrajectoryStore.create(
                self.path, run_id=run_id, max_inflight=max(group_budget * group_size, group_size),
                record_bytes=1024**2, storage_bytes=128 * 1024**2,
            )
        self.group_size = group_size
        self.replay = PrioritizedReplay(
            ReplayConfig(capacity=group_budget * group_size), seed=replay_seed,
        )
        self.groups: list[dict[str, Any]] = []

    def add_group(
        self,
        *,
        task: Any,
        family: str,
        prompt_messages: Sequence[Mapping[str, str]],
        samples: Sequence[Sample],
        rows: Sequence[Mapping[str, Any]],
        policy_id: str,
        policy_version: int,
        policy_checkpoint_sha256: str,
        group_id: str,
    ) -> dict[str, Any]:
        if len(samples) != self.group_size or len(rows) != self.group_size:
            raise ValueError("only complete rollout groups may enter trajectory storage")
        if sorted(sample.sample_index for sample in samples) != list(range(self.group_size)):
            raise ValueError("rollout group must contain each declared member exactly once")
        if any(sample.instance_id != task.task_id or sample.behavior_checkpoint_id != policy_id
               or sample.behavior_step != policy_version for sample in samples):
            raise ValueError("rollout group policy or task identity mismatch")

        # Verify the entire externally supplied group *before* registering any
        # trajectory IDs, policies or verifiers. Otherwise one malformed row
        # can strand partially enrolled durable trajectories and block retry.
        group_samples(samples, group_size=self.group_size)
        prompt_json = _json(list(prompt_messages))
        task_digest = task.sha256()
        for sample, row in zip(samples, rows, strict=True):
            if (row.get("instance_id") != task.task_id
                    or int(row.get("sample_index", -1)) != sample.sample_index):
                raise ValueError("sample and persisted row identities differ")
            probabilities = row.get("sampler_logprobs")
            if (not isinstance(probabilities, (list, tuple))
                    or len(probabilities) != len(sample.completion_ids)
                    or not all(type(value) in (int, float) and math.isfinite(value)
                               for value in probabilities)):
                raise ValueError("sampler log-probabilities must be finite and token-aligned")
            detail = row.get("reward_detail")
            if not isinstance(detail, Mapping) or detail.get("trusted_contract") is not True:
                raise ValueError("untrusted reward cannot enter VARE replay")
            if type(detail.get("passed")) is not bool:
                raise ValueError("trusted outcome must be boolean")
            if (type(sample.reward) not in (int, float)
                    or not math.isfinite(sample.reward)
                    or sample.reward not in (0.0, 1.0)
                    or sample.reward != float(detail["passed"])):
                raise ValueError("binary trusted result and GRPO reward disagree")
            observed_reward = row.get("reward")
            if (type(observed_reward) not in (int, float)
                    or not math.isfinite(observed_reward)
                    or observed_reward != sample.reward):
                raise ValueError("persisted reward and sample reward disagree")
            if not isinstance(row.get("response_text"), str) or type(row.get("seed")) is not int:
                raise ValueError("generation evidence is missing a response or integer seed")
            for name in ("generation_seconds", "sample_generation_seconds", "temperature"):
                if name == "sample_generation_seconds" and name not in row:
                    continue
                if name == "temperature" and name not in row:
                    continue
                value = row.get(name)
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or (name == "temperature" and value <= 0)
                        or (name != "temperature" and value < 0)):
                    raise ValueError(f"invalid generation evidence: {name}")

        verifier_id = samples[0].verifier
        verifier_config_sha256 = verifier_id
        if any(sample.verifier != verifier_id for sample in samples):
            raise ValueError("rollout group mixes verifier identities")
        policy = PolicyCheckpoint(policy_id, policy_version, policy_checkpoint_sha256)
        self.store.register_policy(policy)
        self.store.register_verifier(verifier_id, 0, verifier_config_sha256)

        replay_task = Task(
            id=task.task_id,
            prompt=prompt_json,
            family=family,
            metadata={
                "task_sha256": task_digest,
                "policy_checkpoint_sha256": policy_checkpoint_sha256,
                "verifier_config_sha256": verifier_config_sha256,
                "vare_rollout_group": group_id,
                "vare_rollout_group_size": self.group_size,
            },
        )

        sample_by_id: dict[str, tuple[Sample, Mapping[str, Any]]] = {}
        for sample, row in zip(samples, rows, strict=True):
            if row.get("instance_id") != task.task_id or int(row.get("sample_index", -1)) != sample.sample_index:
                raise ValueError("sample and persisted row identities differ")
            trajectory_id = f"{group_id}:member:{sample.sample_index}"
            inserted = self.store.submit(
                trajectory_id,
                replay_task,
                expected_steps=1,
                group_id=group_id,
                member=sample.sample_index,
                group_size=self.group_size,
            )
            if not inserted:
                raise RuntimeError("a prospective rollout group cannot be enrolled twice")
            sample_by_id[trajectory_id] = (sample, row)

        while sample_by_id:
            lease = self.store.claim("execute", "vare-ig-exec", lease_seconds=30)
            if lease is None or lease.trajectory_id not in sample_by_id:
                raise RuntimeError("trajectory store did not claim an enrolled group member")
            sample, row = sample_by_id[lease.trajectory_id]
            sampler_logprobs = [float(value) for value in row["sampler_logprobs"]]
            if len(sampler_logprobs) != len(sample.completion_ids) or not sample.completion_ids:
                raise ValueError("generation tokens and exact sampler log-probabilities must align")
            reward_detail = dict(row["reward_detail"])
            if reward_detail.get("trusted_contract") is not True:
                raise ValueError("untrusted reward cannot enter VARE replay")
            if type(reward_detail.get("passed")) is not bool or \
                    float(sample.reward) != float(bool(reward_detail["passed"])):
                raise ValueError("binary trusted result and GRPO reward disagree")
            generation = {
                "prompt_id": _sha({"messages": prompt_messages, "tokens": sample.prompt_ids}),
                "prompt": prompt_json,
                "response": str(row["response_text"]),
                "logprob": math.fsum(sampler_logprobs),
                "token_count": len(sample.completion_ids),
                "latency_s": float(row.get("sample_generation_seconds", row["generation_seconds"])),
                "metadata": {
                    "prompt_token_ids": list(sample.prompt_ids),
                    "response_token_ids": list(sample.completion_ids),
                    "response_token_logprobs": sampler_logprobs,
                    "temperature": float(row.get("temperature", 1.0)),
                    "seed": int(row["seed"]),
                    "sampling_distribution": row.get("sampling_distribution"),
                },
            }
            state = {
                "task_sha256": task.sha256(),
                "sample_index": sample.sample_index,
                "seed": int(row["seed"]),
                "phase": row.get("phase"),
                "generation_cost_attributed_seconds": float(row["generation_seconds"]),
                "finish": row.get("finish"),
                "candidate_source_sha256": row.get("candidate_source_sha256"),
                "reward_detail": reward_detail,
                "inference_profile": row.get("inference_profile"),
            }
            agent_step = AgentStep(
                index=0,
                event_id=f"completion-{sample.sample_index}",
                policy=policy,
                action={"kind": "answer", "response_sha256": hashlib.sha256(
                    str(row["response_text"]).encode("utf-8")).hexdigest()},
                observation={"reward": float(sample.reward), "verifier_id": verifier_id},
                state=state,
                generation=generation,
            )
            self.store.append_step(lease, agent_step, return_head=True)
            self.store.finish(lease)
            del sample_by_id[lease.trajectory_id]

        for _ in range(self.group_size):
            lease = self.store.claim("verify", "vare-ig-verifier", lease_seconds=30)
            if lease is None:
                raise RuntimeError("completed rollout lacks a verification lease")
            # Resolve the completion by the durable trajectory identity; the
            # sample order is independent of lease order.
            trajectory_id = lease.trajectory_id
            if not trajectory_id.startswith(group_id + ":member:"):
                raise RuntimeError("verification lease escaped the current rollout group")
            sample_index = int(trajectory_id.rsplit(":", 1)[-1])
            original = next((item for item in rows if int(item["sample_index"]) == sample_index), None)
            if original is None:
                raise RuntimeError("verification lease has no matching generated row")
            detail = dict(original["reward_detail"])
            trajectory = self.store.trajectory(trajectory_id)
            verification = Verification(
                score=float(original["reward"]),
                passed=bool(detail.get("passed")),
                confidence=1.0,
                verifier_version=0,
                verifier_name=verifier_id,
                trusted=True,
                metadata={
                    "trajectory_sha256": trajectory["head"],
                    "verifier_config_sha256": verifier_config_sha256,
                    "verifier_id": verifier_id,
                    "task_sha256": task.sha256(),
                    "grade_status": detail.get("grade_status"),
                    "case_results": detail.get("cases", []),
                    "case_set_sha256": detail.get("case_set_sha256"),
                    "candidate_source_sha256": original.get("candidate_source_sha256"),
                },
            )
            self.store.record_verification(lease, verification)

        heads: dict[str, str] = {}
        experiences: list[Experience] = []
        for trajectory in self.store.group(group_id):
            if trajectory["state"] != "verified":
                raise RuntimeError("trajectory group did not reach verified state")
            trajectory_id = trajectory["id"]
            heads[trajectory_id] = trajectory["head"]
            step = next(self.store.iter_steps(trajectory_id))
            generation = step["generation"]
            verification = Verification(**json.loads(trajectory["verification"]))
            state = step["state"]
            replay_attempt = Attempt(
                task=replay_task,
                output=generation["response"],
                policy_id=step["policy"]["policy_id"],
                policy_version=int(step["policy"]["version"]),
                created_step=int(state["sample_index"]),
                logprob=float(generation["logprob"]),
                latency_ms=float(generation["latency_s"]) * 1000,
                metadata={
                    "vare_rollout_group": group_id,
                    "vare_rollout_group_size": self.group_size,
                    "policy_checkpoint_sha256": policy_checkpoint_sha256,
                    "verifier_config_sha256": verifier_config_sha256,
                    "trajectory_id": trajectory_id,
                    "trajectory_sha256": trajectory["head"],
                    "sample_index": int(state["sample_index"]),
                    "prompt_token_ids": generation["metadata"]["prompt_token_ids"],
                    "completion_token_ids": generation["metadata"]["response_token_ids"],
                    "sampler_logprobs": generation["metadata"]["response_token_logprobs"],
                    "seed": state["seed"],
                    "phase": state["phase"],
                    "temperature": generation["metadata"]["temperature"],
                    "sampling_distribution": generation["metadata"]["sampling_distribution"],
                    "finish": state["finish"],
                    "task_sha256": state["task_sha256"],
                    "candidate_source_sha256": state["candidate_source_sha256"],
                    "reward_detail": state["reward_detail"],
                    "inference_profile": state.get("inference_profile"),
                },
            )
            if not verification.trusted:
                raise RuntimeError("untrusted persisted verification cannot enter replay")
            experiences.append(Experience(
                attempt=replay_attempt,
                verification=verification,
                policy_lag=0,
                verifier_lag=0,
                shift_score=0.0,
            ))

        if not self.replay.add_group(experiences, [1.0] * self.group_size):
            raise RuntimeError("group-aware replay rejected a trusted homogeneous group")
        record = {
            "group_id": group_id,
            "task_id": task.task_id,
            "trajectory_heads": heads,
            "policy_id": policy_id,
            "policy_version": policy_version,
            "policy_checkpoint_sha256": policy_checkpoint_sha256,
            "verifier_id": verifier_id,
            "verifier_config_sha256": verifier_config_sha256,
            "group_size": self.group_size,
            "phase": rows[0].get("phase"),
            "replay_admitted": True,
            "freshness_checked_at_policy_version": policy_version,
        }
        self.groups.append(record)
        return record

    def sample_current(self, *, count: int, policy_id: str, policy_version: int,
                       policy_checkpoint_sha256: str) -> tuple[list[Sample], list[dict[str, Any]]]:
        def freshness(experience: Experience) -> float | None:
            attempt = experience.attempt
            verification = experience.verification
            metadata = attempt.metadata
            config_sha256 = verification.metadata.get("verifier_config_sha256")
            lag = policy_version - attempt.policy_version
            if (
                not verification.trusted
                or verification.verifier_version != 0
                or attempt.policy_id != policy_id
                or lag != 0
                or metadata.get("policy_checkpoint_sha256") != policy_checkpoint_sha256
                or metadata.get("verifier_config_sha256") != config_sha256
                or verification.verifier_name != metadata.get("verifier_config_sha256")
                or verification.metadata.get("trajectory_sha256") != metadata.get("trajectory_sha256")
            ):
                return None
            return 1.0

        selected = self.replay.sample_current(count, freshness_fn=freshness, grouped=True)
        if len(selected) != count:
            raise RuntimeError("freshness filtering rejected or lost a rollout group")
        samples: list[Sample] = []
        rows: list[dict[str, Any]] = []
        for experience in selected:
            attempt = experience.attempt
            metadata = attempt.metadata
            sample = Sample(
                instance_id=attempt.task.id,
                behavior_checkpoint_id=attempt.policy_id,
                behavior_step=attempt.policy_version,
                prompt_ids=list(metadata["prompt_token_ids"]),
                completion_ids=list(metadata["completion_token_ids"]),
                reward=float(experience.verification.score),
                verifier=experience.verification.verifier_name,
                sample_index=int(metadata["sample_index"]),
                meta={
                    "task_sha256": metadata["task_sha256"],
                    "phase": metadata["phase"],
                    "trajectory_id": metadata["trajectory_id"],
                    "trajectory_sha256": metadata["trajectory_sha256"],
                    "vare_rollout_group": metadata["vare_rollout_group"],
                    "policy_checkpoint_sha256": metadata["policy_checkpoint_sha256"],
                    "verifier_config_sha256": metadata["verifier_config_sha256"],
                },
            )
            samples.append(sample)
            rows.append({
                "instance_id": attempt.task.id,
                "behavior_checkpoint_id": attempt.policy_id,
                "behavior_step": attempt.policy_version,
                "prompt_ids": sample.prompt_ids,
                "completion_ids": sample.completion_ids,
                "reward": sample.reward,
                "verifier": sample.verifier,
                "sample_index": sample.sample_index,
                "behavior_logprobs": None,
                "task_sha256": metadata["task_sha256"],
                "phase": metadata["phase"],
                "response_text": attempt.output,
                "sampler_logprobs": metadata["sampler_logprobs"],
                "candidate_source_sha256": metadata["candidate_source_sha256"],
                "reward_detail": metadata["reward_detail"],
                "seed": metadata["seed"],
                "finish": metadata["finish"],
                "generation_seconds": attempt.latency_ms / 1000.0 if attempt.latency_ms is not None else 0.0,
                "sampling_distribution": metadata["sampling_distribution"],
                "temperature": metadata["temperature"],
            })
        group_samples(samples, group_size=self.group_size)
        return samples, rows


def acknowledge_durable_decision(run_dir: Path, *, update_step: int,
                                 acknowledgment_path: Path, decision: Mapping[str, Any]) -> dict[str, Any] | None:
    """Close verified consumer groups only after a durable learner decision."""
    store_path = run_dir / "trajectory-store.sqlite3"
    if not store_path.is_file():
        return None
    selection_paths = (
        run_dir / "scheduler-selection.json",
        run_dir / "recovery-trajectory-replay.json",
    )
    records: list[dict[str, Any]] = []
    for path in selection_paths:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.extend(payload.get("trajectory_groups", []))
    groups = [record for record in records if int(record["policy_version"]) == update_step - 1
              and (update_step == 1 or record.get("phase") == f"train-step-{update_step - 1}")]
    if update_step == 0:
        groups = [record for record in records if record.get("phase") == "train-step-0"]
    if not groups:
        return None

    ack_sha256 = hashlib.sha256(acknowledgment_path.read_bytes()).hexdigest()
    acknowledgment_record = run_dir / f"trajectory-consumption-{update_step:04d}.json"
    if acknowledgment_record.is_file():
        existing = json.loads(acknowledgment_record.read_text(encoding="utf-8"))
        expected = {record["group_id"]: record["trajectory_heads"] for record in groups}
        observed = {record["group_id"]: record["heads"] for record in existing.get("groups", [])}
        if (
            existing.get("update_step") != update_step
            or existing.get("acknowledgment_sha256") != ack_sha256
            or observed != expected
        ):
            raise RuntimeError("trajectory consumption record conflicts with the durable learner acknowledgment")
        return existing

    store = TrajectoryStore(store_path)
    group_states = []
    for record in groups:
        members = store.group(record["group_id"])
        heads = {row["id"]: row["head"] for row in members}
        if len(members) != int(record["group_size"]) or heads != record["trajectory_heads"]:
            raise RuntimeError("durable rollout heads differ from the recorded group enrollment")
        states = {row["state"] for row in members}
        if states == {"verified"}:
            store.close_group(
                record["group_id"], heads, reason="consumed",
                decision={
                    "update_step": update_step,
                    "acknowledgment_path": acknowledgment_path.name,
                    "acknowledgment_sha256": ack_sha256,
                    "learner_decision": dict(decision),
                },
            )
            state = "consumed_now"
        elif states == {"consumed"}:
            state = "consumed_before_reconciliation"
        else:
            raise RuntimeError(f"cannot acknowledge incomplete rollout group: {record['group_id']} {states}")
        group_states.append({"group_id": record["group_id"], "state": state, "heads": heads})

    report = {
        "schema": "vare_ig_trajectory_consumer_ack_v1",
        "update_step": update_step,
        "acknowledgment_path": acknowledgment_path.name,
        "acknowledgment_sha256": ack_sha256,
        "groups": group_states,
        "optimizer_commit_claim": bool(decision.get("update_applied")),
        "claim_boundary": "trajectory consumer acknowledgment follows the durable learner decision; it is not the optimizer transaction itself",
    }
    _write_once(acknowledgment_record, report)
    return report
