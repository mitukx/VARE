"""Authenticated whole-group admission for terminal-action on-policy GRPO.

This module produces existing Experience values; it does not optimize or claim
that a single ratio corrects a multi-policy trajectory. Other algorithms and
multi-policy paths fail closed. Model adapters must verify actual weight IDs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import hmac
import json
import math

from .evidence import canonical_json, sha256_json
from .trajectory_store import PolicyCheckpoint, TrajectoryStore, identity, sha
from .types import Attempt, Experience, Task, Verification, snapshot_verification


@dataclass(frozen=True, repr=False, slots=True)
class TrajectoryRewardRoot:
    """Operator-owned signer; do not expose its key to agent/tool workers.

    The MAC establishes origin under this trust root, not verifier accuracy.
    The complete trace head already binds task, group, steps and checkpoints.
    """
    key: bytes
    run_id: str
    protocol_sha256: str

    def __post_init__(self):
        if type(self.key) is not bytes or len(self.key) < 32:
            raise ValueError("reward key must contain at least 32 bytes")
        identity(self.run_id)
        sha(self.protocol_sha256)

    def _mac(self, trajectory_id, head, verification):
        body = asdict(snapshot_verification(verification))
        body["metadata"].pop("trajectory_reward_receipt", None)
        if body["metadata"].get("trajectory_sha256") != sha(head):
            raise ValueError("reward must bind the complete trajectory")
        payload = {"domain": "vare.trajectory-reward.v1", "run_id": self.run_id,
                   "protocol_sha256": self.protocol_sha256, "trajectory_id": identity(trajectory_id),
                   "head": head, "verification": body}
        return hmac.new(self.key, canonical_json(payload).encode(), hashlib.sha256).hexdigest()

    def attest(self, trajectory_id, head, verification):
        value = snapshot_verification(verification)
        if "trajectory_reward_receipt" in value.metadata:
            raise ValueError("reward already has a receipt")
        return replace(value, metadata={**value.metadata, "trajectory_reward_receipt": self._mac(trajectory_id, head, value)})

    def verifies(self, trajectory_id, head, verification):
        try:
            receipt = verification.metadata["trajectory_reward_receipt"]
            sha(receipt)
            return hmac.compare_digest(receipt, self._mac(trajectory_id, head, verification))
        except (ValueError, TypeError, KeyError, AttributeError):
            return False


@dataclass(frozen=True, slots=True)
class TrajectoryAdmission:
    group_id: str
    admitted: bool
    reasons: tuple[str, ...]
    heads: dict[str, str]
    experiences: tuple[Experience, ...]
    policy_lags: tuple[int, ...]
    verifier_lags: tuple[int, ...]


def admit_terminal_grpo_group(
    store: TrajectoryStore, group_id: str, *, active_policy: PolicyCheckpoint,
    verifier_name: str, verifier_version: int, verifier_config_sha256: str,
    trust_root: TrajectoryRewardRoot,
) -> TrajectoryAdmission:
    """At most 1024 bounded members. Recheck at the actual trainer boundary too.

    Only the terminal inference receives the trajectory reward. Intermediate
    tool/inference tokens are preserved but not implicitly assigned that reward.
    The task's algorithm declaration must explicitly opt into this attribution.
    """
    rows = store.group(group_id)
    heads = {row["id"]: row["head"] for row in rows}
    reasons, experiences, policy_lags, verifier_lags = set(), [], [], []
    if (not rows or len(rows) != rows[0]["group_size"] or len(rows) < 2
            or [row["member"] for row in rows] != list(range(len(rows)))):
        return TrajectoryAdmission(group_id, False, ("incomplete_group",), heads, (), (), ())
    if any(row["state"] != "verified" for row in rows):
        return TrajectoryAdmission(group_id, False, ("unverified_group",), heads, (), (), ())
    if store.config()["run_id"] != trust_root.run_id:
        reasons.add("foreign_run")
    for row in rows:
        spec = json.loads(row["task"])
        task = Task(**spec["task"])
        if task.metadata.get("learning_rule") != "terminal_action_on_policy_grpo_v1":
            reasons.add("unsupported_reward_attribution")
        head = sha256_json({"trajectory_id": row["id"], "spec": spec})
        policies, last, count = set(), None, 0
        for item in store.iter_steps(row["id"]):
            if item["index"] != count:
                reasons.add("step_gap")
            head = sha256_json({"previous": head, "step_sha256": sha256_json(item)})
            policies.add((item["policy"]["policy_id"], item["policy"]["version"], item["policy"]["checkpoint_sha256"]))
            last, count = item, count + 1
        if head != row["head"] or count != spec["expected_steps"]:
            reasons.add("incomplete_or_corrupt_trace")
        if len(policies) != 1:
            reasons.add("multi_policy_trajectory")
        for policy_id, version, checkpoint in policies:
            policy_lags.append(active_policy.version - version)
            if (policy_id, version, checkpoint) != (active_policy.policy_id, active_policy.version, active_policy.checkpoint_sha256):
                reasons.add("off_policy_checkpoint")
        verification = snapshot_verification(Verification(**json.loads(row["verification"])))
        verifier_lags.append(verifier_version - verification.verifier_version)
        if (verification.verifier_name != verifier_name or verification.verifier_version != verifier_version
                or verification.metadata.get("verifier_config_sha256") != verifier_config_sha256):
            reasons.add("verifier_stale_or_mismatched")
        if not verification.trusted or not trust_root.verifies(row["id"], row["head"], verification):
            reasons.add("unauthenticated_reward")
        generation = last["generation"] if last else None
        if (last is None or generation is None or last["action"]["kind"] not in {"answer", "inference"}
                or not generation["metadata"].get("response_token_ids")
                or not generation["metadata"].get("prompt_token_ids")
                or generation["metadata"].get("response_token_logprobs") is None):
            reasons.add("missing_terminal_behavior")
            continue
        if generation["prompt_id"] != task.id:
            reasons.add("terminal_task_mismatch")
        behavior = generation["metadata"]
        temperature = behavior.get("sampling_temperature")
        if (type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0
                or behavior.get("logprob_distribution") != "neutral_temperature_scaled_policy"
                or not isinstance(behavior.get("model"), str) or not behavior["model"]):
            reasons.add("unknown_behavior_distribution")
        # Store both original task identity and the actual terminal model input.
        metadata = {**task.metadata, "vare_rollout_group": group_id,
                    "vare_rollout_group_size": len(rows), "trajectory_id": row["id"],
                    "trajectory_sha256": row["head"], "source_task_sha256": sha256_json(spec["task"])}
        training_task = replace(task, prompt=generation["prompt"], metadata=metadata)
        attempt = Attempt(training_task, generation["response"], active_policy.policy_id,
                          active_policy.version, 0, metadata={**metadata, "rvl_generation": generation,
                                                             "policy_checkpoint_sha256": active_policy.checkpoint_sha256})
        experiences.append(Experience(attempt, verification, 0, 0, 0.))
    # Reject heterogeneous terminal prompts; one GRPO group means one model input.
    if len({exp.attempt.task.prompt for exp in experiences}) > 1:
        reasons.add("heterogeneous_group_prompt")
    if len({tuple(exp.attempt.metadata["rvl_generation"]["metadata"]["prompt_token_ids"]) for exp in experiences}) > 1:
        reasons.add("heterogeneous_group_prompt_tokens")
    identities = set()
    for exp in experiences:
        behavior = exp.attempt.metadata["rvl_generation"]["metadata"]
        identities.add(canonical_json({key: behavior.get(key) for key in ("backend", "model", "sampling_temperature", "logprob_distribution")}))
    if len(identities) > 1:
        reasons.add("heterogeneous_behavior_distribution")
    admitted = not reasons and len(experiences) == len(rows)
    return TrajectoryAdmission(group_id, admitted, tuple(sorted(reasons)), heads,
                               tuple(experiences) if admitted else (), tuple(policy_lags), tuple(verifier_lags))
