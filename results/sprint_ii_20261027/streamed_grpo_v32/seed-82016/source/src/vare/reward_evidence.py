"""Operator-attested RLVR reward receipts for RVL replay curriculum admission.

RVL's SQLite behavior hash is not keyed, and its metadata 'trusted' bit is a
claim supplied with the reward. Neither constitutes independent adjudication.
This opt-in contract binds one reward to the *exact* scored generation, frozen
grader/protocol identities, source group, policy/verifier versions, and run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import re
from collections.abc import Sequence
from typing import Mapping

from .evidence import canonical_json
from .types import Experience


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_DOMAIN = "vare.rvl-reward-attestation.v1"
_SUPPORTED_BEHAVIOR = frozenset({"generation_match", "verified_record_match"})


def validate_complete_rvl_reward_batch(experiences: Sequence[Experience]) -> None:
    """Reject partial or duplicated replay-backed RVL members at use time.

    Reward receipts are member-scoped, so verifying each receipt separately is
    insufficient: the same valid member could otherwise be repeated to fill a
    group. Live RVL rollouts also carry ``rvl_generation`` but do not have
    replay member indices yet; those remain governed by their authorizer.
    """
    attempt_metadata = [experience.attempt.metadata for experience in experiences]
    marked = [
        isinstance(metadata, dict)
        and ("rvl_group_id" in metadata or "rvl_generation_index" in metadata)
        for metadata in attempt_metadata
    ]
    if not any(marked):
        return
    if not all(marked):
        raise ValueError("authenticated batch cannot mix RVL and non-RVL experiences")

    grouped: dict[str, tuple[int, set[int]]] = {}
    for experience, metadata in zip(experiences, attempt_metadata, strict=True):
        if not isinstance(metadata, dict):
            raise ValueError("authenticated RVL experience metadata must be an object")
        task_metadata = experience.attempt.task.metadata
        group_id = metadata.get("rvl_group_id")
        rollout_group_id = metadata.get("vare_rollout_group")
        group_size = metadata.get("vare_rollout_group_size")
        member_index = metadata.get("rvl_generation_index")
        if (
            not isinstance(group_id, str)
            or not group_id
            or rollout_group_id != group_id
            or type(group_size) is not int
            or group_size <= 0
            or type(member_index) is not int
            or not 0 <= member_index < group_size
            or not isinstance(metadata.get("rvl_generation"), dict)
            or task_metadata.get("rvl_group_id", group_id) != group_id
            or task_metadata.get("vare_rollout_group") != group_id
            or task_metadata.get("vare_rollout_group_size") != group_size
        ):
            raise ValueError("authenticated RVL member identity is incomplete or inconsistent")

        previous = grouped.get(group_id)
        if previous is None:
            indices: set[int] = set()
            grouped[group_id] = (group_size, indices)
        else:
            expected_size, indices = previous
            if expected_size != group_size:
                raise ValueError("authenticated RVL group has inconsistent declared sizes")
        if member_index in indices:
            raise ValueError("authenticated RVL group repeats a member index")
        indices.add(member_index)

    for group_id, (group_size, indices) in grouped.items():
        # Bounds and uniqueness were checked while collecting indices. Having
        # exactly group_size distinct in-range indices proves full coverage
        # without allocating a set proportional to an untrusted declaration.
        if len(indices) != group_size:
            raise ValueError(f"authenticated RVL group {group_id!r} is incomplete")


@dataclass(frozen=True, repr=False, slots=True)
class RewardEvidenceRoot:
    """Trusted evaluator's signing context (HMAC key never enters learner/replay).

    In a real deployment keep the independent signer isolated from agents.
    The trusted consumer needs the key for verification; use public-key
    signatures instead if consumers must not possess a signing capability.
    """

    key: bytes
    run_id: str
    protocol_sha256: str
    evaluator_sha256: str
    verifier_config_sha256: str

    def __post_init__(self) -> None:
        if type(self.key) is not bytes or len(self.key) < 32:
            raise ValueError("reward evidence key must be at least 32 bytes")
        if not isinstance(self.run_id, str) or not 16 <= len(self.run_id) <= 256:
            raise ValueError("run ID must be a unique operator-issued value")
        for name in ("protocol_sha256", "evaluator_sha256", "verifier_config_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA.fullmatch(value) is None:
                raise ValueError(f"{name} requires a pinned lowercase SHA-256")

    @staticmethod
    def record_key(experience: Experience) -> str:
        """Stable group-member locator for an independently stored receipt."""
        md = experience.attempt.metadata
        if not isinstance(md, dict):
            raise ValueError("attempt metadata is missing")
        group = md.get("rvl_group_id")
        index = md.get("rvl_generation_index")
        if (not isinstance(group, str) or not group
                or type(index) is not int or index < 0):
            raise ValueError("reward lacks a complete group/member identity")
        body = {"group_id": group, "generation_index": index}
        return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()

    def _payload(self, experience: Experience) -> dict[str, object]:
        # Import lazily: importing integrations.rvl at module load initializes
        # the integrations package, which imports RVLGRPOHooks and this module.
        from .integrations.rvl import RVLTokenReplayReader

        attempt = experience.attempt
        verification = experience.verification
        md = attempt.metadata
        vmd = verification.metadata
        if not isinstance(md, dict) or not isinstance(vmd, dict):
            raise ValueError("missing reward provenance")
        group = md.get("rvl_group_id")
        index = md.get("rvl_generation_index")
        size = md.get("vare_rollout_group_size")
        generation = md.get("rvl_generation")
        recorded_sha = md.get("rvl_generation_behavior_sha256")
        group_hash = md.get("rvl_group_behavior_sha256")
        hash_status = md.get("rvl_group_behavior_hash_status")
        snapshot = vmd.get("rvl_ready_reward_snapshot")
        if (
            not isinstance(group, str) or not group
            or type(index) is not int or index < 0
            or type(size) is not int or size <= index
            or type(attempt.policy_version) is not int or attempt.policy_version < 0
            or type(verification.verifier_version) is not int or verification.verifier_version < 0
            or not isinstance(attempt.policy_id, str) or not attempt.policy_id
            or not isinstance(attempt.task.id, str) or not attempt.task.id
            or not isinstance(attempt.task.prompt, str)
            or not isinstance(attempt.output, str)
            or not isinstance(verification.verifier_name, str) or not verification.verifier_name
            or type(verification.passed) is not bool
            or type(verification.trusted) is not bool
            or verification.trusted is not True
            or type(verification.score) not in (int, float)
            or not math.isfinite(float(verification.score))
            or not 0.0 <= verification.score <= 1.0
            or isinstance(snapshot, bool) or not isinstance(snapshot, (int, float))
            or not math.isfinite(float(snapshot))
            or float(snapshot) != float(verification.score)
            or vmd.get("rvl_replay_status") != "ready"
            or vmd.get("rvl_group_id") != group
            or md.get("vare_rollout_group") != group
            or vmd.get("rvl_verification_config_sha256") != self.verifier_config_sha256
            or hash_status not in _SUPPORTED_BEHAVIOR
            or not isinstance(group_hash, str) or _SHA.fullmatch(group_hash) is None
            or not isinstance(recorded_sha, str) or _SHA.fullmatch(recorded_sha) is None
            or not isinstance(generation, dict)
            or recorded_sha != RVLTokenReplayReader._generation_behavior_digest(generation)
            or generation.get("prompt_id") != attempt.task.id
            or generation.get("prompt") != attempt.task.prompt
            or generation.get("response") != attempt.output
            or attempt.task.metadata.get("rvl_group_id") != group
        ):
            raise ValueError("missing or inconsistent reward provenance")

        # Bind the complete generation JSON so response/token metadata edits
        # invalidate an issued receipt even if the displayed answer is equal.
        # Do not rely on the unkeyed group hash as authentication.
        generation_hash = hashlib.sha256(
            canonical_json(generation).encode("utf-8")
        ).hexdigest()
        return {
            "domain": _DOMAIN,
            "run_id": self.run_id,
            "protocol_sha256": self.protocol_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "verifier_config_sha256": self.verifier_config_sha256,
            "group_id": group,
            "member_index": index,
            "group_size": size,
            "group_behavior_sha256": group_hash,
            "generation_sha256": generation_hash,
            "task_id": attempt.task.id,
            "prompt": attempt.task.prompt,
            "response": attempt.output,
            "policy_id": attempt.policy_id,
            "policy_version": attempt.policy_version,
            "verifier_name": verification.verifier_name,
            "verifier_version": verification.verifier_version,
            "reward": float(verification.score),
            "passed": verification.passed,
            "trusted": verification.trusted,
        }

    def attest(self, experience: Experience) -> str:
        """Trusted grader ONLY: sign independently established reward evidence."""
        encoded = canonical_json(self._payload(experience)).encode("utf-8")
        return hmac.new(self.key, encoded, hashlib.sha256).hexdigest()

    def verifies(self, experience: Experience, receipt: str | None) -> bool:
        """Reject missing, edited, mismatched or foreign-study reward receipts."""
        if not isinstance(receipt, str) or _SHA.fullmatch(receipt) is None:
            return False
        try:
            expected = self.attest(experience)
        except (ValueError, TypeError, KeyError, OverflowError, AttributeError):
            return False
        return hmac.compare_digest(expected, receipt)


@dataclass(frozen=True, slots=True)
class RewardEvidenceManifest:
    """Operator-pinned sidecar: the replay DB cannot mint its own receipts."""

    root: RewardEvidenceRoot
    receipts: Mapping[str, str]

    def verified(self, experience: Experience) -> bool:
        try:
            key = self.root.record_key(experience)
        except (ValueError, TypeError, AttributeError):
            return False
        return self.root.verifies(experience, self.receipts.get(key))
