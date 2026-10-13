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
from typing import Mapping

from .evidence import canonical_json
from .integrations.rvl import RVLTokenReplayReader
from .types import Experience


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_DOMAIN = "vare.rvl-reward-attestation.v1"
_SUPPORTED_BEHAVIOR = frozenset({"generation_match", "verified_record_match"})


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
            or type(verification.score) not in (int, float)
            or not math.isfinite(float(verification.score))
            or not 0.0 <= verification.score <= 1.0
            or type(snapshot) not in (int, float) or not math.isfinite(float(snapshot))
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
