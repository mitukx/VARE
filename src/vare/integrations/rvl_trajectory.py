"""Strict bridge from one complete RVL replay group to trajectory E0 records."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

from .rvl import RVLTokenReplayReader
from ..trajectory_reference import (
    TrajectoryContractError,
    ValidatedGroup,
    token_ids_sha256,
    validate_event_records,
)
from ..types import Experience


def _required_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrajectoryContractError(f"{field} must be an object")
    return value


def _response_record_fingerprint_sha256(
    prompt_id: str, prompt: str, response: str
) -> str:
    """Fingerprint the replayed task/response tuple, not verifier artifact bytes."""
    payload = json.dumps(
        {"prompt_id": prompt_id, "prompt": prompt, "response": response},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rvl_experiences_to_trajectory_group(
    experiences: Sequence[Experience],
) -> ValidatedGroup:
    """Convert one complete ready RVL group without retokenizing its output.

    RVL remains the owner of replay and training. This adapter is deliberately
    limited to one prompt/completion per execution, policy-generated response
    tokens, a recorded positive temperature, and ready final verifier outcomes.
    It does not reconstruct multi-turn or tool-call ancestry.
    """
    if not experiences:
        raise TrajectoryContractError("RVL replay group cannot be empty")

    first_metadata = _required_mapping(experiences[0].attempt.metadata, "attempt metadata")
    group_id = first_metadata.get("vare_rollout_group")
    group_size = first_metadata.get("vare_rollout_group_size")
    if not isinstance(group_id, str) or not group_id:
        raise TrajectoryContractError("RVL experience lacks a complete replay group ID")
    if type(group_size) is not int or group_size != len(experiences):
        raise TrajectoryContractError(
            "RVL replay group is incomplete relative to its declared group size"
        )

    records: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    group_prompt_identity: tuple[str, str] | None = None
    group_behavior_identity: tuple[Any, ...] | None = None
    group_evaluator_identity: tuple[Any, ...] | None = None
    group_source_hash: str | None = None
    for experience in experiences:
        attempt = experience.attempt
        metadata = _required_mapping(attempt.metadata, "attempt metadata")
        if (
            metadata.get("vare_rollout_group") != group_id
            or metadata.get("vare_rollout_group_size") != group_size
        ):
            raise TrajectoryContractError("RVL experiences mix replay group identities")
        generation = _required_mapping(metadata.get("rvl_generation"), "rvl_generation")
        generation_metadata = _required_mapping(
            generation.get("metadata"), "rvl_generation.metadata"
        )
        verification_metadata = _required_mapping(
            experience.verification.metadata, "verification metadata"
        )
        reward_snapshot = verification_metadata.get("rvl_ready_reward_snapshot")
        if (
            isinstance(reward_snapshot, bool)
            or not isinstance(reward_snapshot, Real)
            or not math.isfinite(float(reward_snapshot))
            or not 0.0 <= float(reward_snapshot) <= 1.0
        ):
            raise TrajectoryContractError(
                "RVL replay lacks a valid read-time reward snapshot"
            )
        current_reward = RVLTokenReplayReader._require_ready_reward(
            experience.verification.score
        )
        if current_reward != float(reward_snapshot):
            raise TrajectoryContractError(
                "RVL evaluation score changed after replay reading"
            )
        if metadata.get("rvl_group_id") != group_id:
            raise TrajectoryContractError(
                "RVL attempt lineage group ID differs from the replay group"
            )
        if verification_metadata.get("rvl_group_id") != group_id:
            raise TrajectoryContractError(
                "RVL verifier lineage group ID differs from the replay group"
            )
        if experience.verifier_lag != 0:
            raise TrajectoryContractError(
                "RVL verifier configuration hash is not version-bound for stale groups"
            )
        source_hash = metadata.get("rvl_group_behavior_sha256")
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(char not in "0123456789abcdef" for char in source_hash)
            or metadata.get("rvl_group_behavior_hash_status")
            not in {"generation_match", "verified_record_match"}
        ):
            raise TrajectoryContractError(
                "RVL source behavior hash does not match a supported replay encoding"
            )
        if group_source_hash is None:
            group_source_hash = source_hash
        elif group_source_hash != source_hash:
            raise TrajectoryContractError("RVL experiences mix source behavior hashes")
        prompt_id = generation.get("prompt_id")
        prompt = generation.get("prompt")
        response = generation.get("response")
        if not isinstance(prompt_id, str) or not isinstance(prompt, str) or not isinstance(response, str):
            raise TrajectoryContractError("RVL generation task and response fields must be strings")
        if str(attempt.task.id) != prompt_id or attempt.task.prompt != prompt:
            raise TrajectoryContractError("RVL task identity differs from its stored generation")
        if group_prompt_identity is None:
            group_prompt_identity = (prompt_id, prompt)
        elif group_prompt_identity != (prompt_id, prompt):
            raise TrajectoryContractError(
                "RVL GRPO group contains more than one prompt identity"
            )
        if attempt.output != response:
            raise TrajectoryContractError("RVL attempt output differs from its stored generation")

        generation_index = metadata.get("rvl_generation_index")
        if type(generation_index) is not int or generation_index < 0:
            raise TrajectoryContractError("RVL generation index must be a nonnegative integer")
        if generation_index in seen_indices:
            raise TrajectoryContractError("RVL replay group repeats a generation index")
        seen_indices.add(generation_index)

        prompt_token_ids = generation_metadata.get("prompt_token_ids")
        response_token_ids = generation_metadata.get("response_token_ids")
        response_token_logprobs = generation_metadata.get("response_token_logprobs")
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            raise TrajectoryContractError("RVL generation lacks exact prompt_token_ids")
        if not isinstance(response_token_ids, list) or not response_token_ids:
            raise TrajectoryContractError("RVL generation lacks exact response_token_ids")
        if not isinstance(response_token_logprobs, list):
            raise TrajectoryContractError("RVL generation lacks response token log-probabilities")
        if len(response_token_ids) != len(response_token_logprobs):
            raise TrajectoryContractError("RVL response token IDs/log-probabilities do not align")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in response_token_logprobs
        ):
            raise TrajectoryContractError("RVL response token log-probabilities must be finite")
        token_count = generation.get("token_count")
        if type(token_count) is not int or token_count != len(response_token_ids):
            raise TrajectoryContractError("RVL token_count differs from exact response token IDs")

        temperature = generation_metadata.get("sampling_temperature")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, Real)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0
        ):
            raise TrajectoryContractError(
                "RVL generation lacks a finite positive sampling_temperature"
            )
        distribution = generation_metadata.get("logprob_distribution")
        if distribution != "neutral_temperature_scaled_policy":
            raise TrajectoryContractError(
                "RVL log-probabilities are not identified as temperature-scaled policy behavior"
            )
        behavior_identity = (
            attempt.policy_id,
            attempt.policy_version,
            generation_metadata.get("model"),
            float(temperature),
            distribution,
        )
        if group_behavior_identity is None:
            group_behavior_identity = behavior_identity
        elif group_behavior_identity != behavior_identity:
            raise TrajectoryContractError(
                "RVL GRPO group mixes policy or behavior-distribution identities"
            )
        aggregate_logprob = generation.get("logprob")
        if isinstance(aggregate_logprob, bool) or not isinstance(aggregate_logprob, Real):
            raise TrajectoryContractError("RVL generation lacks its aggregate log-probability")
        if not math.isfinite(float(aggregate_logprob)) or not math.isclose(
            float(aggregate_logprob),
            math.fsum(float(value) for value in response_token_logprobs),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise TrajectoryContractError(
                "RVL aggregate log-probability differs from token log-probabilities"
            )

        evaluator_config_sha256 = verification_metadata.get(
            "rvl_verification_config_sha256"
        )
        if verification_metadata.get("rvl_replay_status") != "ready":
            raise TrajectoryContractError("only ready RVL evaluations can become final outcomes")
        if not isinstance(evaluator_config_sha256, str):
            raise TrajectoryContractError("RVL replay lacks verifier configuration provenance")
        evaluator_identity = (
            experience.verification.verifier_name,
            experience.verification.verifier_version,
            evaluator_config_sha256,
        )
        if group_evaluator_identity is None:
            group_evaluator_identity = evaluator_identity
        elif group_evaluator_identity != evaluator_identity:
            raise TrajectoryContractError(
                "RVL GRPO group mixes evaluator identities or configurations"
            )

        expected_generation_hash = metadata.get("rvl_generation_behavior_sha256")
        if (
            not isinstance(expected_generation_hash, str)
            or expected_generation_hash
            != RVLTokenReplayReader._generation_behavior_digest(generation)
        ):
            raise TrajectoryContractError(
                "RVL generation behavior payload changed after replay reading"
            )

        execution_id = f"{group_id}:member-{generation_index}"
        call_id = "generation"
        records.append(
            {
                "schema_version": 1,
                "group_id": group_id,
                "group_size": group_size,
                "execution_id": execution_id,
                "event_id": call_id,
                "event_type": "model_call",
                "branch_id": "main",
                "causal_parent_id": None,
                "context_parent_id": None,
                "role": "main",
                "policy_id": attempt.policy_id,
                "policy_version": attempt.policy_version,
                "model_id": generation_metadata.get("model"),
                "sampling_temperature": temperature,
                "logprob_distribution": distribution,
                "input_token_ids": prompt_token_ids,
                "input_token_ids_sha256": token_ids_sha256(prompt_token_ids),
                "output_token_ids": response_token_ids,
                "output_logprobs": response_token_logprobs,
                "token_provenance": ["policy"] * len(response_token_ids),
            }
        )
        records.append(
            {
                "schema_version": 1,
                "group_id": group_id,
                "group_size": group_size,
                "execution_id": execution_id,
                "event_id": "evaluation",
                "event_type": "evaluation",
                "branch_id": "main",
                "causal_parent_id": call_id,
                "context_parent_id": call_id,
                "evaluator_status": "valid",
                "evaluator_id": experience.verification.verifier_name,
                "evaluator_config_sha256": evaluator_config_sha256,
                "evaluator_version": experience.verification.verifier_version,
                "assessment_scope": "final",
                "artifact_sha256": _response_record_fingerprint_sha256(
                    prompt_id, prompt, response
                ),
                "score": experience.verification.score,
                "designated_outcome": True,
            }
        )

    group = validate_event_records(records)
    if seen_indices != set(range(group_size)):
        raise TrajectoryContractError(
            "RVL replay group generation indices do not cover every declared member"
        )
    if len(group) != 1:
        raise TrajectoryContractError("RVL bridge unexpectedly produced multiple groups")
    return group[0]
