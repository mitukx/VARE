"""CPU reference checks for event-graph RL data semantics.

This module does not train a policy or define a production rollout format. It
validates a small JSONL contract used by synthetic fixtures and makes the
resulting policy-token targets and execution-normalized weights inspectable.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class TrajectoryContractError(ValueError):
    """An event graph cannot be interpreted without changing its semantics."""


_PROVENANCE = {"policy", "environment", "harness", "unknown"}
_EVENT_TYPES = {
    "model_call",
    "tool_call",
    "tool_result",
    "compaction",
    "subagent_start",
    "subagent_end",
    "retry",
    "timeout",
    "cancelled",
    "error",
    "evaluation",
}


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    """Hash an exact token-ID sequence using canonical JSON encoding."""
    encoded = json.dumps(
        list(token_ids), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryContractError(f"{key} must be a nonempty string")
    return value


def _optional_id(record: Mapping[str, Any], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryContractError(f"{key} must be null or a nonempty string")
    return value


def _token_ids(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        type(token) is not int or token < 0 for token in value
    ):
        raise TrajectoryContractError(
            f"{field} must be a list of nonnegative integer token IDs"
        )
    return tuple(value)


def _finite_logprob(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TrajectoryContractError(f"{field} must be a finite log-probability <= 0")
    result = float(value)
    if not math.isfinite(result) or result > 0.0:
        raise TrajectoryContractError(f"{field} must be a finite log-probability <= 0")
    return result


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise TrajectoryContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class TrajectoryEvent:
    group_id: str
    group_size: int
    execution_id: str
    event_id: str
    event_type: str
    branch_id: str
    causal_parent_id: str | None
    context_parent_id: str | None
    role: str | None = None
    policy_id: str | None = None
    policy_version: int | None = None
    model_id: str | None = None
    sampling_temperature: float | None = None
    logprob_distribution: str | None = None
    input_token_ids: tuple[int, ...] = ()
    input_token_ids_sha256: str | None = None
    output_token_ids: tuple[int, ...] = ()
    output_logprobs: tuple[float | None, ...] = ()
    token_provenance: tuple[str, ...] = ()
    evaluator_status: str | None = None
    evaluator_id: str | None = None
    evaluator_config_sha256: str | None = None
    evaluator_version: int | None = None
    assessment_scope: str | None = None
    artifact_sha256: str | None = None
    score: float | None = None
    designated_outcome: bool = False

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "TrajectoryEvent":
        if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
            raise TrajectoryContractError("schema_version must be integer 1")
        group_size = record.get("group_size")
        if type(group_size) is not int or group_size <= 0:
            raise TrajectoryContractError("group_size must be a positive integer")

        event_type = _required_text(record, "event_type")
        if event_type not in _EVENT_TYPES:
            raise TrajectoryContractError(f"unknown event_type {event_type!r}")
        event = cls(
            group_id=_required_text(record, "group_id"),
            group_size=group_size,
            execution_id=_required_text(record, "execution_id"),
            event_id=_required_text(record, "event_id"),
            event_type=event_type,
            branch_id=_required_text(record, "branch_id"),
            causal_parent_id=_optional_id(record, "causal_parent_id"),
            context_parent_id=_optional_id(record, "context_parent_id"),
        )
        if event_type == "model_call":
            return cls._model_call(event, record)
        if event_type == "evaluation":
            return cls._evaluation(event, record)
        return event

    @classmethod
    def _model_call(
        cls, event: "TrajectoryEvent", record: Mapping[str, Any]
    ) -> "TrajectoryEvent":
        role = _required_text(record, "role")
        policy_id = _required_text(record, "policy_id")
        policy_version = record.get("policy_version")
        if type(policy_version) is not int or policy_version < 0:
            raise TrajectoryContractError("policy_version must be a nonnegative integer")
        model_id = _required_text(record, "model_id")
        raw_temperature = record.get("sampling_temperature")
        if isinstance(raw_temperature, bool) or not isinstance(raw_temperature, Real):
            raise TrajectoryContractError(
                "sampling_temperature must be a finite positive number"
            )
        sampling_temperature = float(raw_temperature)
        if not math.isfinite(sampling_temperature) or sampling_temperature <= 0.0:
            raise TrajectoryContractError(
                "sampling_temperature must be a finite positive number"
            )
        logprob_distribution = _required_text(record, "logprob_distribution")
        if logprob_distribution != "neutral_temperature_scaled_policy":
            raise TrajectoryContractError(
                "unsupported behavior log-probability distribution"
            )
        input_ids = _token_ids(record.get("input_token_ids"), "input_token_ids")
        context_hash = _require_sha256(
            record.get("input_token_ids_sha256"), "input_token_ids_sha256"
        )
        if token_ids_sha256(input_ids) != context_hash:
            raise TrajectoryContractError(
                f"{event.execution_id}/{event.event_id}: input token hash mismatch"
            )

        output_ids = _token_ids(record.get("output_token_ids"), "output_token_ids")
        raw_logprobs = record.get("output_logprobs")
        raw_provenance = record.get("token_provenance")
        if not isinstance(raw_logprobs, list) or len(raw_logprobs) != len(output_ids):
            raise TrajectoryContractError(
                "output_logprobs must align one-for-one with output_token_ids"
            )
        if not isinstance(raw_provenance, list) or len(raw_provenance) != len(output_ids):
            raise TrajectoryContractError(
                "token_provenance must align one-for-one with output_token_ids"
            )
        logprobs: list[float | None] = []
        provenance: list[str] = []
        for index, (raw_logprob, raw_origin) in enumerate(
            zip(raw_logprobs, raw_provenance)
        ):
            if not isinstance(raw_origin, str) or raw_origin not in _PROVENANCE:
                raise TrajectoryContractError(
                    f"token_provenance[{index}] is not a recognized provenance value"
                )
            if raw_origin == "policy":
                logprobs.append(
                    _finite_logprob(raw_logprob, f"output_logprobs[{index}]")
                )
            elif raw_logprob is None:
                logprobs.append(None)
            else:
                logprobs.append(
                    _finite_logprob(raw_logprob, f"output_logprobs[{index}]")
                )
            provenance.append(raw_origin)
        return cls(
            group_id=event.group_id,
            group_size=event.group_size,
            execution_id=event.execution_id,
            event_id=event.event_id,
            event_type=event.event_type,
            branch_id=event.branch_id,
            causal_parent_id=event.causal_parent_id,
            context_parent_id=event.context_parent_id,
            role=role,
            policy_id=policy_id,
            policy_version=policy_version,
            model_id=model_id,
            sampling_temperature=sampling_temperature,
            logprob_distribution=logprob_distribution,
            input_token_ids=input_ids,
            input_token_ids_sha256=context_hash,
            output_token_ids=output_ids,
            output_logprobs=tuple(logprobs),
            token_provenance=tuple(provenance),
        )

    @classmethod
    def _evaluation(
        cls, event: "TrajectoryEvent", record: Mapping[str, Any]
    ) -> "TrajectoryEvent":
        status = _required_text(record, "evaluator_status")
        if status not in {"valid", "pending", "invalid"}:
            raise TrajectoryContractError(
                "evaluator_status must be valid, pending, or invalid"
            )
        version = record.get("evaluator_version")
        if type(version) is not int or version < 0:
            raise TrajectoryContractError(
                "evaluator_version must be a nonnegative integer"
            )
        evaluator_id = _required_text(record, "evaluator_id")
        evaluator_config_sha256 = _require_sha256(
            record.get("evaluator_config_sha256"), "evaluator_config_sha256"
        )
        designated = record.get("designated_outcome")
        if type(designated) is not bool:
            raise TrajectoryContractError("designated_outcome must be bool")
        scope = _required_text(record, "assessment_scope")
        if scope not in {"partial", "final", "intermediate"}:
            raise TrajectoryContractError(
                "assessment_scope must be partial, final, or intermediate"
            )
        if designated and scope != "final":
            raise TrajectoryContractError(
                "a designated training outcome must have final assessment_scope"
            )
        artifact = record.get("artifact_sha256")
        if artifact is not None:
            artifact = _require_sha256(artifact, "artifact_sha256")
        raw_score = record.get("score")
        score: float | None = None
        if status == "valid":
            if artifact is None:
                raise TrajectoryContractError(
                    "a valid evaluation must identify its scored artifact"
                )
            if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
                raise TrajectoryContractError(
                    "a valid evaluation must have a finite score in [0, 1]"
                )
            score = float(raw_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise TrajectoryContractError(
                    "a valid evaluation must have a finite score in [0, 1]"
                )
        elif raw_score is not None:
            raise TrajectoryContractError(
                "pending or invalid evaluation cannot carry a training score"
            )
        return cls(
            group_id=event.group_id,
            group_size=event.group_size,
            execution_id=event.execution_id,
            event_id=event.event_id,
            event_type=event.event_type,
            branch_id=event.branch_id,
            causal_parent_id=event.causal_parent_id,
            context_parent_id=event.context_parent_id,
            evaluator_status=status,
            evaluator_id=evaluator_id,
            evaluator_config_sha256=evaluator_config_sha256,
            evaluator_version=version,
            assessment_scope=scope,
            artifact_sha256=artifact,
            score=score,
            designated_outcome=designated,
        )


@dataclass(frozen=True, slots=True)
class TrainingTarget:
    group_id: str
    execution_id: str
    event_id: str
    branch_id: str
    role: str
    policy_id: str
    token_index: int
    token_id: int
    input_token_ids_sha256: str
    behavior_logprob: float
    model_id: str
    sampling_temperature: float
    logprob_distribution: str
    policy_version: int
    outcome_score: float
    evaluator_id: str
    evaluator_config_sha256: str
    evaluator_version: int
    assessment_scope: str
    artifact_sha256: str
    weight: float

    @property
    def target_id(self) -> str:
        return f"{self.group_id}:{self.execution_id}:{self.event_id}:{self.token_index}"


@dataclass(frozen=True, slots=True)
class ValidatedExecution:
    execution_id: str
    group_id: str
    group_size: int
    events: tuple[TrajectoryEvent, ...]

    def designated_outcome(self) -> TrajectoryEvent | None:
        outcomes = [
            event
            for event in self.events
            if event.event_type == "evaluation" and event.designated_outcome
        ]
        return outcomes[0] if outcomes else None

    def event_paths(self, relation: str = "context") -> Iterator[tuple[str, ...]]:
        """Yield root-to-leaf paths without merging causal and context ancestry."""
        if relation not in {"context", "causal"}:
            raise ValueError("relation must be 'context' or 'causal'")
        parent_field = f"{relation}_parent_id"
        by_id = {event.event_id: event for event in self.events}
        children: dict[str | None, list[str]] = {}
        for event in self.events:
            parent_id = getattr(event, parent_field)
            children.setdefault(parent_id, []).append(event.event_id)
        for values in children.values():
            values.sort()
        leaves = sorted(
            event_id for event_id in by_id if not children.get(event_id)
        )
        for leaf in leaves:
            reverse_path: list[str] = []
            current: str | None = leaf
            while current is not None:
                reverse_path.append(current)
                current = getattr(by_id[current], parent_field)
            yield tuple(reversed(reverse_path))


@dataclass(frozen=True, slots=True)
class ValidatedGroup:
    group_id: str
    group_size: int
    executions: tuple[ValidatedExecution, ...]

    def materialize(self) -> tuple[TrainingTarget, ...]:
        """Select policy-generated targets once and weight executions equally.

        The returned weights sum to 1 / group_size for each execution with at
        least one policy target. An all-masked execution contributes zero but
        remains in the original group denominator.
        """
        targets: list[TrainingTarget] = []
        for execution in self.executions:
            outcome = execution.designated_outcome()
            if outcome is None:
                raise TrajectoryContractError(
                    f"{execution.execution_id}: designated evaluation is missing"
                )
            if outcome.evaluator_status != "valid":
                raise TrajectoryContractError(
                    f"{execution.execution_id}: designated evaluation is "
                    f"{outcome.evaluator_status}; only valid assessments enter a group"
                )
            if outcome.assessment_scope != "final":
                raise TrajectoryContractError(
                    f"{execution.execution_id}: designated evaluation must be final"
                )
            assert outcome.score is not None
            assert outcome.evaluator_id is not None
            assert outcome.evaluator_config_sha256 is not None
            assert outcome.evaluator_version is not None
            assert outcome.assessment_scope is not None
            assert outcome.artifact_sha256 is not None
            policy_tokens = [
                (event, index)
                for event in execution.events
                if event.event_type == "model_call"
                for index, origin in enumerate(event.token_provenance)
                if origin == "policy"
            ]
            if not policy_tokens:
                continue
            weight = 1.0 / (self.group_size * len(policy_tokens))
            for event, index in policy_tokens:
                assert event.policy_version is not None
                assert event.role is not None
                assert event.policy_id is not None
                assert event.model_id is not None
                assert event.sampling_temperature is not None
                assert event.logprob_distribution is not None
                assert event.input_token_ids_sha256 is not None
                behavior_logprob = event.output_logprobs[index]
                assert behavior_logprob is not None
                targets.append(
                    TrainingTarget(
                        group_id=self.group_id,
                        execution_id=execution.execution_id,
                        event_id=event.event_id,
                        branch_id=event.branch_id,
                        role=event.role,
                        policy_id=event.policy_id,
                        token_index=index,
                        token_id=event.output_token_ids[index],
                        input_token_ids_sha256=event.input_token_ids_sha256,
                        behavior_logprob=behavior_logprob,
                        model_id=event.model_id,
                        sampling_temperature=event.sampling_temperature,
                        logprob_distribution=event.logprob_distribution,
                        policy_version=event.policy_version,
                        outcome_score=outcome.score,
                        evaluator_id=outcome.evaluator_id,
                        evaluator_config_sha256=outcome.evaluator_config_sha256,
                        evaluator_version=outcome.evaluator_version,
                        assessment_scope=outcome.assessment_scope,
                        artifact_sha256=outcome.artifact_sha256,
                        weight=weight,
                    )
                )
        return tuple(
            sorted(targets, key=lambda target: (target.execution_id, target.event_id, target.token_index))
        )


def _validate_parent_forest(execution: ValidatedExecution, relation: str) -> None:
    parent_field = f"{relation}_parent_id"
    by_id = {event.event_id: event for event in execution.events}
    for event in execution.events:
        parent_id = getattr(event, parent_field)
        if parent_id is not None and parent_id not in by_id:
            raise TrajectoryContractError(
                f"{execution.execution_id}/{event.event_id}: unknown {relation} parent "
                f"{parent_id}"
            )
    state: dict[str, int] = {}
    for start_id in sorted(by_id):
        path: list[str] = []
        current_id: str | None = start_id
        while current_id is not None and state.get(current_id, 0) != 2:
            if state.get(current_id, 0) == 1:
                raise TrajectoryContractError(
                    f"{execution.execution_id}: cycle in {relation} parent links"
                )
            state[current_id] = 1
            path.append(current_id)
            current_id = getattr(by_id[current_id], parent_field)
        for event_id in path:
            state[event_id] = 2


def validate_event_records(records: Iterable[Mapping[str, Any]]) -> tuple[ValidatedGroup, ...]:
    """Validate JSON-like records and return complete, deterministic groups."""
    events: list[TrajectoryEvent] = []
    seen: set[tuple[str, str, str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TrajectoryContractError(f"record {index} must be an object")
        event = TrajectoryEvent.from_mapping(record)
        identity = (event.group_id, event.execution_id, event.event_id)
        if identity in seen:
            raise TrajectoryContractError(
                "duplicate event identity "
                f"{event.group_id}/{event.execution_id}/{event.event_id}"
            )
        seen.add(identity)
        events.append(event)

    grouped: dict[str, list[TrajectoryEvent]] = {}
    for event in events:
        grouped.setdefault(event.group_id, []).append(event)

    result: list[ValidatedGroup] = []
    for group_id in sorted(grouped):
        group_events = grouped[group_id]
        sizes = {event.group_size for event in group_events}
        if len(sizes) != 1:
            raise TrajectoryContractError(f"{group_id}: inconsistent group_size")
        group_size = sizes.pop()
        by_execution: dict[str, list[TrajectoryEvent]] = {}
        for event in group_events:
            by_execution.setdefault(event.execution_id, []).append(event)
        if len(by_execution) != group_size:
            raise TrajectoryContractError(
                f"{group_id}: incomplete group, expected {group_size} executions, "
                f"found {len(by_execution)}"
            )

        executions: list[ValidatedExecution] = []
        for execution_id in sorted(by_execution):
            execution_events = tuple(
                sorted(by_execution[execution_id], key=lambda event: event.event_id)
            )
            execution = ValidatedExecution(
                execution_id=execution_id,
                group_id=group_id,
                group_size=group_size,
                events=execution_events,
            )
            _validate_parent_forest(execution, "causal")
            _validate_parent_forest(execution, "context")
            evaluation_events = [
                event for event in execution_events if event.event_type == "evaluation"
            ]
            designated = [event for event in evaluation_events if event.designated_outcome]
            if len(designated) > 1:
                raise TrajectoryContractError(
                    f"{execution_id}: multiple designated outcome evaluations"
                )
            branches = {
                event.branch_id
                for event in execution_events
                if event.event_type != "evaluation"
            }
            if any(event.branch_id not in branches for event in designated):
                raise TrajectoryContractError(
                    f"{execution_id}: designated evaluation references an unknown branch"
                )
            executions.append(execution)
        result.append(
            ValidatedGroup(
                group_id=group_id,
                group_size=group_size,
                executions=tuple(executions),
            )
        )
    return tuple(result)


def read_event_jsonl(path: str | Path) -> tuple[ValidatedGroup, ...]:
    """Read and validate one raw JSONL trace file."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrajectoryContractError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise TrajectoryContractError(f"line {line_number}: expected a JSON object")
            records.append(record)
    return validate_event_records(records)
