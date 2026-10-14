from pathlib import Path

import pytest

from vare.trajectory_reference import (
    TrajectoryContractError,
    read_event_jsonl,
    token_ids_sha256,
    validate_event_records,
)


def _model(
    execution_id,
    event_id,
    *,
    group_size=2,
    branch="main",
    role="main",
    causal_parent=None,
    context_parent=None,
    input_ids=(1, 2),
    output_ids=(10,),
    logprobs=(-0.1,),
    provenance=None,
):
    input_ids = list(input_ids)
    if provenance is None:
        provenance = ("policy",) * len(output_ids)
    return {
        "schema_version": 1,
        "group_id": "g0",
        "group_size": group_size,
        "execution_id": execution_id,
        "event_id": event_id,
        "event_type": "model_call",
        "branch_id": branch,
        "role": role,
        "policy_id": "fixture-policy-v7",
        "causal_parent_id": causal_parent,
        "context_parent_id": context_parent,
        "policy_version": 7,
        "model_id": "fixture-model-revision",
        "sampling_temperature": 0.8,
        "logprob_distribution": "neutral_temperature_scaled_policy",
        "input_token_ids": input_ids,
        "input_token_ids_sha256": token_ids_sha256(input_ids),
        "output_token_ids": list(output_ids),
        "output_logprobs": list(logprobs),
        "token_provenance": list(provenance),
    }


def _event(execution_id, event_id, event_type, *, branch="main", causal=None, context=None, group_size=2):
    return {
        "schema_version": 1,
        "group_id": "g0",
        "group_size": group_size,
        "execution_id": execution_id,
        "event_id": event_id,
        "event_type": event_type,
        "branch_id": branch,
        "causal_parent_id": causal,
        "context_parent_id": context,
    }


def _evaluation(
    execution_id,
    event_id,
    *,
    branch="main",
    parent=None,
    status="valid",
    score=1.0,
    designated=True,
    scope="final",
    group_size=2,
):
    record = _event(
        execution_id,
        event_id,
        "evaluation",
        branch=branch,
        causal=parent,
        context=parent,
        group_size=group_size,
    )
    record.update(
        {
            "evaluator_id": "fixture-grader-v1",
            "evaluator_config_sha256": "b" * 64,
            "evaluator_status": status,
            "evaluator_version": 3,
            "assessment_scope": scope,
            "artifact_sha256": "a" * 64 if status == "valid" else None,
            "score": score if status == "valid" else None,
            "designated_outcome": designated,
        }
    )
    return record


def _two_execution_trace():
    # e0 has two context paths that share m0, then diverge after compaction and
    # delegation. The causal and context edges intentionally differ for s0.
    rows = [
        _model("e0", "m0", output_ids=(10, 11), logprobs=(-0.1, -0.2)),
        _event("e0", "compact0", "compaction", causal="m0", context="m0"),
        _model(
            "e0",
            "m1",
            causal_parent="compact0",
            context_parent="compact0",
            output_ids=(12,),
            logprobs=(-0.3,),
        ),
        _model(
            "e0",
            "s0",
            branch="subagent-0",
            role="subagent",
            causal_parent="m0",
            context_parent=None,
            input_ids=(90,),
            output_ids=(13,),
            logprobs=(-0.4,),
        ),
        # Assistant role alone is not enough to make a harness-inserted token
        # a policy target.
        _model(
            "e0",
            "inserted0",
            causal_parent="m1",
            context_parent="m1",
            output_ids=(99,),
            logprobs=(None,),
            provenance=("harness",),
        ),
        _event("e0", "timeout0", "timeout", causal="m1", context="inserted0"),
        _evaluation(
            "e0",
            "eval-partial",
            parent="timeout0",
            score=0.75,
            designated=False,
            scope="partial",
        ),
        _evaluation(
            "e0",
            "eval-final",
            parent="eval-partial",
            score=1.0,
            scope="final",
        ),
        _model("e1", "m0", output_ids=(20,), logprobs=(-1.25,)),
        _evaluation("e1", "eval0", parent="m0", score=0.0),
    ]
    return rows


def test_reference_preserves_two_parent_relations_and_deduplicates_shared_targets():
    group = validate_event_records(_two_execution_trace())[0]
    e0 = next(execution for execution in group.executions if execution.execution_id == "e0")

    context_paths = set(e0.event_paths("context"))
    assert (
        "m0", "compact0", "m1", "inserted0", "timeout0", "eval-partial", "eval-final"
    ) in context_paths
    assert ("s0",) in context_paths
    causal_paths = set(e0.event_paths("causal"))
    assert ("m0", "s0") in causal_paths
    assert (
        "m0", "compact0", "m1", "timeout0", "eval-partial", "eval-final"
    ) in causal_paths
    assert ("m0", "compact0", "m1", "inserted0") in causal_paths

    targets = group.materialize()
    e0_targets = [target for target in targets if target.execution_id == "e0"]
    e1_targets = [target for target in targets if target.execution_id == "e1"]
    assert {(target.event_id, target.token_index) for target in e0_targets} == {
        ("m0", 0),
        ("m0", 1),
        ("m1", 0),
        ("s0", 0),
    }
    assert len({target.target_id for target in targets}) == len(targets)
    assert {target.weight for target in e0_targets} == {0.125}
    assert [target.weight for target in e1_targets] == [0.5]
    assert sum(target.weight for target in targets) == pytest.approx(1.0)
    assert e1_targets[0].outcome_score == 0.0
    assert e0_targets[0].assessment_scope == "final"
    assert {target.outcome_score for target in e0_targets} == {1.0}
    assert {target.evaluator_id for target in e0_targets} == {"fixture-grader-v1"}
    assert {target.evaluator_config_sha256 for target in e0_targets} == {"b" * 64}
    assert {target.policy_id for target in e0_targets} == {"fixture-policy-v7"}
    assert {target.model_id for target in e0_targets} == {"fixture-model-revision"}
    assert {target.sampling_temperature for target in e0_targets} == {0.8}
    assert {
        target.logprob_distribution for target in e0_targets
    } == {"neutral_temperature_scaled_policy"}
    assert {
        (target.event_id, target.token_index): target.behavior_logprob
        for target in e0_targets
    } == {("m0", 0): -0.1, ("m0", 1): -0.2, ("m1", 0): -0.3, ("s0", 0): -0.4}


def test_pending_or_invalid_assessment_never_becomes_zero_reward():
    rows = _two_execution_trace()
    rows[-1] = _evaluation("e1", "eval0", parent="m0", status="pending")
    group = validate_event_records(rows)[0]
    with pytest.raises(TrajectoryContractError, match="evaluation is pending"):
        group.materialize()

    rows[-1] = _evaluation("e1", "eval0", parent="m0", status="invalid")
    group = validate_event_records(rows)[0]
    with pytest.raises(TrajectoryContractError, match="evaluation is invalid"):
        group.materialize()


def test_invalid_evaluation_cannot_smuggle_in_a_zero_score():
    rows = _two_execution_trace()
    rows[-1]["evaluator_status"] = "invalid"
    rows[-1]["score"] = 0.0
    with pytest.raises(TrajectoryContractError, match="cannot carry a training score"):
        validate_event_records(rows)


def test_group_must_contain_all_declared_executions():
    rows = [row for row in _two_execution_trace() if row["execution_id"] == "e0"]
    with pytest.raises(TrajectoryContractError, match="incomplete group"):
        validate_event_records(rows)


def test_execution_with_no_policy_tokens_keeps_its_group_denominator():
    rows = _two_execution_trace()
    rows[8] = _model(
        "e1",
        "m0",
        output_ids=(20,),
        logprobs=(None,),
        provenance=("harness",),
    )
    group = validate_event_records(rows)[0]
    targets = group.materialize()
    assert {target.execution_id for target in targets} == {"e0"}
    assert sum(target.weight for target in targets) == pytest.approx(0.5)


def test_every_execution_requires_exactly_one_designated_outcome():
    rows = _two_execution_trace()
    rows.pop()
    group = validate_event_records(rows)[0]
    with pytest.raises(TrajectoryContractError, match="designated evaluation is missing"):
        group.materialize()

    rows = _two_execution_trace()
    rows.append(_evaluation("e1", "eval1", parent="m0"))
    with pytest.raises(TrajectoryContractError, match="multiple designated"):
        validate_event_records(rows)


@pytest.mark.parametrize("scope", ["partial", "intermediate"])
def test_nonfinal_assessment_cannot_be_designated_as_training_outcome(scope):
    rows = _two_execution_trace()
    rows[-1]["assessment_scope"] = scope
    with pytest.raises(TrajectoryContractError, match="must have final assessment_scope"):
        validate_event_records(rows)


def test_valid_partial_assessment_is_retained_but_final_outcome_drives_targets():
    group = validate_event_records(_two_execution_trace())[0]
    execution = next(item for item in group.executions if item.execution_id == "e0")
    partial = next(event for event in execution.events if event.event_id == "eval-partial")
    assert partial.score == 0.75
    assert not partial.designated_outcome
    targets = [target for target in group.materialize() if target.execution_id == "e0"]
    assert targets
    assert {target.outcome_score for target in targets} == {1.0}
    assert {target.assessment_scope for target in targets} == {"final"}


@pytest.mark.parametrize("score", [True, float("nan"), float("inf"), -0.1, 1.1, "0"])
def test_valid_evaluation_score_must_be_a_finite_unit_interval_number(score):
    rows = _two_execution_trace()
    rows[-1]["score"] = score
    with pytest.raises(TrajectoryContractError, match=r"finite score in \[0, 1\]"):
        validate_event_records(rows)


def test_valid_assessment_must_bind_a_hashed_artifact():
    rows = _two_execution_trace()
    rows[-1]["artifact_sha256"] = None
    with pytest.raises(TrajectoryContractError, match="identify its scored artifact"):
        validate_event_records(rows)


def test_evaluation_must_bind_evaluator_identity_and_config():
    rows = _two_execution_trace()
    rows[-1]["evaluator_id"] = "  "
    with pytest.raises(TrajectoryContractError, match="evaluator_id must be a nonempty string"):
        validate_event_records(rows)

    rows = _two_execution_trace()
    rows[-1]["evaluator_config_sha256"] = "not-a-hash"
    with pytest.raises(TrajectoryContractError, match="evaluator_config_sha256"):
        validate_event_records(rows)


def test_context_hash_mutation_is_detected_before_materialization():
    rows = _two_execution_trace()
    rows[0]["input_token_ids"] = [1, 2, 3]
    with pytest.raises(TrajectoryContractError, match="input token hash mismatch"):
        validate_event_records(rows)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(output_logprobs=[]),
        lambda row: row.update(token_provenance=["unknown-origin"]),
        lambda row: row.update(output_logprobs=[float("nan")]),
        lambda row: row.update(output_logprobs=[0.01]),
        lambda row: row.update(output_token_ids=[-1]),
        lambda row: row.update(policy_version=True),
        lambda row: row.update(logprob_distribution="greedy_scores_not_trainable"),
    ],
)
def test_model_call_alignment_and_provenance_are_strict(mutation):
    rows = _two_execution_trace()
    mutation(rows[0])
    with pytest.raises(TrajectoryContractError):
        validate_event_records(rows)


@pytest.mark.parametrize("temperature", [True, 0, -0.1, float("nan"), float("inf")])
def test_model_call_requires_a_finite_positive_behavior_temperature(temperature):
    rows = _two_execution_trace()
    rows[0]["sampling_temperature"] = temperature
    with pytest.raises(TrajectoryContractError, match="sampling_temperature"):
        validate_event_records(rows)


def test_parent_must_exist_and_each_relation_must_be_acyclic():
    rows = _two_execution_trace()
    rows[1]["context_parent_id"] = "missing"
    with pytest.raises(TrajectoryContractError, match="unknown context parent"):
        validate_event_records(rows)

    rows = _two_execution_trace()
    rows[0]["causal_parent_id"] = "compact0"
    with pytest.raises(TrajectoryContractError, match="cycle in causal"):
        validate_event_records(rows)


def test_duplicate_event_identity_is_rejected_even_if_payload_matches():
    rows = _two_execution_trace()
    rows.append(dict(rows[0]))
    with pytest.raises(TrajectoryContractError, match="duplicate event identity"):
        validate_event_records(rows)


def test_event_identity_may_be_reused_across_distinct_groups():
    rows = _two_execution_trace()
    second_group = [dict(row, group_id="g1") for row in rows]

    groups = validate_event_records([*rows, *second_group])

    assert [group.group_id for group in groups] == ["g0", "g1"]
    assert all(len(group.materialize()) > 0 for group in groups)


def test_long_event_chain_validates_without_python_recursion():
    first = _model("e0", "m0", group_size=1)
    rows = [first]
    parent = "m0"
    for index in range(1100):
        event_id = f"tool-{index:04d}"
        rows.append(
            _event("e0", event_id, "tool_result", causal=parent, context=parent, group_size=1)
        )
        parent = event_id
    rows.append(_evaluation("e0", "eval0", parent=parent, group_size=1))

    group = validate_event_records(rows)[0]
    execution = group.executions[0]
    path = next(execution.event_paths("context"))
    assert len(path) == len(rows)
    assert group.materialize()[0].weight == 1.0


def test_jsonl_reader_replays_raw_trace_deterministically():
    path = Path(__file__).parent / "fixtures" / "trajectory_reference" / "v1.jsonl"
    first = read_event_jsonl(path)[0].materialize()
    second = read_event_jsonl(path)[0].materialize()
    assert first == second
