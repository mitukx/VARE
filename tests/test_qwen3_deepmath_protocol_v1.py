from __future__ import annotations

import hashlib

import pytest

from scripts.prompt_id_hash_v3 import canonical_prompt_id_sha256
from scripts.qwen3_deepmath_protocol_v1 import (
    GATE_COUNT,
    GATE_SEED,
    MATH_INSTRUCTION,
    UPDATE_COUNT,
    UPDATE_SEED,
    cohort_rows,
    grpo_training_record,
    prompt_id,
    prompt_messages,
    rank_prompt,
    sft_record,
)


def _rows(count: int) -> list[dict]:
    return [
        {"prompt_key": f"question {index}", "prompt": f"Question {index}", "solution": str(index)}
        for index in range(count)
    ]


def test_prompt_ids_and_rankings_are_stable():
    key = "normalized math question"
    assert prompt_id(key) == hashlib.sha256(key.encode()).hexdigest()
    assert rank_prompt(GATE_SEED, key) == rank_prompt(GATE_SEED, key)
    assert rank_prompt(GATE_SEED, key) != rank_prompt(UPDATE_SEED, key)


def test_cohorts_are_disjoint_and_exclude_retired_ids():
    rows = _rows(400)
    retired = {prompt_id(rows[index]["prompt_key"]) for index in range(20)}
    gate, update, stats = cohort_rows(
        rows,
        retired,
        is_training_target_valid=lambda _solution: True,
    )
    gate_ids = {prompt_id(row["prompt_key"]) for row in gate}
    update_ids = {prompt_id(row["prompt_key"]) for row in update}

    assert len(gate) == GATE_COUNT
    assert len(update) == UPDATE_COUNT
    assert gate_ids.isdisjoint(update_ids)
    assert gate_ids.isdisjoint(retired)
    assert update_ids.isdisjoint(retired)
    assert stats["gate_ids_sha256"] == canonical_prompt_id_sha256(gate_ids)
    assert stats["update_ids_sha256"] == canonical_prompt_id_sha256(update_ids)
    assert stats["cohorts_disjoint"] is True


def test_selection_skips_invalid_rows_in_deterministic_order():
    rows = _rows(420)
    gate_ranked = sorted(rows, key=lambda row: rank_prompt(GATE_SEED, row["prompt_key"]))
    invalid = {prompt_id(row["prompt_key"]) for row in gate_ranked[:9]}
    gate, update, stats = cohort_rows(
        rows,
        (),
        is_training_target_valid=lambda solution: prompt_id(f"question {solution}") not in invalid,
    )
    assert stats["gate_unparseable_candidates_skipped"] == 9
    assert len(gate) == GATE_COUNT
    assert len(update) == UPDATE_COUNT


def test_prompt_instruction_is_identical_for_grpo_eval_and_sft():
    prompt = prompt_messages("Solve 2+2.")
    record = sft_record("Solve 2+2.", "4")
    assert prompt == record["prompt"]
    assert prompt[0] == {"role": "system", "content": MATH_INSTRUCTION}
    assert record["completion"] == [{"role": "assistant", "content": r"\boxed{4}"}]


def test_grpo_record_has_same_fixed_instruction_and_reward_provenance_id():
    row = {"prompt": [{"role": "user", "content": "Solve 2+2."}], "prompt_key": "solve 2+2", "solution": "4"}
    prepared = grpo_training_record(row)
    assert prepared["prompt"] == prompt_messages("Solve 2+2.")
    assert prepared["prompt_id"] == prompt_id("solve 2+2")
    assert prepared["solution"] == "4"


def test_grpo_record_accepts_deduplicated_deepmath_string_prompt():
    prepared = grpo_training_record({"prompt": "Solve 2+2.", "prompt_key": "solve 2+2"})
    assert prepared["prompt"] == prompt_messages("Solve 2+2.")
    assert prepared["prompt_id"] == prompt_id("solve 2+2")


def test_grpo_record_rejects_unexpected_prompt_shape():
    with pytest.raises(ValueError, match="exactly one user prompt"):
        grpo_training_record({"prompt": [], "prompt_key": "key"})


def test_cohort_selection_fails_closed_if_not_enough_valid_rows():
    with pytest.raises(ValueError, match="only selected"):
        cohort_rows(_rows(20), (), is_training_target_valid=lambda _solution: True)
