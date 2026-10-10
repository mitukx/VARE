from __future__ import annotations

import pytest

from scripts.analyze_qwen25_v7_failure_modes import (
    FAILURE_ANNOTATIONS,
    _manual_annotation_rows,
    generated_token_count,
    quantile,
)


def test_generated_token_count_includes_first_eos_and_ignores_padding() -> None:
    assert generated_token_count([10, 11, 2, 0, 0], eos_token_id=2) == 3
    assert generated_token_count([10, 11], eos_token_id=2) == 2
    assert generated_token_count([10, 11, 0, 0], eos_token_id=None) == 4


def test_manual_truncation_annotations_cover_all_fixed_examples_and_quotes() -> None:
    completions = []
    for (prompt_index, completion_index), annotation in sorted(FAILURE_ANNOTATIONS.items()):
        completions.append({
            "prompt_index": prompt_index,
            "completion_index": completion_index,
            "prompt_id": f"prompt-{prompt_index}",
            "generated_token_count": 1024,
            "eos_reached": False,
            "truncated": True,
            "training_reward": 0.0,
            "independent_task_success": False,
            "valid_final_box": False,
            "completion": f"prefix {annotation['quote']} suffix",
        })

    rows = _manual_annotation_rows(completions)

    assert len(rows) == 26
    assert {(row["prompt_index"], row["completion_index"]) for row in rows} == set(FAILURE_ANNOTATIONS)
    assert all(row["evidence_quote"] in completions[index]["completion"] for index, row in enumerate(rows))


def test_manual_annotation_rejects_quote_not_present_in_source_text() -> None:
    completions = [{
        "prompt_index": prompt_index,
        "completion_index": completion_index,
        "prompt_id": f"prompt-{prompt_index}",
        "generated_token_count": 1024,
        "eos_reached": False,
        "truncated": True,
        "training_reward": 0.0,
        "independent_task_success": False,
        "valid_final_box": False,
        "completion": "quote deliberately absent",
    } for prompt_index, completion_index in sorted(FAILURE_ANNOTATIONS)]

    with pytest.raises(ValueError, match="quote not found verbatim"):
        _manual_annotation_rows(completions)


def test_quantile_uses_linear_interpolation() -> None:
    assert quantile([1, 2, 3, 4], 0.5) == 2.5
    assert quantile([], 0.5) is None
