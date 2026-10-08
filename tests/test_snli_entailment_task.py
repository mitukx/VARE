from __future__ import annotations

import pytest

from scripts.snli_entailment_task import (
    balanced_accuracy,
    class_recall,
    denylist_hashes,
    option_order,
    premise_hash,
    render_prompt,
    row_hash,
    select_validation_rows,
)
from scripts.audit_snli_entailment_base_feasibility_v1 import reconstruct_rows


class PromptLabelView:
    def __init__(self, rows, columns):
        assert set(columns) == {"premise", "hypothesis", "label"}
        self.rows = rows

    def select_columns(self, columns):
        assert set(columns) == {"premise", "hypothesis", "label"}
        return self

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)


def test_selection_is_balanced_deterministic_and_excludes_prompt_hashes():
    rows = [
        {"premise": f"p{i}", "hypothesis": f"h{i}", "label": i % 3}
        for i in range(18)
    ] + [{"premise": "invalid", "hypothesis": "row", "label": -1}]
    view = PromptLabelView(rows, {"premise", "hypothesis", "label"})
    excluded = {row_hash("p0", "h0")}
    one = select_validation_rows(view, 3, excluded_text_hashes=excluded)
    two = select_validation_rows(view, 3, excluded_text_hashes=excluded)
    assert one == two
    assert [sum(row["label"] == cls for row in one) for cls in (0, 1)] == [3, 3]
    assert all(row["prompt_sha256"] not in excluded for row in one)
    assert all(row["source_label"] in (0, 1, 2) for row in one)


def test_selection_excludes_every_row_sharing_a_pilot_premise():
    view = PromptLabelView(
        [
            {"premise": "shared", "hypothesis": "h0", "label": 0},
            {"premise": "shared", "hypothesis": "h1", "label": 0},
            {"premise": "other", "hypothesis": "h2", "label": 0},
            {"premise": "negative", "hypothesis": "h3", "label": 1},
            {"premise": "negative-2", "hypothesis": "h4", "label": 2},
        ],
        {"premise", "hypothesis", "label"},
    )
    selected = select_validation_rows(view, 1, excluded_premise_hashes={premise_hash("shared")})
    assert {row["premise_sha256"] for row in selected}.isdisjoint({premise_hash("shared")})
    assert {row["dataset_index"] for row in selected} == {2, 3}


def test_denylist_parses_typed_hashes_and_fails_closed_on_unknown_fields():
    parsed = denylist_hashes({"hashes": [
        {"field": "text_hash", "sha256": "a" * 64},
        {"field": "premise_hash", "sha256": "b" * 64},
    ]})
    assert parsed == ({"a" * 64}, {"b" * 64})
    with pytest.raises(ValueError, match="invalid hash record"):
        denylist_hashes({"hashes": [{"field": "row_hash", "sha256": "c" * 64}]})


def test_independent_auditor_reconstructs_typed_hash_exclusions():
    rows = [
        {"premise": "shared", "hypothesis": "h0", "label": 0},
        {"premise": "shared", "hypothesis": "h1", "label": 0},
        {"premise": "other", "hypothesis": "h2", "label": 0},
        {"premise": "other-2", "hypothesis": "h5", "label": 0},
        {"premise": "negative", "hypothesis": "h3", "label": 1},
        {"premise": "negative-2", "hypothesis": "h4", "label": 2},
    ]
    view = PromptLabelView(rows, {"premise", "hypothesis", "label"})
    text_hashes = {row_hash("other", "h2")}
    premise_hashes = {premise_hash("shared")}
    denylist = {"hashes": [
        *({"field": "text_hash", "sha256": digest} for digest in text_hashes),
        *({"field": "premise_hash", "sha256": digest} for digest in premise_hashes),
    ]}
    expected = select_validation_rows(
        view, 1, excluded_text_hashes=text_hashes,
        excluded_premise_hashes=premise_hashes,
    )
    assert reconstruct_rows(view, denylist, 1) == expected


def test_selection_fails_closed_on_insufficient_binary_class_support():
    view = PromptLabelView(
        [{"premise": f"p{i}", "hypothesis": f"h{i}", "label": 0} for i in range(5)],
        {"premise", "hypothesis", "label"},
    )
    with pytest.raises(ValueError, match="not enough eligible"):
        select_validation_rows(view, 1)


def test_option_order_and_prompt_are_deterministic_and_clear():
    digest = row_hash("A premise.", "A hypothesis.")
    assert option_order(digest) in {("entailment", "not_entailment"), ("not_entailment", "entailment")}
    prompt = render_prompt("A premise.", "A hypothesis.", digest)
    assert "the hypothesis follows from the premise" in prompt
    assert "the hypothesis does not follow from the premise" in prompt
    assert prompt.endswith("Answer with one letter only: A or B.")
    assert prompt == render_prompt("A premise.", "A hypothesis.", digest)


def test_balanced_metrics_reject_misalignment_and_measure_each_class():
    labels = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]
    assert balanced_accuracy(predictions, labels) == pytest.approx(0.75)
    assert class_recall(predictions, labels, 0) == pytest.approx(0.5)
    assert class_recall(predictions, labels, 1) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        balanced_accuracy([0], labels)
