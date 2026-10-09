#!/usr/bin/env python3
"""Hash-ranked binary SNLI entailment screen helpers."""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence


def row_hash(premise: str, hypothesis: str) -> str:
    return hashlib.sha256((premise + "\n" + hypothesis).encode("utf-8")).hexdigest()


def premise_hash(premise: str) -> str:
    return hashlib.sha256(premise.encode("utf-8")).hexdigest()


def denylist_hashes(denylist: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return legacy text and premise hashes while rejecting unknown hash fields."""
    text_hashes: set[str] = set()
    premise_hashes: set[str] = set()
    for item in denylist.get("hashes", []):
        field = item.get("field")
        digest = item.get("sha256")
        if field not in {"text_hash", "premise_hash"} or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("pilot denylist contains an invalid hash record")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("pilot denylist contains a non-hexadecimal hash") from exc
        (text_hashes if field == "text_hash" else premise_hashes).add(digest)
    if not text_hashes or not premise_hashes:
        raise ValueError("pilot denylist is missing a required hash type")
    return text_hashes, premise_hashes


def select_validation_rows(
    dataset: Any,
    per_class: int,
    *,
    excluded_text_hashes: Iterable[str] = (),
    excluded_premise_hashes: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Select balanced entailment/not-entailment rows by prompt hash, without model outcomes."""
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    view = dataset.select_columns(["premise", "hypothesis", "label"])
    excluded_text = set(excluded_text_hashes)
    excluded_premise = set(excluded_premise_hashes)
    ranked = {0: [], 1: []}
    for index in range(len(view)):
        row = view[index]
        label = int(row["label"])
        if label not in (0, 1, 2):
            continue
        digest = row_hash(row["premise"], row["hypothesis"])
        source_premise_hash = premise_hash(row["premise"])
        if digest in excluded_text or source_premise_hash in excluded_premise:
            continue
        binary_label = int(label != 0)
        ranked[binary_label].append((digest, source_premise_hash, index, label))
    for values in ranked.values():
        values.sort(key=lambda item: (item[0], item[1]))
    if any(len(rows) < per_class for rows in ranked.values()):
        raise ValueError("not enough eligible SNLI rows in both binary classes")
    selected = []
    for binary_label in (0, 1):
        for rank, (digest, source_premise_hash, index, source_label) in enumerate(ranked[binary_label][:per_class]):
            selected.append({
                "dataset_split": "validation",
                "dataset_index": int(index),
                "prompt_sha256": digest,
                "premise_sha256": source_premise_hash,
                "source_label": int(source_label),
                "label": int(binary_label),
                "class_rank": int(rank),
            })
    selected.sort(key=lambda row: (row["label"], row["class_rank"]))
    if len({row["prompt_sha256"] for row in selected}) != len(selected):
        raise ValueError("duplicate premise/hypothesis prompt hash in selected cohort")
    return selected


def option_order(prompt_sha256: str) -> tuple[str, str]:
    """Return the binary class names shown for A and B, deterministically shuffled by prompt."""
    if len(prompt_sha256) != 64:
        raise ValueError("expected a SHA-256 prompt digest")
    flip = int(hashlib.sha256(("snli-binary-order-v1:" + prompt_sha256).encode()).hexdigest()[-1], 16) & 1
    labels = ("entailment", "not_entailment")
    return labels[::-1] if flip else labels


def render_prompt(premise: str, hypothesis: str, prompt_sha256: str) -> str:
    option_a, option_b = option_order(prompt_sha256)
    meanings = {
        "entailment": "the hypothesis follows from the premise",
        "not_entailment": "the hypothesis does not follow from the premise",
    }
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Which relation holds? Choose the best answer.\n"
        f"A. {meanings[option_a]}\n"
        f"B. {meanings[option_b]}\n\n"
        "Answer with one letter only: A or B."
    )


def balanced_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if len(predictions) != len(labels) or not labels:
        raise ValueError("prediction/label lengths differ or are empty")
    recalls = []
    for class_id in (0, 1):
        indices = [i for i, value in enumerate(labels) if int(value) == class_id]
        if not indices:
            raise ValueError("balanced accuracy requires both classes")
        recalls.append(sum(int(predictions[i]) == class_id for i in indices) / len(indices))
    return sum(recalls) / 2


def class_recall(predictions: Sequence[int], labels: Sequence[int], class_id: int) -> float:
    indices = [i for i, value in enumerate(labels) if int(value) == class_id]
    if len(predictions) != len(labels) or not indices:
        raise ValueError("class recall requires aligned vectors and class support")
    return sum(int(predictions[i]) == class_id for i in indices) / len(indices)
