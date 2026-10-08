#!/usr/bin/env python3
"""Hash-ranked verifier-labeled response pairs for the BoolQ reading task."""
from __future__ import annotations
import hashlib
import re
from typing import Any

LABEL_RE = re.compile(r"\s*(?:the answer is\s+)?(yes|no)[.!]?\s*", re.IGNORECASE)


def digest_question(question: str, passage: str) -> str:
    return hashlib.sha256((question + "\n" + passage).encode("utf-8")).hexdigest()


def parse_generated_label(text: str) -> str | None:
    match = LABEL_RE.fullmatch(text)
    return match.group(1).lower() if match else None


def build_sequence_example(row: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    expected = "yes" if bool(row["answer"]) else "no"
    key = digest_question(row["question"], row["passage"])
    user_message = (
        "Read the passage and answer the question using the passage only. "
        "Reply with exactly one word: Yes or No.\n\n"
        f"Passage: {row['passage']}\n\nQuestion: {row['question']}"
    )
    return {
        "dataset_split": split,
        "dataset_index": int(index),
        "question": row["question"],
        "passage": row["passage"],
        "source_answer": bool(row["answer"]),
        "verifier_answer": expected,
        "rejected_answer": None,
        "base_generated_text": None,
        "rejected_source": None,
        "user_message": user_message,
        "question_sha256": key,
    }


def make_sequence_rows(ds: Any, split: str, start_rank: int, stop_rank: int) -> list[dict[str, Any]]:
    if start_rank < 0 or stop_rank <= start_rank or stop_rank > len(ds):
        raise ValueError("invalid BoolQ hash-rank range")
    ranked = sorted(range(len(ds)), key=lambda i: (digest_question(ds[i]["question"], ds[i]["passage"]), i))
    return [build_sequence_example(ds[i], i, split) for i in ranked[start_rank:stop_rank]]


def attach_base_rollout_rejections(rows: list[dict[str, Any]], generations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != len(generations):
        raise ValueError("base rollout count differs from row count")
    output = []
    for row, generation in zip(rows, generations):
        if row["dataset_index"] != generation["dataset_index"]:
            raise ValueError("base rollout dataset index differs")
        generated = generation["generated_text"].strip()
        if not generated:
            raise ValueError("base rollout is empty")
        parsed = parse_generated_label(generated)
        if parsed == row["verifier_answer"]:
            rejected = "no" if parsed == "yes" else "yes"
            source = "base_rollout_correct_label_flipped"
        else:
            rejected = generated
            source = "base_rollout_incorrect_or_unparseable"
        item = dict(row)
        item.update({"rejected_answer": rejected, "base_generated_text": generated, "rejected_source": source})
        output.append(item)
    return output
