#!/usr/bin/env python3
"""Deterministic verifier-labeled A/B preference examples from GSM8K."""
from __future__ import annotations

from decimal import Decimal
import hashlib
import re
from typing import Any

ANSWER_RE = re.compile(r"####\s*([+-]?(?:\d[\d,]*)(?:\.\d+)?)\s*$")


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_answer(answer: str) -> Decimal:
    match = ANSWER_RE.search(answer.strip())
    if match is None:
        raise ValueError("GSM8K answer has no supported #### final number")
    return Decimal(match.group(1).replace(",", ""))


def format_number(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return format(normalized.quantize(Decimal(1)), "f")
    return format(normalized, "f")


def build_example(row: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    answer = parse_answer(row["answer"])
    key = digest_text("vare-gsm8k-choice-v1\0" + row["question"])
    offset = Decimal(1) if int(key[:2], 16) % 2 == 0 else Decimal(-1)
    wrong = answer + offset
    if wrong == answer:
        raise AssertionError("distractor must differ from verifier answer")
    correct_is_a = int(key[2:4], 16) % 2 == 0
    candidate_a, candidate_b = (answer, wrong) if correct_is_a else (wrong, answer)
    prompt = (
        "Solve the following word problem. Choose the correct final numeric answer.\n\n"
        f"Problem: {row['question']}\n\n"
        f"A) {format_number(candidate_a)}\nB) {format_number(candidate_b)}\n\n"
        "Answer with A or B:"
    )
    return {
        "dataset_split": split,
        "dataset_index": int(index),
        "question": row["question"],
        "source_answer": row["answer"],
        "verifier_answer": format_number(answer),
        "candidate_a": format_number(candidate_a),
        "candidate_b": format_number(candidate_b),
        "correct_label": "A" if correct_is_a else "B",
        "prompt": prompt,
        "question_sha256": key,
    }


def select_indices(rows: list[dict[str, Any]], count: int) -> list[int]:
    """Pick examples by question hash, independent of answer/model outcomes."""
    ranked = sorted(
        range(len(rows)),
        key=lambda i: digest_text("vare-gsm8k-sample-v1\0" + rows[i]["question"]),
    )
    return ranked[:count]


def make_split(ds: Any, split: str, count: int) -> list[dict[str, Any]]:
    indices = select_indices([ds[i] for i in range(len(ds))], count)
    return [build_example(ds[i], i, split) for i in indices]
