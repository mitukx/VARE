#!/usr/bin/env python3
"""Verifier-labeled numeric response pairs for sequence-level DPO."""
from __future__ import annotations

from decimal import Decimal
import re
from typing import Any

from gsm8k_preference_task import digest_text, format_number, parse_answer, select_indices

FINAL_NUMBER_RE = re.compile(r"([+-]?(?:\d[\d,]*)(?:\.\d+)?)\s*[.!?]?\s*$")


def build_sequence_example(row: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    answer = parse_answer(row["answer"])
    key = digest_text("vare-gsm8k-sequence-choice-v1\0" + row["question"])
    wrong = answer + (Decimal(1) if int(key[:2], 16) % 2 == 0 else Decimal(-1))
    if wrong == answer:
        raise AssertionError("rejected answer must differ from verifier answer")
    user_message = (
        "Solve this math word problem. Reply with only the final number, without explanation.\n\n"
        + row["question"]
    )
    return {
        "dataset_split": split,
        "dataset_index": int(index),
        "question": row["question"],
        "source_answer": row["answer"],
        "verifier_answer": format_number(answer),
        "rejected_answer": format_number(wrong),
        "user_message": user_message,
        "question_sha256": key,
    }


def make_sequence_rows(ds: Any, split: str, start_rank: int, stop_rank: int) -> list[dict[str, Any]]:
    if start_rank < 0 or stop_rank <= start_rank or stop_rank > len(ds):
        raise ValueError("invalid question-rank range")
    rows = [ds[i] for i in range(len(ds))]
    all_indices = select_indices(rows, len(rows))
    return [build_sequence_example(ds[index], index, split) for index in all_indices[start_rank:stop_rank]]


def parse_generated_number(text: str) -> Decimal | None:
    match = FINAL_NUMBER_RE.search(text.strip())
    if match is None:
        return None
    try:
        return Decimal(match.group(1).replace(",", ""))
    except Exception:
        return None
