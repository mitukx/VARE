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


def build_rationale_example(row: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    """Build a verifier-chosen rationale row; rejected text is added after base rollout."""
    answer = parse_answer(row["answer"])
    rationale = row["answer"].rsplit("####", 1)[0].strip()
    if not rationale:
        raise ValueError("GSM8K verifier rationale is empty")
    key = digest_text("vare-gsm8k-sequence-choice-v1\0" + row["question"])
    user_message = (
        "Solve this math word problem. Show concise work and end with #### followed by the answer.\n\n"
        + row["question"]
    )
    return {
        "dataset_split": split,
        "dataset_index": int(index),
        "question": row["question"],
        "source_answer": row["answer"],
        "verifier_answer": f"{rationale}\n#### {format_number(answer)}",
        "expected_number": format_number(answer),
        "rejected_answer": None,
        "base_generated_text": None,
        "rejected_source": None,
        "user_message": user_message,
        "question_sha256": key,
    }


def make_rationale_rows(ds: Any, split: str, start_rank: int, stop_rank: int) -> list[dict[str, Any]]:
    if start_rank < 0 or stop_rank <= start_rank or stop_rank > len(ds):
        raise ValueError("invalid question-rank range")
    rows = [ds[i] for i in range(len(ds))]
    all_indices = select_indices(rows, len(rows))
    return [build_rationale_example(ds[index], index, split) for index in all_indices[start_rank:stop_rank]]


def attach_base_rollout_rejections(rows: list[dict[str, Any]], generations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use each frozen base rollout as reject; counterfactually flip only already-correct finals."""
    if len(rows) != len(generations):
        raise ValueError("base rollout count differs from row count")
    output = []
    for row, generation in zip(rows, generations):
        from decimal import Decimal
        if row["dataset_index"] != generation["dataset_index"]:
            raise ValueError("base rollout dataset index differs")
        generated = generation["generated_text"].strip()
        if not generated:
            raise ValueError("base rollout is empty")
        expected = Decimal(row.get("expected_number", row["verifier_answer"]))
        parsed = parse_generated_number(generated)
        if parsed == expected:
            replacement = expected + (Decimal(1) if int(row["question_sha256"][:2], 16) % 2 == 0 else Decimal(-1))
            if replacement == expected:
                raise AssertionError("counterfactual rejected number must differ")
            match = FINAL_NUMBER_RE.search(generated)
            if match is not None:
                rejected = generated[:match.start(1)] + format_number(replacement) + generated[match.end(1):]
            else:
                rejected = generated + f"\n#### {format_number(replacement)}"
            source = "base_rollout_correct_final_flipped"
        else:
            rejected = generated
            source = "base_rollout_incorrect_or_unparseable"
        item = dict(row)
        item.update({"rejected_answer": rejected, "base_generated_text": generated, "rejected_source": source})
        output.append(item)
    return output


def parse_generated_number(text: str) -> Decimal | None:
    match = FINAL_NUMBER_RE.search(text.strip())
    if match is None:
        return None
    try:
        return Decimal(match.group(1).replace(",", ""))
    except Exception:
        return None
