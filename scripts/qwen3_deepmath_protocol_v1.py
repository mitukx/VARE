"""Frozen prompt construction and cohort selection for Qwen3 / DeepMath v1."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable

from scripts.prompt_id_hash_v3 import canonical_prompt_id_sha256


GATE_SEED = "qwen3-deepmath-gate-v1"
UPDATE_SEED = "qwen3-deepmath-update-v1"
GATE_COUNT = 32
UPDATE_COUNT = 256
MATH_INSTRUCTION = (
    r"Please reason step by step and put your final answer within \boxed{}."
)


def prompt_id(prompt_key: str) -> str:
    return hashlib.sha256(prompt_key.encode("utf-8")).hexdigest()


def rank_prompt(seed: str, prompt_key: str) -> str:
    return hashlib.sha256((seed + "\0" + prompt_key).encode("utf-8")).hexdigest()


def answer_target(solution: str) -> str:
    """Extract the pinned math-verify parser's answer rendering for answer-only SFT."""
    from math_verify import parse

    parsed = parse(solution)
    if not parsed:
        raise ValueError("solution has no math-verify answer")
    candidate = parsed[1] if len(parsed) > 1 and isinstance(parsed[1], str) else parsed[0]
    target = str(candidate).strip()
    if not target:
        raise ValueError("solution's parsed SFT answer is empty")
    return target


def cohort_rows(
    eligible_rows: list[dict],
    retired_prompt_ids: Iterable[str],
    *,
    is_training_target_valid: Callable[[str], bool],
) -> tuple[list[dict], list[dict], dict]:
    """Select disjoint, parseable train-only cohorts while excluding retired IDs.

    The two phases use independent salted rankings. Gate rows are excluded from
    the update cohort even when they would rank early under its independent salt.
    """
    retired = set(retired_prompt_ids)
    ids = [prompt_id(row["prompt_key"]) for row in eligible_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("eligible rows must have unique normalized prompt IDs")

    def take(seed: str, count: int, excluded: set[str]) -> tuple[list[dict], int]:
        rows = sorted(
            (row for row in eligible_rows if prompt_id(row["prompt_key"]) not in excluded),
            key=lambda row: rank_prompt(seed, row["prompt_key"]),
        )
        selected: list[dict] = []
        skipped = 0
        for row in rows:
            if is_training_target_valid(row["solution"]):
                selected.append(row)
                if len(selected) == count:
                    break
            else:
                skipped += 1
        if len(selected) != count:
            raise ValueError(f"only selected {len(selected)} of {count} parseable rows")
        return selected, skipped

    gate, skipped_gate = take(GATE_SEED, GATE_COUNT, retired)
    gate_ids = {prompt_id(row["prompt_key"]) for row in gate}
    update, skipped_update = take(UPDATE_SEED, UPDATE_COUNT, retired | gate_ids)
    update_ids = {prompt_id(row["prompt_key"]) for row in update}
    if gate_ids & update_ids or gate_ids & retired or update_ids & retired:
        raise AssertionError("cohort selection violated exclusion/disjointness")
    return gate, update, {
        "retired_prompt_count": len(retired),
        "gate_count": len(gate),
        "gate_ids_sha256": canonical_prompt_id_sha256(gate_ids),
        "gate_unparseable_candidates_skipped": skipped_gate,
        "update_count": len(update),
        "update_ids_sha256": canonical_prompt_id_sha256(update_ids),
        "update_unparseable_candidates_skipped": skipped_update,
        "cohorts_disjoint": True,
    }


def prompt_messages(problem: str) -> list[dict[str, str]]:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty string")
    return [
        {"role": "system", "content": MATH_INSTRUCTION},
        {"role": "user", "content": problem},
    ]


def grpo_training_record(row: dict) -> dict:
    """Add the fixed prompt and provenance expected by TRL's reward callback."""
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        problem = prompt
    elif (
        isinstance(prompt, list)
        and len(prompt) == 1
        and isinstance(prompt[0], dict)
        and prompt[0].get("role") == "user"
    ):
        problem = prompt[0].get("content", "")
    else:
        raise ValueError("DeepMath rows must contain exactly one user prompt before preparation")
    key = row.get("prompt_key")
    if not isinstance(key, str) or not key:
        raise ValueError("DeepMath rows require a normalized prompt key")
    prepared = dict(row)
    prepared["prompt"] = prompt_messages(problem)
    prepared["prompt_id"] = prompt_id(key)
    return prepared


def sft_record(problem: str, answer: str) -> dict[str, list[dict[str, str]]]:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    return {
        "prompt": prompt_messages(problem),
        "completion": [{"role": "assistant", "content": rf"\boxed{{{answer.strip()}}}"}],
    }


def tokenize_generation_prompt(tokenizer, problem: str) -> dict:
    return tokenizer.apply_chat_template(
        prompt_messages(problem),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
