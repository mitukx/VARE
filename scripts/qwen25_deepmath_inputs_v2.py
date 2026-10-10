"""Shared conversational records for the pinned DeepMath GRPO/SFT protocol."""

from __future__ import annotations


def prompt_messages(problem: str) -> list[dict[str, str]]:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty string")
    # DeepMath-103K stores exactly this one user message. Keep it unchanged so
    # TRL's GRPO prompt path and the base/evaluation gates receive the same chat.
    return [{"role": "user", "content": problem}]


def sft_example(problem: str, answer: str) -> dict[str, list[dict[str, str]]]:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    return {
        "prompt": prompt_messages(problem),
        "completion": [{"role": "assistant", "content": rf"\boxed{{{answer.strip()}}}"}],
    }


def tokenize_generation_prompt(tokenizer, problem: str):
    """Mirror TRL 1.1.0's conversational prompt template call for one row."""
    return tokenizer.apply_chat_template(
        prompt_messages(problem),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
