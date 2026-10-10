"""Pure-Python accounting for the frozen Qwen DeepMath inference gate."""

from __future__ import annotations

from collections.abc import Iterable


def completion_shape(token_ids: Iterable[int], eos_token_id: int | None, cap: int) -> dict:
    """Return token accounting from generated IDs, counting first EOS only."""
    ids = [int(token_id) for token_id in token_ids]
    count = 0
    eos_reached = False
    for token_id in ids:
        count += 1
        if eos_token_id is not None and token_id == eos_token_id:
            eos_reached = True
            break
    return {
        "generated_token_count": count,
        "eos_reached": eos_reached,
        "truncated": count >= cap and not eos_reached,
    }


def summarize_completions(completions: list[dict]) -> dict:
    """Summarize gate outcomes using per-completion retained observations."""
    if not completions:
        raise ValueError("cannot summarize an empty completion set")
    lengths = [int(item["generated_token_count"]) for item in completions]
    eos_count = sum(bool(item["eos_reached"]) for item in completions)
    truncated_count = sum(bool(item["truncated"]) for item in completions)
    disagreement_count = sum(
        bool(item["training_reward"]) != bool(item["independent_task_success"])
        for item in completions
    )
    return {
        "completion_count": len(completions),
        "completion_tokens_mean": sum(lengths) / len(lengths),
        "completion_tokens_min": min(lengths),
        "completion_tokens_max": max(lengths),
        "eos_reached_count": eos_count,
        "eos_reached_rate": eos_count / len(completions),
        "truncation_count": truncated_count,
        "truncation_rate": truncated_count / len(completions),
        "training_reward_independent_checker_disagreement_count": disagreement_count,
    }
