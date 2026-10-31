"""GRPO stub (not part of the v1 comparison).

v1 compares no-update, SFT and RFT only. A single-turn GRPO step on this task
format would reuse ``rollouts.jsonl`` groups (``group_size`` samples per task)
and ``trusted_full_v1`` rewards. Estimated peak memory for Qwen3-4B 4bit with
G=4, beta=0 (no reference model), prompt <= 2.5k tokens and completion <= 512
tokens is 8-12 GiB, close to the measured 10.5-13 GiB free on the target Mac,
so it is deferred until the SFT/RFT memory and time measurements exist.

Only the advantage computation is implemented, because group semantics must
stay intact whichever trainer is later used (AGENTS.md invariant 4).
"""
from __future__ import annotations

from typing import Mapping, Sequence


def group_advantages(rewards: Sequence[float], *, normalize_std: bool = False) -> list[float]:
    """Group-relative advantages. All-equal groups return zeros (no signal)."""
    if not rewards:
        raise ValueError("empty group")
    mean = sum(rewards) / len(rewards)
    adv = [r - mean for r in rewards]
    if normalize_std:
        var = sum(a * a for a in adv) / len(adv)
        std = var ** 0.5
        adv = [a / std if std > 0 else 0.0 for a in adv]
    return adv


def informative_groups(rollouts: Sequence[Mapping[str, object]]) -> dict[str, list[float]]:
    """Task -> advantages for groups with reward variance (the only groups a
    GRPO update could learn from). Used to report GRPO feasibility from the
    RFT rollouts without running GRPO."""
    groups: dict[str, list[float]] = {}
    for r in rollouts:
        groups.setdefault(str(r["instance_id"]), []).append(float(r["reward"]))  # type: ignore[arg-type]
    return {k: group_advantages(v) for k, v in groups.items() if len(set(v)) > 1}


def train_step(*_args, **_kwargs):  # pragma: no cover - deliberate stub
    raise NotImplementedError("GRPO is out of scope for coding-policy-learning v1")
