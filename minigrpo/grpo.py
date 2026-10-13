"""Group-relative policy objective.

Advantages are standardized inside each intact group. A tied group has
advantage 0 and is kept. The token loss is a PPO clip. Optional reference KL
uses the Schulman k3 estimator. Dr.GRPO mode skips the within-group standard
deviation and does not divide token losses by response length.

Log-ratio differences are clamped to ``[-20, 20]`` before ``exp`` so a single
overflowing token cannot make the step non-finite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from minigrpo.expr import evaluate_expression, parse_final_integer

LOG_RATIO_LIMIT = 20.0
ADVANTAGE_EPS = 1e-8
LORA_KEYS = ("self_attn.q_proj", "self_attn.v_proj")


@dataclass(frozen=True, slots=True)
class TokenRow:
    logp: tuple[float, ...]
    logp_old: tuple[float, ...]
    mask: tuple[int, ...]
    advantage: float
    logp_ref: tuple[float, ...] | None = None
    entropy: tuple[float, ...] | None = None


def group_advantages(
    rewards: list[float] | tuple[float, ...],
    group_size: int,
    *,
    dr_grpo: bool = False,
    eps: float = ADVANTAGE_EPS,
) -> list[float]:
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if not rewards or len(rewards) % group_size != 0:
        raise ValueError("rewards must be a positive multiple of group_size")
    if any(isinstance(reward, bool) or not isinstance(reward, (int, float)) for reward in rewards):
        raise ValueError("rewards must be finite numbers")
    if any(not math.isfinite(float(reward)) for reward in rewards):
        raise ValueError("rewards must be finite numbers")
    advantages: list[float] = []
    for start in range(0, len(rewards), group_size):
        group = [float(reward) for reward in rewards[start : start + group_size]]
        mean = sum(group) / group_size
        if dr_grpo:
            advantages.extend(reward - mean for reward in group)
            continue
        variance = sum((reward - mean) ** 2 for reward in group) / group_size
        std = math.sqrt(variance)
        if std == 0.0:
            advantages.extend(0.0 for _ in group)
        else:
            scale = std + eps
            advantages.extend((reward - mean) / scale for reward in group)
    if len(advantages) != len(rewards):
        raise RuntimeError("group was truncated")
    return advantages


def importance_ratio(logp: float, logp_old: float) -> float:
    delta = min(max(logp - logp_old, -LOG_RATIO_LIMIT), LOG_RATIO_LIMIT)
    return math.exp(delta)


def reference_kl(logp: float, logp_ref: float) -> float:
    delta = min(max(logp_ref - logp, -LOG_RATIO_LIMIT), LOG_RATIO_LIMIT)
    return math.exp(delta) - delta - 1.0


def clipped_surrogate(logp: float, logp_old: float, advantage: float, clip_eps: float) -> float:
    ratio = importance_ratio(logp, logp_old)
    unclipped = ratio * advantage
    clipped_ratio = min(max(ratio, 1.0 - clip_eps), 1.0 + clip_eps)
    clipped = clipped_ratio * advantage
    return -min(unclipped, clipped)


def sequence_loss(
    rows: tuple[TokenRow, ...] | list[TokenRow],
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
    dr_grpo: bool = False,
) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("loss requires an intact group")
    if not 0.0 < clip_eps < 1.0:
        raise ValueError("clip_eps must lie in (0, 1)")
    if kl_coef < 0.0:
        raise ValueError("kl_coef must be non-negative")
    total = 0.0
    kl_sum = 0.0
    kl_count = 0
    entropy_sum = 0.0
    entropy_count = 0
    tokens = 0
    for row in rows:
        if len({len(row.logp), len(row.logp_old), len(row.mask)}) != 1:
            raise ValueError("logp, logp_old, and mask lengths differ")
        if row.logp_ref is not None and len(row.logp_ref) != len(row.logp):
            raise ValueError("reference logp length differs")
        if kl_coef > 0.0 and row.logp_ref is None:
            raise ValueError("kl_coef requires reference log probabilities")
        active = [index for index, bit in enumerate(row.mask) if bit]
        if not active:
            raise ValueError("completion mask is empty")
        token_losses: list[float] = []
        for index in active:
            token_loss = clipped_surrogate(row.logp[index], row.logp_old[index], row.advantage, clip_eps)
            if row.logp_ref is not None:
                kl = reference_kl(row.logp[index], row.logp_ref[index])
                kl_sum += kl
                kl_count += 1
                if kl_coef:
                    token_loss += kl_coef * kl
            token_losses.append(token_loss)
            if row.entropy is not None:
                entropy_sum += row.entropy[index]
                entropy_count += 1
        tokens += len(active)
        total += sum(token_losses) if dr_grpo else sum(token_losses) / len(active)
    return {
        "entropy_mean": None if entropy_count == 0 else entropy_sum / entropy_count,
        "kl_mean": 0.0 if kl_count == 0 else kl_sum / kl_count,
        "loss": total / len(rows),
        "sequence_count": len(rows),
        "token_count": tokens,
    }


def correctness_reward(expression: str, text: str) -> float:
    parsed = parse_final_integer(text)
    if parsed is None:
        return 0.0
    try:
        truth = evaluate_expression(expression)
    except ValueError:
        return 0.0
    return 1.0 if parsed == truth else 0.0


def format_reward(text: str) -> float:
    return 1.0 if parse_final_integer(text) is not None else 0.0


def incorrect_reward(expression: str, text: str) -> float:
    """1 when the strict answer is wrong. A missing final line counts as wrong."""
    return 1.0 - correctness_reward(expression, text)


def reward_value(mode: str, expression: str, text: str) -> float:
    if mode == "correctness":
        return correctness_reward(expression, text)
    if mode == "format":
        return format_reward(text)
    raise ValueError(f"unknown reward: {mode}")


def cpu_policy_step() -> dict[str, float]:
    """One finite-difference update on a two-member group. No model weights."""
    advantages = group_advantages([1.0, 0.0], 2)
    if len(advantages) != 2:
        raise RuntimeError("cpu smoke dropped a group member")

    def loss_at(theta: float) -> float:
        rows = [
            TokenRow(logp=(theta,), logp_old=(0.0,), mask=(1,), advantage=advantages[0]),
            TokenRow(logp=(0.0,), logp_old=(0.0,), mask=(1,), advantage=advantages[1]),
        ]
        loss = sequence_loss(rows)["loss"]
        if not isinstance(loss, float):
            raise RuntimeError("cpu smoke loss was not a float")
        return loss

    before = loss_at(0.0)
    grad = (loss_at(1e-4) - loss_at(-1e-4)) / 2e-4
    theta = -0.05 * grad
    after = loss_at(theta)
    if not (theta > 0.0 and after < before and math.isfinite(after)):
        raise RuntimeError("cpu smoke did not increase the preferred completion")
    return {"grad": grad, "loss_after": after, "loss_before": before, "theta": theta}
