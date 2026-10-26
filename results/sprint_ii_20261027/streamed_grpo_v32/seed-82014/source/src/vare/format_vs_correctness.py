"""Refusal for a five-arm format-versus-correctness comparison.

The exact-integer v2 base gate authorizes only the separate one-update smoke.
This module does not read a smoke result and does not start an arm.
"""

from __future__ import annotations

PROTOCOL_ID = "format-vs-correctness-comparison-v1"
PROTOCOL_SHA256 = "fcc8b9abcc29287022f15aeb72876a41f7a75036da28b59ced2a0f23d85edeae"
ARMS = (
    "base",
    "format_sft",
    "format_reward_rl",
    "correctness_reward_rl",
    "random_reward_rl",
)


def comparison_refusal() -> dict[str, object]:
    return {
        "arms": list(ARMS),
        "authorized": False,
        "minimum_seeds": 5,
        "optimizer_steps": "not_frozen",
        "primary_contrast": (
            "correctness_reward_rl minus format_sft, strict pass@1 on the "
            "committed confirm seeds, paired bootstrap"
        ),
        "protocol_id": PROTOCOL_ID,
        "reason": (
            "The five-arm comparison is not started. The exact-integer v2 base "
            "gate authorizes only the separate one-update save/reload smoke. A "
            "retained smoke is not an input to this entry point. The optimizer "
            "step count is not frozen. Confirm seeds stay out of training."
        ),
        "status": "refused",
    }
