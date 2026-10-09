#!/usr/bin/env python3
"""Recompute a frozen source/formula audit of current TRL AsyncGRPO normalization.

This intentionally does not import TRL or claim an end-to-end Trainer reproduction.
It pins the inspected production source and evaluates the exact normalization
expressions against a scalar token-gradient fixture.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/trl_async_window_normalization_current_main_v1.lock.json"
SOURCE = ROOT / "results/trl-async-window-normalization-current-main-v1/source/async_grpo_trainer.py"
BASE_SOURCE = ROOT / "results/trl-async-window-normalization-current-main-v1/source/base_trainer.py"
OUTPUT = ROOT / "results/trl-async-window-normalization-current-main-v1/run-1/summary.json"


def accumulated_microbatch_gradient(microbatches: list[dict[str, float]], steps: int) -> float:
    """Current source: each local sum is divided by its own active-token count, then by K."""
    total = 0.0
    for batch in microbatches:
        token_denominator = max(batch["active_tokens"], 1.0)
        total += batch["summed_token_gradient"] / token_denominator / steps
    return total


def pooled_token_gradient(microbatches: list[dict[str, float]]) -> float:
    numerator = sum(batch["summed_token_gradient"] for batch in microbatches)
    denominator = sum(batch["active_tokens"] for batch in microbatches)
    return numerator / denominator if denominator else 0.0


def evaluate_case(microbatches: list[dict[str, float]], steps: int, learning_rate: float) -> dict:
    current = accumulated_microbatch_gradient(microbatches, steps)
    pooled = pooled_token_gradient(microbatches)
    return {
        "current_main_gradient": current,
        "pooled_token_gradient": pooled,
        "absolute_gradient_error": abs(current - pooled),
        "gradient_ratio_to_pooled": current / pooled if pooled else None,
        "current_main_sgd_delta": -learning_rate * current,
        "pooled_token_sgd_delta": -learning_rate * pooled,
        "absolute_sgd_delta_error": learning_rate * abs(current - pooled),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    source_bytes = SOURCE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    expected_hash = protocol["source"]["sha256"]
    if source_hash != expected_hash:
        raise SystemExit(f"source hash mismatch: expected {expected_hash}, got {source_hash}")
    source = source_bytes.decode()
    required = [
        "tokens_per_rank = (global_n_tokens / world_size).clamp(min=1.0)",
        "loss = loss / tokens_per_rank.to(torch.float32)",
        "loss = loss / self.current_gradient_accumulation_steps",
    ]
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise SystemExit(f"pinned source no longer contains expected formula fragments: {missing}")
    base_bytes = BASE_SOURCE.read_bytes()
    base_hash = hashlib.sha256(base_bytes).hexdigest()
    expected_base_hash = protocol["trainer_scaling_source"]["sha256"]
    if base_hash != expected_base_hash:
        raise SystemExit(f"base trainer hash mismatch: expected {expected_base_hash}, got {base_hash}")
    if "loss_is_scaled_for_ga = True" not in source:
        raise SystemExit("AsyncGRPOTrainer no longer declares its loss as scaled for accumulation")
    if "loss_is_scaled_for_ga" not in base_bytes.decode():
        raise SystemExit("pinned _BaseTrainer scaling contract is absent")

    fixture = protocol["fixed_fixture"]
    negative = protocol["negative_control"]
    counterexample = evaluate_case(
        fixture["microbatches"], fixture["accumulation_steps"], fixture["learning_rate"]
    )
    control = evaluate_case(negative["microbatches"], fixture["accumulation_steps"], fixture["learning_rate"])
    result = {
        "study_id": protocol["study_id"],
        "source_commit": protocol["source"]["commit"],
        "source_sha256": source_hash,
        "trainer_scaling_source_sha256": base_hash,
        "source_formula_fragments_verified": required,
        "counterexample": counterexample,
        "negative_control": control,
        "decision": "source_formula counterexample reproduced" if counterexample["absolute_gradient_error"] > 0 and math.isclose(control["absolute_gradient_error"], 0.0) else "protocol failure",
        "claim_boundary": "Exact arithmetic from the pinned source formula; not an end-to-end Trainer, DDP, model-update, or capability experiment.",
    }
    if result["decision"] != "source_formula counterexample reproduced":
        raise SystemExit("frozen acceptance rule failed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
