"""Post hoc source-level counterexample for declared promotion slice coverage."""
from dataclasses import fields
import json

from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.types import EvaluationReport


def decide(slices, required):
    kwargs = {"min_primary_gain": 0.01, "min_eval_examples": 1}
    if "required_slice_names" in {item.name for item in fields(PromotionConfig)}:
        kwargs["required_slice_names"] = required
    gate = PromotionGate(PromotionConfig(**kwargs))
    incumbent = EvaluationReport("incumbent", 0.50, slices[0], n=100)
    candidate = EvaluationReport("candidate", 0.80, slices[1], n=100)
    decision = gate.decide(incumbent, candidate)
    return {"accepted": decision.accepted, "reasons": list(decision.reasons)}


print(json.dumps({
    "omitted_by_both": decide(
        ({"easy": 0.50}, {"easy": 0.80}), ("easy", "hard")
    ),
    "declared_complete_control": decide(
        ({"easy": 0.50, "hard": 0.40}, {"easy": 0.80, "hard": 0.40}),
        ("easy", "hard"),
    ),
}, sort_keys=True))
