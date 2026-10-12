"""Strict and lenient integer extractors.

The strict rule is the frozen final line ``#### <integer>``. The lenient rule
is pre-declared: the last ``\\boxed{integer}`` if one exists, otherwise the
last integer token. Neither rule replaces a base-gate decision.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from vare.heldout_exact_integer import evaluate_expression, parse_final_integer

_BOXED = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
_INTEGER = re.compile(r"(?<!\d)-?\d+(?!\d)")


def lenient_integer(text: str) -> int | None:
    boxed = _BOXED.findall(text)
    if boxed:
        return int(boxed[-1])
    tokens = _INTEGER.findall(text)
    if not tokens:
        return None
    return int(tokens[-1])


def score_output(expression: str, text: str) -> dict[str, bool]:
    truth = evaluate_expression(expression)
    strict = parse_final_integer(text)
    extracted = lenient_integer(text)
    return {
        "lenient_correct": extracted == truth,
        "lenient_extracted": extracted is not None,
        "strict_correct": strict == truth,
        "strict_format": strict is not None,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def summarize_two_tier(scores: Sequence[Mapping[str, bool]]) -> dict[str, object]:
    if not scores:
        raise ValueError("two-tier summary requires rows")
    total = len(scores)
    compliant = sum(bool(row["strict_format"]) for row in scores)
    strict_correct = sum(bool(row["strict_correct"]) for row in scores)
    extracted = sum(bool(row["lenient_extracted"]) for row in scores)
    lenient_correct = sum(bool(row["lenient_correct"]) for row in scores)
    return {
        "conditional_accuracy": _rate(strict_correct, compliant),
        "format_compliance_rate": compliant / total,
        "format_compliant": compliant,
        "gate_result": False,
        "lenient_accuracy_counting_miss_as_wrong": lenient_correct / total,
        "lenient_conditional_accuracy": _rate(lenient_correct, extracted),
        "lenient_correct": lenient_correct,
        "lenient_extracted": extracted,
        "n": total,
        "posthoc": True,
        "strict_correct": strict_correct,
    }
