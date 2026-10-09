#!/usr/bin/env python3
"""Independent integer-arithmetic verification of the frozen truncation audit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/trl_vllm_truncated_support_audit_v1.lock.json"


def keep_indices(weights: list[int], transform: dict) -> list[int]:
    total = sum(weights)
    ordered = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    kind = transform["name"]
    if kind == "identity":
        return list(range(len(weights)))
    if kind == "top_k":
        return ordered[: int(transform["parameters"]["top_k"])]
    if kind == "top_p":
        numerator, denominator = map(int, transform["parameters"]["top_p"].split("/"))
        selected = []
        cumulative = 0
        for index in ordered:
            selected.append(index)
            cumulative += weights[index]
            if cumulative * denominator >= total * numerator:
                return selected
        return selected
    if kind == "min_p":
        numerator, denominator = map(int, transform["parameters"]["min_p"].split("/"))
        maximum = max(weights)
        return [i for i, value in enumerate(weights) if value * denominator >= maximum * numerator]
    raise AssertionError(f"unknown transform {kind}")


def ftext(numerator: int, denominator: int) -> str:
    from math import gcd

    common = gcd(numerator, denominator)
    numerator //= common
    denominator //= common
    return str(numerator) if denominator == 1 else f"{numerator}/{denominator}"


def independently_derive(lock: dict) -> dict:
    transforms = lock["frozen_exact_inputs"]["sampling_transforms"]
    contexts = lock["frozen_exact_inputs"]["contexts"]
    output = {}
    for context in contexts:
        # The locked probabilities have common denominator ten. Work only on
        # integer weights so this checker shares no probability-normalization
        # code with the Fraction-based primary runner.
        weights = [int(FractionString.split("/")[0]) * (10 // int(FractionString.split("/")[1])) if "/" in FractionString else int(FractionString) * 10 for FractionString in context["p"]]
        total = sum(weights)
        if total != 10:
            raise AssertionError("the independent checker expects the frozen denominator-10 input")
        rows = {}
        for transform in transforms:
            support = keep_indices(weights, transform)
            z = sum(weights[i] for i in support)
            # For every retained action (w/10) / (w/z) == z/10.
            ratios = [ftext(z, total) for _ in support]
            rows[transform["name"]] = {
                "support": support,
                "retained_mass": ftext(z, total),
                "ratios": ratios,
                "mean_ratio_under_behavior": ftext(z, total),
            }
        output[context["name"]] = rows
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    independent = independently_derive(lock)
    for context in result["results"]:
        for row in context["results"]:
            other = independent[context["context"]][row["transform"]]
            if row["support"] != other["support"]:
                raise AssertionError("independent support calculation disagrees")
            if row["retained_raw_mass"] != other["retained_mass"]:
                raise AssertionError("independent retained mass disagrees")
            if row["source_ratio_p_over_q"] != other["ratios"]:
                raise AssertionError("independent ratio calculation disagrees")
            if row["behavior_expected_source_ratio"] != other["mean_ratio_under_behavior"]:
                raise AssertionError("independent expectation calculation disagrees")
            # With R=[0,0,1] and theta added only to logit 3, the score is
            # 1-p3 only when action 3 is sampled; all other rewarded terms are 0.
            # Compute that contribution directly from integer weights.
            weights = [
                int(value.split("/")[0]) * (10 // int(value.split("/")[1])) if "/" in value else int(value) * 10
                for value in context["raw_p"]
            ]
            raw_score_gradient = ftext(weights[2] * (sum(weights) - weights[2]), sum(weights) ** 2)
            sampled_score_gradient = raw_score_gradient if 2 in row["support"] else "0"
            raw_reward = ftext(weights[2], sum(weights))
            sampled_reward = raw_reward if 2 in row["support"] else "0"
            if row["raw_policy_reward_gradient_theta"] != raw_score_gradient:
                raise AssertionError("raw-policy score gradient does not match independent formula")
            if row["source_corrected_sample_score_gradient_theta"] != sampled_score_gradient:
                raise AssertionError("source-corrected score gradient does not match independent formula")
            if row["raw_policy_reward_expectation"] != raw_reward:
                raise AssertionError("raw-policy reward expectation does not match independent formula")
            if row["source_corrected_sample_reward_expectation"] != sampled_reward:
                raise AssertionError("sampled corrected reward expectation does not match independent formula")
            expected_q = [
                ftext(weights[index], sum(weights[i] for i in row["support"]))
                if index in row["support"]
                else "0"
                for index in range(len(weights))
            ]
            if row["processed_q"] != expected_q:
                raise AssertionError("processed behavior distribution does not match independent formula")
            expected_log_gap = -math.log(sum(weights[i] for i in row["support"]) / sum(weights))
            if not math.isclose(row["abs_log_probability_difference"], expected_log_gap, rel_tol=1e-14, abs_tol=1e-14):
                raise AssertionError("absolute log-probability gap does not match independent formula")
    if result["sequence_case"]["two_position_source_ratio"] != "16/25":
        raise AssertionError("sequence product does not match the exact prediction")
    output = {
        "protocol_id": lock["protocol_id"],
        "status": "pass",
        "method": "independent integer weights; no import from the primary runner",
        "independently_derived": independent,
        "source_corrected_raw_policy_reward": "0 under truncated q; matches raw-policy expectation only when the rewarded action remains in support",
        "raw_policy_reward_gradient_theta": "4/25 for c1; 9/100 for c2 when action 3 remains in support, otherwise the sampled estimate is 0",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    from fractions import Fraction

    raise SystemExit(main())
