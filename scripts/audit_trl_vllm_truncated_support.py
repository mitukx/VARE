#!/usr/bin/env python3
"""Exact finite-support audit of TRL's vLLM sampling correction semantics."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/trl_vllm_truncated_support_audit_v1.lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def support_for(probabilities: list[Fraction], transform: dict) -> list[int]:
    name = transform["name"]
    if name == "identity":
        return list(range(len(probabilities)))
    order = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    if name == "top_k":
        return order[: transform["parameters"]["top_k"]]
    if name == "top_p":
        threshold = Fraction(transform["parameters"]["top_p"])
        chosen = []
        cumulative = Fraction(0)
        for index in order:
            chosen.append(index)
            cumulative += probabilities[index]
            if cumulative >= threshold:
                break
        return chosen
    if name == "min_p":
        threshold = Fraction(transform["parameters"]["min_p"]) * max(probabilities)
        return [index for index, probability in enumerate(probabilities) if probability >= threshold]
    raise ValueError(f"unsupported frozen transform: {name}")


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def analyze_context(name: str, raw: list[str], transforms: list[dict], reward: list[int]) -> dict:
    probabilities = [Fraction(value) for value in raw]
    if sum(probabilities) != 1 or len(probabilities) != len(reward):
        raise ValueError(f"invalid frozen probabilities or reward for {name}")
    rows = []
    for transform in transforms:
        support = support_for(probabilities, transform)
        retained_mass = sum((probabilities[index] for index in support), Fraction(0))
        q = {
            index: probabilities[index] / retained_mass
            for index in support
        }
        ratios = {
            index: probabilities[index] / q[index]
            for index in support
        }
        behavior_mean_ratio = sum(
            (q[index] * ratios[index] for index in support), Fraction(0)
        )
        raw_reward = sum(
            (probabilities[index] * reward[index] for index in range(len(probabilities))),
            Fraction(0),
        )
        corrected_sample_reward = sum(
            (q[index] * ratios[index] * reward[index] for index in support),
            Fraction(0),
        )
        p3 = probabilities[2]
        score_corrected_gradient = sum(
            (
                q[index]
                * ratios[index]
                * reward[index]
                * ((Fraction(1) if index == 2 else Fraction(0)) - p3)
                for index in support
            ),
            Fraction(0),
        )
        rows.append(
            {
                "transform": transform["name"],
                "support": support,
                "retained_raw_mass": fraction_text(retained_mass),
                "processed_q": [fraction_text(q.get(index, Fraction(0))) for index in range(len(probabilities))],
                "source_ratio_p_over_q": [fraction_text(ratios[index]) for index in support],
                "behavior_expected_source_ratio": fraction_text(behavior_mean_ratio),
                "abs_log_probability_difference": -math.log(float(retained_mass)),
                "raw_policy_reward_expectation": fraction_text(raw_reward),
                "source_corrected_sample_reward_expectation": fraction_text(corrected_sample_reward),
                "raw_policy_reward_gradient_theta": fraction_text(p3 * (1 - p3)),
                "source_corrected_sample_score_gradient_theta": fraction_text(score_corrected_gradient),
            }
        )
    return {"context": name, "raw_p": raw, "results": rows}


def run(source_root: Path) -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    provenance = {}
    for relative, expected in lock["source"]["files"].items():
        found = sha256(source_root / relative)
        if found != expected:
            raise ValueError(f"pinned source hash mismatch for {relative}: {found}")
        provenance[relative] = found

    contexts = lock["frozen_exact_inputs"]["contexts"]
    transforms = lock["frozen_exact_inputs"]["sampling_transforms"]
    reward = lock["frozen_exact_inputs"]["support_omission_reward"]
    results = [
        analyze_context(row["name"], row["p"], transforms, reward)
        for row in contexts
    ]
    c1_top_p = next(row for row in results[0]["results"] if row["transform"] == "top_p")
    sequence_ratio = Fraction(c1_top_p["retained_raw_mass"]) ** 2
    expected = {
        "identity": ("1", ["1", "1", "1"]),
        "top_p": ("3/4", ["4/5", "9/10"]),
        "top_k": ("2", ["4/5", "9/10"]),
        "min_p": ("1/2", ["4/5", "7/10"]),
    }
    actual = {}
    for transform in transforms:
        actual[transform["name"]] = [
            next(row for row in context["results"] if row["transform"] == transform["name"])["retained_raw_mass"]
            for context in results
        ]
    if actual != {key: list(value[1]) if key != "identity" else ["1", "1"] for key, value in expected.items()}:
        raise AssertionError(f"support masses differ from frozen predictions: {actual}")
    if sequence_ratio != Fraction(16, 25):
        raise AssertionError(f"sequence-level source ratio differs: {sequence_ratio}")
    for context in results:
        for row in context["results"]:
            if row["behavior_expected_source_ratio"] != row["retained_raw_mass"]:
                raise AssertionError("E_q[p/q] did not equal retained support mass")

    return {
        "protocol_id": lock["protocol_id"],
        "status": "pass",
        "source_revision": lock["source"]["revision"],
        "source_sha256": provenance,
        "results": results,
        "sequence_case": {
            "per_position_retained_mass": c1_top_p["retained_raw_mass"],
            "two_position_source_ratio": fraction_text(sequence_ratio),
        },
        "decision": "The pinned source ratio equals p/q on sampled support. Truncation makes q zero outside that support, so this ratio cannot recover arbitrary raw-policy expectations; the exact counterexample applies to the raw-policy estimand, not to every possible intended objective.",
        "claim_limits": lock["claim_limits"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="pinned TRL checkout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
