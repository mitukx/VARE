#!/usr/bin/env python3
"""Post-hoc same-host audit of the exploratory GRPO omission development screen.

This implementation reads the retained law panel and summary, reconstructs the
four-member GRPO target, Walsh coefficients, audit estimators, and fixed-budget
moments directly, and fails if the retained summary disagrees. It deliberately
does not import the exploratory runner or its estimator helpers. This is not an
external replication and does not establish preregistered evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    ROOT
    / "results/grpo-action-conditioned-omission-v1/development-seed-20261016"
)
BITS = tuple(itertools.product((0, 1), repeat=4))
INDICES = tuple(range(4))
ARMS = ("m1", "m2", "m3_uniform", "m3_structured", "m4_full")
BUDGET_GROUPS = {
    "m1": 12,
    "m2": 6,
    "m3_uniform": 4,
    "m3_structured": 4,
    "m4_full": 3,
}
TOL = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_probability(action: tuple[int, ...]) -> float:
    return (0.25 ** sum(action)) * (0.75 ** (4 - sum(action)))


def character(labels: tuple[int, ...], support: tuple[int, ...]) -> int:
    return -1 if sum(labels[index] for index in support) % 2 else 1


def standardized_advantages(labels: tuple[int, ...]) -> tuple[float, ...]:
    """Population-standardized binary group rewards; constants map to zero."""
    positive = sum(labels)
    if positive in (0, 4):
        return (0.0, 0.0, 0.0, 0.0)
    mean = positive / 4.0
    standard_deviation = math.sqrt(mean * (1.0 - mean))
    return tuple((value - mean) / standard_deviation for value in labels)


def clean_update(action: tuple[int, ...], labels: tuple[int, ...]) -> float:
    advantages = standardized_advantages(labels)
    return sum(
        advantage * (sampled_action - 0.25)
        for advantage, sampled_action in zip(advantages, action, strict=True)
    )


def walsh_coefficients(action: tuple[int, ...]) -> dict[tuple[int, ...], float]:
    coefficients: dict[tuple[int, ...], float] = {}
    for order in range(5):
        for support in itertools.combinations(INDICES, order):
            coefficients[support] = sum(
                clean_update(action, labels) * character(labels, support)
                for labels in BITS
            ) / 16.0
    return coefficients


def uniform_inclusion_probabilities(size: int) -> dict[tuple[int, ...], float]:
    denominator = math.comb(4, size)
    result: dict[tuple[int, ...], float] = {}
    for order in range(size + 1):
        for support in itertools.combinations(INDICES, order):
            result[support] = math.comb(4 - order, size - order) / denominator
    return result


def omission_probabilities(
    omission: tuple[float, ...],
) -> dict[tuple[int, ...], float]:
    return {
        support: sum(omission[index] for index in INDICES if index not in support)
        for order in range(4)
        for support in itertools.combinations(INDICES, order)
    }


def weighted_omission_distribution(
    coefficients: dict[tuple[int, ...], float],
) -> tuple[float, ...]:
    weights = [
        abs(coefficients[tuple(index for index in INDICES if index != omitted)])
        for omitted in INDICES
    ]
    total = sum(weights)
    if total == 0.0:
        return (0.25, 0.25, 0.25, 0.25)
    return tuple(1.0 / 16.0 + 0.75 * weight / total for weight in weights)


def partial_estimate(
    labels: tuple[int, ...],
    observed: tuple[int, ...],
    coefficients: dict[tuple[int, ...], float],
    inclusion: dict[tuple[int, ...], float],
) -> float:
    estimate = 0.0
    for order in range(len(observed) + 1):
        for support in itertools.combinations(observed, order):
            probability = inclusion[support]
            if probability <= 0.0:
                raise ValueError(f"nonpositive inclusion probability for {support}")
            estimate += (
                coefficients[support]
                * character(labels, support)
                / probability
            )
    return estimate


def decode_panel_law(law: dict[str, Any]) -> list[list[float]]:
    encoded = law["conditional_probabilities"]
    if len(encoded) != 16 or any(len(row) != 16 for row in encoded):
        raise ValueError(f"{law.get('law_id')}: expected a 16-by-16 conditional table")
    decoded: list[list[float]] = []
    for action_index, row in enumerate(encoded):
        probabilities = []
        for pair in row:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"{law.get('law_id')}: malformed probability at row {action_index}")
            numerator, denominator = map(int, pair)
            if denominator <= 0 or numerator < 0 or numerator > denominator:
                raise ValueError(f"{law.get('law_id')}: invalid probability fraction {pair}")
            probabilities.append(numerator / denominator)
        if abs(sum(probabilities) - 1.0) > TOL:
            raise ValueError(f"{law.get('law_id')}: conditional row {action_index} does not sum to one")
        decoded.append(probabilities)
    return decoded


def calculate_law(law: dict[str, Any]) -> dict[str, Any]:
    probabilities = decode_panel_law(law)
    moments = {arm: [0.0, 0.0] for arm in ARMS}
    pointwise_unbiased = True

    for action_index, action in enumerate(BITS):
        action_weight = action_probability(action)
        coefficients = walsh_coefficients(action)
        # The order-four coefficient must vanish for this target. If not, the
        # three-label omission estimator is not pointwise unbiased.
        if abs(coefficients[INDICES]) > TOL:
            raise ValueError(f"{law['law_id']}: nonzero order-four coefficient")

        q = weighted_omission_distribution(coefficients)
        if any(value <= 0.0 for value in q) or abs(sum(q) - 1.0) > TOL:
            raise ValueError(f"{law['law_id']}: invalid omission distribution {q}")
        structured_inclusion = omission_probabilities(q)
        uniform_three_inclusion = omission_probabilities((0.25, 0.25, 0.25, 0.25))
        uniform_inclusion = {
            size: uniform_inclusion_probabilities(size) for size in (1, 2)
        }

        # Verify unbiasedness independently of the law's support: for every
        # possible clean reward vector, averaging over omissions reconstructs
        # the complete-group update.
        for labels in BITS:
            structured_mean = 0.0
            uniform_mean = 0.0
            for omitted in INDICES:
                observed = tuple(index for index in INDICES if index != omitted)
                structured_mean += q[omitted] * partial_estimate(
                    labels, observed, coefficients, structured_inclusion
                )
                uniform_mean += 0.25 * partial_estimate(
                    labels, observed, coefficients, uniform_three_inclusion
                )
            target = clean_update(action, labels)
            pointwise_unbiased &= (
                abs(structured_mean - target) <= TOL
                and abs(uniform_mean - target) <= TOL
            )

        for labels_index, labels in enumerate(BITS):
            label_weight = action_weight * probabilities[action_index][labels_index]
            if label_weight == 0.0:
                continue
            target = clean_update(action, labels)

            for size in (1, 2):
                subset_values = []
                for observed in itertools.combinations(INDICES, size):
                    subset_values.append(
                        partial_estimate(
                            labels,
                            observed,
                            coefficients,
                            uniform_inclusion[size],
                        )
                    )
                mean = sum(subset_values) / len(subset_values)
                second = sum(value * value for value in subset_values) / len(subset_values)
                moments[f"m{size}"][0] += label_weight * mean
                moments[f"m{size}"][1] += label_weight * second

            uniform_values = []
            structured_values = []
            for omitted in INDICES:
                observed = tuple(index for index in INDICES if index != omitted)
                uniform_values.append(
                    partial_estimate(
                        labels, observed, coefficients, uniform_three_inclusion
                    )
                )
                structured_values.append(
                    partial_estimate(
                        labels, observed, coefficients, structured_inclusion
                    )
                )
            moments["m3_uniform"][0] += label_weight * sum(uniform_values) / 4.0
            moments["m3_uniform"][1] += label_weight * sum(
                value * value for value in uniform_values
            ) / 4.0
            moments["m3_structured"][0] += label_weight * sum(
                q[j] * structured_values[j] for j in INDICES
            )
            moments["m3_structured"][1] += label_weight * sum(
                q[j] * structured_values[j] * structured_values[j]
                for j in INDICES
            )
            moments["m4_full"][0] += label_weight * target
            moments["m4_full"][1] += label_weight * target * target

    means = {arm: pair[0] for arm, pair in moments.items()}
    full_mse = None
    rows: dict[str, Any] = {}
    for arm, (mean, second) in moments.items():
        variance = max(0.0, second - mean * mean)
        bias = mean - means["m4_full"]
        fixed_budget_mse = variance / BUDGET_GROUPS[arm] + bias * bias
        rows[arm] = {
            "mean": mean,
            "bias": bias,
            "variance": variance,
            "fixed_budget_mse": fixed_budget_mse,
        }
        if arm == "m4_full":
            full_mse = fixed_budget_mse

    assert full_mse is not None
    for row in rows.values():
        row["ratio_to_full"] = row["fixed_budget_mse"] / full_mse if full_mse > 1e-20 else None

    return {
        "law_id": law["law_id"],
        "family": law["family"],
        "true_mean": means["m4_full"],
        "true_variance": max(
            0.0,
            moments["m4_full"][1] - means["m4_full"] ** 2,
        ),
        "arms": rows,
        "pointwise_m3_unbiased": pointwise_unbiased,
        "zero_full_mse": full_mse <= 1e-20,
    }


def assert_close(label: str, observed: Any, expected: Any, failures: list[str]) -> None:
    if observed is None or expected is None:
        if observed is not expected:
            failures.append(f"{label}: {observed!r} != {expected!r}")
        return
    if not math.isfinite(float(observed)) or not math.isfinite(float(expected)):
        failures.append(f"{label}: nonfinite value {observed!r} / {expected!r}")
    elif abs(float(observed) - float(expected)) > TOL:
        failures.append(f"{label}: {observed!r} != {expected!r}")


def audit(bundle: Path) -> tuple[dict[str, Any], list[str]]:
    panel_path = bundle / "law_panel.json"
    summary_path = bundle / "summary.json"
    manifest_path = bundle / "manifest.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    if sha256(panel_path) != manifest.get("law_panel_sha256"):
        failures.append("law_panel.json hash differs from manifest")
    if sha256(summary_path) != manifest.get("summary_sha256"):
        failures.append("summary.json hash differs from manifest")
    if sha256(bundle / "runner-as-run.py") != manifest.get(
        "retained_calculation_helper_snapshot_sha256"
    ):
        failures.append("runner-as-run.py hash differs from manifest")
    if len(panel) != 128:
        failures.append(f"law panel contains {len(panel)} laws, expected 128")
    if summary.get("law_count") != len(panel):
        failures.append("summary law_count differs from retained panel")
    if summary.get("status") != "exploratory-development-only-not-preregistered":
        failures.append("summary status does not identify the screen as exploratory")

    family_counts: dict[str, int] = {}
    recomputed: list[dict[str, Any]] = []
    for law in panel:
        family_counts[law["family"]] = family_counts.get(law["family"], 0) + 1
        recomputed.append(calculate_law(law))
    if family_counts != {
        "random_joint": 32,
        "independent": 32,
        "common_shock": 32,
        "single_stratum_parity": 32,
    }:
        failures.append(f"unexpected law-family counts: {family_counts}")

    retained_rows = summary.get("per_law")
    if not isinstance(retained_rows, list) or len(retained_rows) != len(recomputed):
        failures.append("summary per_law schema/count does not match the panel")
        retained_rows = []

    max_numeric_difference = 0.0
    pointwise_checks = 0
    structured_ratios: list[float] = []
    uniform_ratios: list[float] = []
    structured_uniform_ratios: list[float] = []
    structured_beats_full = 0
    structured_beats_uniform = 0
    structured_at_least_10pct_better = 0

    for index, row in enumerate(recomputed):
        if not row["pointwise_m3_unbiased"]:
            failures.append(f"{row['law_id']}: omission estimator failed pointwise unbiasedness")
        pointwise_checks += 16 * 16
        if row["zero_full_mse"]:
            failures.append(f"{row['law_id']}: unexpected zero full-audit MSE")
        if index >= len(retained_rows):
            continue
        reference = retained_rows[index]
        if reference.get("law_id") != row["law_id"] or reference.get("family") != row["family"]:
            failures.append(f"per-law row {index}: identity/family mismatch")
        if reference.get("pointwise_m3_unbiased") is not True:
            failures.append(f"{row['law_id']}: retained unbiasedness flag is not true")
        if reference.get("zero_full_mse") is not False:
            failures.append(f"{row['law_id']}: retained zero-full-MSE flag differs")

        for key in ("true_mean", "true_variance"):
            prior_failures = len(failures)
            assert_close(f"{row['law_id']} {key}", row[key], reference.get(key), failures)
            if len(failures) == prior_failures:
                max_numeric_difference = max(
                    max_numeric_difference,
                    abs(float(row[key]) - float(reference[key])),
                )
        for arm in ARMS:
            if arm not in reference.get("arms", {}):
                failures.append(f"{row['law_id']}: retained arm missing {arm}")
                continue
            for metric in ("mean", "bias", "variance", "fixed_budget_mse", "ratio_to_full"):
                actual = row["arms"][arm][metric]
                expected = reference["arms"][arm].get(metric)
                prior_failures = len(failures)
                assert_close(f"{row['law_id']} {arm}.{metric}", actual, expected, failures)
                if len(failures) == prior_failures and actual is not None:
                    max_numeric_difference = max(
                        max_numeric_difference,
                        abs(float(actual) - float(expected)),
                    )

        uniform_ratio = row["arms"]["m3_uniform"]["ratio_to_full"]
        structured_ratio = row["arms"]["m3_structured"]["fixed_budget_mse"] / row["arms"]["m4_full"]["fixed_budget_mse"]
        structured_uniform = row["arms"]["m3_structured"]["fixed_budget_mse"] / row["arms"]["m3_uniform"]["fixed_budget_mse"]
        structured_ratios.append(structured_ratio)
        uniform_ratios.append(uniform_ratio)
        structured_uniform_ratios.append(structured_uniform)
        structured_beats_full += int(structured_ratio < 1.0)
        structured_beats_uniform += int(
            row["arms"]["m3_structured"]["fixed_budget_mse"]
            < row["arms"]["m3_uniform"]["fixed_budget_mse"]
        )
        structured_at_least_10pct_better += int(structured_ratio <= 0.90)

    aggregate_expectations = {
        "law_count": len(recomputed),
        "nondegenerate_laws": len(structured_ratios),
        "structured_beats_full_count": structured_beats_full,
        "structured_10pct_below_full_count": structured_at_least_10pct_better,
        "structured_beats_uniform_count": structured_beats_uniform,
        "structured_mse_ratio_full_mean": statistics.fmean(structured_ratios),
        "structured_mse_ratio_full_median": statistics.median(structured_ratios),
        "uniform_mse_ratio_full_median": statistics.median(uniform_ratios),
        "structured_uniform_ratio_median": statistics.median(structured_uniform_ratios),
    }
    for key, actual in aggregate_expectations.items():
        assert_close(f"summary.{key}", actual, summary.get(key), failures)

    report = {
        "audit_classification": "post-hoc same-host independent implementation; not external replication",
        "bundle": str(bundle),
        "law_count": len(recomputed),
        "law_family_counts": family_counts,
        "pointwise_unbiasedness_cases": pointwise_checks,
        "pointwise_unbiasedness_max_error_bound": TOL,
        "structured_beats_full_count": structured_beats_full,
        "structured_beats_uniform_count": structured_beats_uniform,
        "structured_at_least_10pct_better_count": structured_at_least_10pct_better,
        "structured_full_ratio_median": statistics.median(structured_ratios),
        "structured_full_ratio_mean": statistics.fmean(structured_ratios),
        "structured_uniform_ratio_median": statistics.median(structured_uniform_ratios),
        "uniform_full_ratio_median": statistics.median(uniform_ratios),
        "max_numeric_difference_vs_summary": max_numeric_difference,
        "checks_passed": not failures,
        "failures": failures,
        "limitations": [
            "Uses the same retained synthetic panel and mathematical estimand, so this is not fresh confirmation.",
            "Does not reproduce the panel generator or verify the historical source-to-summary provenance.",
            "Does not measure direction error, verifier performance on natural tasks, optimizer effects, or capability.",
        ],
    }
    return report, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    try:
        report, failures = audit(args.bundle.resolve())
    except Exception as exc:  # Fail closed on malformed or incomplete evidence.
        print(json.dumps({
            "audit_classification": "post-hoc same-host independent implementation; not external replication",
            "checks_passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
