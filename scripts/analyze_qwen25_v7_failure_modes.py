#!/usr/bin/env python3
"""Read-only, CPU-only failure-mode analysis of the frozen Qwen v7 gate.

The raw generation artifact remains in Drive. This script verifies its identity,
recounts token- and group-level metrics from retained records, and joins the
predefined manual annotations below without rewriting any source artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.prompt_id_hash_v3 import canonical_prompt_id_sha256
except ModuleNotFoundError:  # Direct ``python scripts/...py`` invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.prompt_id_hash_v3 import canonical_prompt_id_sha256


SOURCE_RESULT_SHA256 = "7b41f9a9e5e92847269fe7eb63ce55110469146163bdf8dadfd1c5d64a7243fe"
EXPECTED_RESULT_IDENTITY = {
    "protocol_id": "qwen25_deepmath_grpo_math500_v7",
    "git_revision": "95316981b8d654a60bb5d22a18bf9e5e8c4cc26e",
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    "model_weights_sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "deepmath_train_sha256": "e0c5b2fc11978d735a7710273920676977b533e185284044c3eafa63a24479d7",
    "gate_prompt_ids_sha256": "0ab6b16559db37f6ff84bb312e18c29046063fac32fae026d5b1265037d03c9a",
    "protocol_lock_sha256": "ab362b7053ef56c274ab35be76f256154df32afddd2f4a4ed4d40e57e7196765",
    "seed": 20261012,
}

# Classification criteria are fixed for all 26 cap-limited completions:
# A = materially sound work toward the target, cut off before completion;
# B = repeated content with little new mathematical progress;
# C = avoidable step-by-step enumeration where a compact method is apparent;
# D = a material mathematical, interpretation, or method error;
# E = an apparent terminal answer is present but its required final-box format
#     is unfinished or invalid;
# F = the retained text does not support a defensible choice.
# Primary categories are mutually exclusive; tags can identify secondary cues.
TAXONOMY = {
    "A": "sound mathematical progress, unfinished at token cap",
    "B": "repetition or degenerate continuation",
    "C": "inefficient enumeration",
    "D": "reasoning or problem-interpretation breakdown",
    "E": "answer candidate reached, final-box format incomplete/invalid",
    "F": "ambiguous from retained text",
}

# Indexes are zero-based, as in the raw result. Quotes are brief evidence cues,
# not substitutes for the retained text. The corresponding prompt ID is joined
# from the immutable raw record when the analysis is run.
FAILURE_ANNOTATIONS: dict[tuple[int, int], dict[str, Any]] = {
    (0, 3): {"primary": "D", "tags": ["C", "false_bound"], "quote": "the sum y will be slightly less than 1", "confidence": 0.99, "ambiguity": "The first claim contradicts the positive series; the remainder enumerates terms only to 70 of 1,000,000."},
    (3, 3): {"primary": "D", "tags": ["symbol_confusion"], "quote": "Since B_0 is a constant, E[B_0] = 0", "confidence": 0.99, "ambiguity": "The target concerns B_t B_s B_v, but the response substitutes B_0 and misuses covariance identities."},
    (4, 0): {"primary": "D", "tags": ["invalid_theorem_application"], "quote": "Gauss's Law states that the flux through a closed surface is equal to the total charge", "confidence": 0.98, "ambiguity": "The response imports an unstated electrostatics model and treats a nonconstant divergence as constant."},
    (10, 1): {"primary": "C", "tags": ["repeated_enumeration"], "quote": "Next, let's check the next multiple of 9", "confidence": 0.97, "ambiguity": "It checks consecutive examples without deriving a stopping bound or exploiting digit-sum structure."},
    (10, 2): {"primary": "C", "tags": ["repeated_enumeration", "sibling_completion_overlap"], "quote": "Next, we check the next multiple of 9", "confidence": 0.98, "ambiguity": "This independently generated sibling repeats the same enumeration and stops at 162."},
    (11, 1): {"primary": "D", "tags": ["condition_misread", "invalid_divisibility"], "quote": "The condition becomes: 2(f(2)! + 1) divides 2(f(2)!) + 1", "confidence": 0.99, "ambiguity": "The opening transforms the stated prime-divisibility condition into a different expression."},
    (14, 2): {"primary": "D", "tags": ["out_of_domain_cases", "case_enumeration"], "quote": "### Case 9: x = 4", "confidence": 0.99, "ambiguity": "The locked prompt as paraphrased in the completion gives 0≤x,y≤3, yet the response tests 4 and also miscomputes earlier cases."},
    (15, 3): {"primary": "D", "tags": ["invalid_complex_algebra", "sibling_repetition"], "quote": "a^{43/2} = 1", "confidence": 0.99, "ambiguity": "It introduces an invalid fractional-power argument for complex values and an irrelevant 2-adic classification."},
    (17, 0): {"primary": "D", "tags": ["algebra_error", "terminal_work_started"], "quote": "48x(x - 1) = 0", "confidence": 0.99, "ambiguity": "The preceding substitution simplifies to a different quadratic; the claimed intersection coordinates are wrong before area counting."},
    (17, 3): {"primary": "D", "tags": ["algebra_error", "terminal_work_started"], "quote": "A(0, 3) and B(8, -3)", "confidence": 0.99, "ambiguity": "The second intersection is incorrect for the stated line and ellipse; subsequent area equations inherit it."},
    (19, 0): {"primary": "B", "tags": ["D", "repeated_integral", "sibling_completion_overlap"], "quote": "∫_0^1 ∫_0^1 y^2 (1 - x) dx dy", "confidence": 0.94, "ambiguity": "The response repeats the same integral and does not reach a correlation; its joint-density reasoning is also suspect."},
    (19, 1): {"primary": "B", "tags": ["D", "unfinished_integral", "sibling_completion_overlap"], "quote": "First, we integrate with respect to u", "confidence": 0.92, "ambiguity": "The derivation remains at an unfinished integral and does not produce a correlation coefficient."},
    (19, 2): {"primary": "B", "tags": ["D", "repeated_integral", "sibling_completion_overlap"], "quote": "2 ∫_0^1 ∫_0^u u v^2 dv du", "confidence": 0.93, "ambiguity": "It repeats equivalent moment calculations but never combines them into the requested coefficient."},
    (19, 3): {"primary": "B", "tags": ["D", "variance_identity_error", "sibling_completion_overlap"], "quote": "Var(U) = Var(X) + Var(Y)", "confidence": 0.99, "ambiguity": "This sibling asserts a false variance identity and is cut off before a result."},
    (21, 2): {"primary": "E", "tags": ["D", "false_monotonicity", "unfinished_box"], "quote": "the range ... is \boxed{[-0.10, ", "confidence": 0.99, "ambiguity": "A terminal range is attempted, but its endpoint is based on false monotonicity and the box is unfinished."},
    (24, 0): {"primary": "D", "tags": ["pde_confusion", "irrelevant_wave_equation"], "quote": "the equation u_t = u_xx must be homogeneous of degree 0", "confidence": 0.99, "ambiguity": "The response switches between heat and wave equations and invents exponential traveling-wave forms."},
    (24, 1): {"primary": "D", "tags": ["energy_assumption", "arithmetic"], "quote": "c_1^2 - 8c_1 + 28 = 0", "confidence": 0.98, "ambiguity": "It assumes squared heat energy is conserved without support and reaches a quadratic with negative discriminant."},
    (24, 2): {"primary": "D", "tags": ["boundary_condition_confusion", "repeated_integral"], "quote": "For x < 0, we have", "confidence": 0.99, "ambiguity": "The domain is [0,4], but the response reasons about x<0 and repeats an incorrectly evaluated steady-state integral."},
    (25, 0): {"primary": "D", "tags": ["unsupported_polynomial_ansatz", "complex_analysis_error"], "quote": "f'(z) = d/dz ((x + yi)^n)", "confidence": 0.99, "ambiguity": "It invents a power-law ansatz and conflates real and imaginary coordinates with complex values."},
    (25, 2): {"primary": "D", "tags": ["invalid_derivative_rule", "constraint_misread"], "quote": "dx/dz = 1/z", "confidence": 0.99, "ambiguity": "The response uses an invalid derivative rule and repeatedly substitutes the condition for unrelated quantities."},
    (26, 0): {"primary": "D", "tags": ["B", "method_index_error", "sibling_completion_overlap"], "quote": "Fifth iteration (i = 4)", "confidence": 0.96, "ambiguity": "For t=0.3 and h=0.1 it continues beyond the requested steps, with inconsistent predictor-corrector values."},
    (26, 1): {"primary": "D", "tags": ["B", "method_index_error", "sibling_completion_overlap"], "quote": "Using Euler's method: y_{-1}", "confidence": 0.99, "ambiguity": "It invents a backward value, changes the stated starting value during arithmetic, and repeats fixed terms."},
    (26, 2): {"primary": "D", "tags": ["B", "method_index_error", "sibling_completion_overlap"], "quote": "For the next step", "confidence": 0.98, "ambiguity": "The derivative update does not follow y'=3ty and the response repeats a drifting recurrence."},
    (26, 3): {"primary": "D", "tags": ["B", "initial_condition_conflict", "sibling_completion_overlap"], "quote": "y_0 = 0", "confidence": 0.99, "ambiguity": "It replaces the stated y(0)=-1 with y_0=0 and then repeats inconsistent predictor/corrector updates."},
    (28, 0): {"primary": "D", "tags": ["eigenvalue_multiplicity_error", "determinant_error"], "quote": "the eigenvalues of A^{100} are 1, 0, and -1", "confidence": 0.99, "ambiguity": "It incorrectly preserves the -1 eigenvalue under an even power and mishandles multiplicities."},
    (29, 2): {"primary": "B", "tags": ["D", "verbatim_formula_loop", "sibling_completion_overlap"], "quote": "= \\frac{\\omega^k k \\ln \\omega^k}{\\omega^{4k} + 1}", "confidence": 0.99, "ambiguity": "The same residue expression is repeated many times; the stated poles are also incorrect for z^4+1."},
}

# Verbatim excerpts from retained completions, checked byte-for-byte below.
_bs = chr(92)
_EVIDENCE_QUOTES = {
    (0, 3): "the sum " + _bs + "(y" + _bs + ") will be slightly less than 1",
    (3, 3): "Since " + _bs + "( B_0 " + _bs + ") is a constant, " + _bs + "( " + _bs + "mathbb{E}[B_0] = 0 " + _bs + ")",
    (11, 1): "2(f(2)! + 1) " + _bs + "mid 2(f(2)!) + 1.",
    (14, 2): "### Case 9: " + _bs + "( x = 4 " + _bs + ")",
    (17, 3): "So, the points of intersection are " + _bs + "(A(0, 3)" + _bs + ") and " + _bs + "(B(8, -3)" + _bs + ").",
    (19, 0): "E[V^2] = " + _bs + "int_0^1 " + _bs + "int_0^1 y^2 (1 - x)",
    (19, 1): "First, we integrate with respect to " + _bs + "(u" + _bs + "):",
    (19, 2): "2 " + _bs + "int_0^1 " + _bs + "int_0^u u v^2 " + _bs + ", dv " + _bs + ", du",
    (19, 3): _bs + "text{Var}(U) = " + _bs + "text{Var}(X) + " + _bs + "text{Var}(Y)",
    (21, 2): "is " + _bs + "(" + _bs + "boxed{[-0.10, ",
    (24, 0): "the equation " + _bs + "( u_{t} = u_{xx} " + _bs + ") must be homogeneous of degree 0",
    (24, 2): "u_x(0, t) = u_x(4, t) = -2",
    (25, 0): "f'(z) = " + _bs + "frac{d}{dz} " + _bs + "left( (x + yi)^n " + _bs + "right)",
    (25, 2): _bs + "frac{dx}{dz} = " + _bs + "frac{1}{z}",
    (26, 1): "y_{-1} = y_0 + h f_{-1}",
    (28, 0): "Since " + _bs + "(A^{100}" + _bs + ") is also diagonalizable, its eigenvalues are 1, 0, and -1.",
}
for _annotation_key, _quote in _EVIDENCE_QUOTES.items():
    FAILURE_ANNOTATIONS[_annotation_key]["quote"] = _quote


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generated_token_count(token_ids: list[int], eos_token_id: int | None) -> int:
    """Count actual IDs through first EOS, excluding any later batch padding."""
    for index, token_id in enumerate(token_ids):
        if eos_token_id is not None and int(token_id) == int(eos_token_id):
            return index + 1
    return len(token_ids)


def quantile(values: list[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _cross_tab(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        result[str(bool(row[left]))][str(bool(row[right]))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(result.items())}


def _record_rows(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = result.get("prompt_groups")
    if not isinstance(groups, list):
        raise ValueError("result is missing prompt_groups")
    completions: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for expected_index, group in enumerate(groups):
        group_index = group["prompt_index"]
        prompt_id = group["prompt_id"]
        if group_index != expected_index:
            raise ValueError(f"prompt group order mismatch: expected {expected_index}, found {group_index}")
        if prompt_id in seen_ids:
            raise ValueError(f"duplicate prompt ID in results: {prompt_id}")
        seen_ids.add(prompt_id)
        members = group.get("completions")
        if not isinstance(members, list):
            raise ValueError(f"group {group_index} is missing completions")
        if len(members) != 4:
            raise ValueError(f"group {group_index} has {len(members)} completions, expected 4")
        tokens: list[int] = []
        rewards: list[float] = []
        lengths: list[int] = []
        truncations = 0
        eos_count = 0
        successes = 0
        valid_boxes = 0
        for completion_index, completion in enumerate(members):
            if completion.get("prompt_id") != prompt_id or completion.get("train_row_hash") != prompt_id:
                raise ValueError(f"prompt identity mismatch at {group_index}:{completion_index}")
            ids = completion.get("completion_token_ids")
            if not isinstance(ids, list) or any(not isinstance(value, int) for value in ids):
                raise ValueError(f"missing/invalid token IDs at {group_index}:{completion_index}")
            if not isinstance(completion.get("completion"), str):
                raise ValueError(f"missing completion text at {group_index}:{completion_index}")
            eos_token_id = group.get("eos_token_id")
            count = generated_token_count(ids, eos_token_id)
            if count != completion.get("generated_token_count"):
                raise ValueError(f"completion token count mismatch at {group_index}:{completion_index}")
            eos = eos_token_id is not None and int(eos_token_id) in ids
            if bool(completion.get("eos_reached")) != bool(eos):
                raise ValueError(f"EOS flag mismatch at {group_index}:{completion_index}")
            truncated = not eos and count >= int(group["max_new_tokens_per_completion"])
            if bool(completion.get("truncated")) != truncated:
                raise ValueError(f"truncation flag mismatch at {group_index}:{completion_index}")
            reward = float(completion["training_reward"])
            if reward not in (0.0, 1.0):
                raise ValueError(f"unexpected non-binary training reward at {group_index}:{completion_index}")
            success = bool(completion["independent_task_success"])
            box = bool(completion["valid_final_box"])
            disagreement = bool(completion.get("training_reward_independent_checker_disagreement"))
            if disagreement != (bool(reward) != success):
                raise ValueError(f"reward/checker disagreement flag mismatch at {group_index}:{completion_index}")
            row = {
                "prompt_index": group_index,
                "completion_index": completion_index,
                "prompt_id": prompt_id,
                "generated_token_count": count,
                "eos_reached": bool(eos),
                "truncated": bool(truncated),
                "training_reward": reward,
                "independent_task_success": success,
                "valid_final_box": box,
                "reward_checker_disagreement": disagreement,
                "completion": completion.get("completion", ""),
            }
            completions.append(row)
            tokens.append(count)
            rewards.append(reward)
            lengths.append(count)
            truncations += int(truncated)
            eos_count += int(bool(eos))
            successes += int(success)
            valid_boxes += int(box)
        expected_ids = group.get("completion_token_ids")
        observed_ids = [member["completion_token_ids"] for member in members]
        if expected_ids != observed_ids:
            raise ValueError(f"group/completion token ID mismatch in group {group_index}")
        expected_counts = group.get("completion_token_counts")
        if expected_counts != tokens:
            raise ValueError(f"group completion token-count mismatch in group {group_index}")
        expected_group_tokens = sum(tokens)
        if expected_group_tokens != group.get("generated_tokens"):
            raise ValueError(f"group token total mismatch in group {group_index}")
        variance = statistics.pvariance(rewards)
        mixed = min(rewards) != max(rewards)
        if abs(variance - float(group.get("reward_variance", math.nan))) > 1e-12:
            raise ValueError(f"reward variance mismatch in group {group_index}")
        if mixed != bool(group.get("mixed_reward")):
            raise ValueError(f"mixed-reward flag mismatch in group {group_index}")
        group_rows.append({
            "prompt_index": group_index,
            "prompt_id": prompt_id,
            "completion_count": len(members),
            "completion_lengths": lengths,
            "mean_completion_length": statistics.mean(lengths),
            "truncated_count": truncations,
            "eos_count": eos_count,
            "correct_count": successes,
            "valid_final_box_count": valid_boxes,
            "reward_values": rewards,
            "mixed_reward": mixed,
            "reward_variance": variance,
            "all_zero_reward": all(value == 0 for value in rewards),
            "all_one_reward": all(value == 1 for value in rewards),
        })
    return completions, group_rows


def _manual_annotation_rows(
    completions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    observed_keys: set[tuple[int, int]] = set()
    for row in completions:
        key = (row["prompt_index"], row["completion_index"])
        if not row["truncated"]:
            continue
        annotation = FAILURE_ANNOTATIONS.get(key)
        if annotation is None:
            raise ValueError(f"truncated completion missing manual annotation: {key}")
        if annotation["quote"] not in row["completion"]:
            raise ValueError(f"annotation quote not found verbatim in completion {key}")
        if annotation["primary"] not in TAXONOMY:
            raise ValueError(f"unknown failure category at {key}")
        observed_keys.add(key)
        rows.append({
            "prompt_id": row["prompt_id"],
            "prompt_index": row["prompt_index"],
            "completion_index": row["completion_index"],
            "token_length": row["generated_token_count"],
            "eos_reached": row["eos_reached"],
            "truncated": row["truncated"],
            "reward": row["training_reward"],
            "independent_correct": row["independent_task_success"],
            "valid_final_box": row["valid_final_box"],
            "primary_category": annotation["primary"],
            "primary_category_definition": TAXONOMY[annotation["primary"]],
            "secondary_tags": annotation["tags"],
            "evidence_quote": annotation["quote"],
            "confidence": annotation["confidence"],
            "ambiguity": annotation["ambiguity"],
        })
    expected_keys = {key for key in FAILURE_ANNOTATIONS}
    if observed_keys != expected_keys:
        raise ValueError(f"manual annotation cohort mismatch; missing={sorted(expected_keys-observed_keys)}, extra={sorted(observed_keys-expected_keys)}")
    return rows


def analyze_records(result: dict[str, Any]) -> dict[str, Any]:
    completions, groups = _record_rows(result)
    annotations = _manual_annotation_rows(completions)
    if len(groups) != 32 or len(completions) != 128:
        raise ValueError(f"expected 32 groups / 128 completions, found {len(groups)} / {len(completions)}")
    ids = [group["prompt_id"] for group in groups]
    prompt_ids_sha256 = canonical_prompt_id_sha256(ids)
    token_total = sum(row["generated_token_count"] for row in completions)
    eos_rows = [row for row in completions if row["eos_reached"]]
    truncated_rows = [row for row in completions if row["truncated"]]
    valid_rows = [row for row in completions if row["valid_final_box"]]
    failed_format = [row for row in completions if not row["valid_final_box"]]
    correct_rows = [row for row in completions if row["independent_task_success"]]
    reward_one = [row for row in completions if row["training_reward"] == 1]
    reward_zero = [row for row in completions if row["training_reward"] == 0]
    group_correct_counts = Counter(str(row["correct_count"]) for row in groups)
    group_reward_types = {
        "all_zero": [row for row in groups if row["all_zero_reward"]],
        "mixed": [row for row in groups if row["mixed_reward"]],
        "all_one": [row for row in groups if row["all_one_reward"]],
    }
    group_truncation = {}
    for name, subset in group_reward_types.items():
        counts = [row["truncated_count"] for row in subset]
        group_truncation[name] = {
            "group_count": len(subset),
            "groups_with_any_truncation": sum(count > 0 for count in counts),
            "truncated_completion_count": sum(counts),
            "all_four_truncated_group_count": sum(count == 4 for count in counts),
            "mean_truncations_per_group": statistics.mean(counts) if counts else None,
        }
    mixed_with_both_rewards = []
    for group in groups:
        own = [row for row in completions if row["prompt_index"] == group["prompt_index"]]
        ones = [row["generated_token_count"] for row in own if row["training_reward"] == 1]
        zeros = [row["generated_token_count"] for row in own if row["training_reward"] == 0]
        if ones and zeros:
            mixed_with_both_rewards.append({
                "prompt_index": group["prompt_index"],
                "prompt_id": group["prompt_id"],
                "mean_correct_length": statistics.mean(ones),
                "mean_incorrect_length": statistics.mean(zeros),
                "correct_minus_incorrect_length": statistics.mean(ones) - statistics.mean(zeros),
                "truncated_count": group["truncated_count"],
            })
    category_counts = Counter(row["primary_category"] for row in annotations)
    all_lengths = [row["generated_token_count"] for row in completions]
    failed_count = len(failed_format)
    truncated_invalid = sum(row["truncated"] and not row["valid_final_box"] for row in completions)
    required_valid = math.ceil(0.90 * len(completions))
    gap = max(0, required_valid - len(valid_rows))
    remaining_truncated = len(truncated_rows)
    return {
        "study": {
            "protocol_id": result["protocol_id"],
            "frozen_main_revision_at_runbook": "0c2ca5953aa3dc2b21fa9d6aa3049ae680ff1ea6",
            "source_result_sha256": None,
            "source_result_file": "Drive-only raw evidence; not committed",
            "runner_git_revision": result["identity"]["execution_code"]["git_revision"],
            "model_id": result["identity"]["model_id"],
            "model_revision": result["identity"]["model_revision"],
            "model_weights_sha256": result["identity"]["model_weights_sha256"],
            "dataset": result["data"]["repo"],
            "dataset_revision": result["data"]["revision"],
            "dataset_split": result["data"]["split"],
            "dataset_sha256": result["data"]["file_sha256"],
            "protocol_lock_sha256": result["identity"]["protocol_lock_sha256"],
            "prompt_count": len(groups),
            "completions_per_prompt": 4,
            "completion_count": len(completions),
            "recomputed_prompt_ids_sha256": prompt_ids_sha256,
            "recorded_prompt_ids_sha256": result["identity"]["gate_prompt_ids_sha256"],
            "seed": result["identity"]["seed"],
            "optimizer_updates": result["optimizer_updates"],
            "math500_loaded": result["math500_loaded"],
            "sampling": result["identity"]["generation"],
            "runtime": result["identity"]["runtime"],
            "resource_budget": result["identity"]["experiment_budget"],
        },
        "integrity_recount": {
            "prompt_groups": len(groups),
            "completions": len(completions),
            "completion_records_have_prompt_id_text_token_ids_reward_checker_eos_truncation": True,
            "unique_prompt_ids": len(set(ids)),
            "token_ids_recount_total": token_total,
            "recorded_generated_token_total": result["generated_tokens"],
            "recorded_metric_token_total": result["generation"]["generated_tokens_from_generated_token_ids"],
            "recomputed_prompt_ids_sha256": prompt_ids_sha256,
        },
        "metrics": {
            "independent_task_success": {"numerator": len(correct_rows), "denominator": len(completions), "rate": len(correct_rows) / len(completions)},
            "valid_final_box": {"numerator": len(valid_rows), "denominator": len(completions), "rate": len(valid_rows) / len(completions)},
            "eos_reached": {"numerator": len(eos_rows), "denominator": len(completions), "rate": len(eos_rows) / len(completions)},
            "truncated": {"numerator": len(truncated_rows), "denominator": len(completions), "rate": len(truncated_rows) / len(completions)},
            "training_reward_independent_checker_disagreement": {
                "numerator": sum(row["reward_checker_disagreement"] for row in completions),
                "denominator": len(completions),
            },
            "generated_tokens": token_total,
            "completion_length_tokens": {
                "mean": statistics.mean(all_lengths),
                "median": statistics.median(all_lengths),
                "q25": quantile(all_lengths, 0.25),
                "q75": quantile(all_lengths, 0.75),
                "min": min(all_lengths),
                "max": max(all_lengths),
            },
            "termination_conditional_rates": {
                "eos_valid_final_box": sum(row["eos_reached"] and row["valid_final_box"] for row in completions) / len(eos_rows),
                "token_cap_valid_final_box": sum(row["truncated"] and row["valid_final_box"] for row in completions) / len(truncated_rows),
                "valid_format_failures_that_are_truncated": truncated_invalid / failed_count if failed_count else None,
                "prompt_groups_with_any_truncation": sum(row["truncated_count"] > 0 for row in groups) / len(groups),
                "all_four_truncated_prompt_groups": sum(row["truncated_count"] == 4 for row in groups),
            },
            "training_reward_group_mixed_count": sum(row["mixed_reward"] for row in groups),
            "mean_within_group_reward_variance_population": statistics.mean(row["reward_variance"] for row in groups),
            "training_reward_group_variance_by_group": [row["reward_variance"] for row in groups],
        },
        "cross_tabs": {
            "termination_by_valid_final_box": {
                "eos": {"valid": sum(row["eos_reached"] and row["valid_final_box"] for row in completions), "invalid": sum(row["eos_reached"] and not row["valid_final_box"] for row in completions)},
                "token_cap": {"valid": sum(row["truncated"] and row["valid_final_box"] for row in completions), "invalid": sum(row["truncated"] and not row["valid_final_box"] for row in completions)},
            },
            "termination_by_reward": {
                "eos": {"reward_1": sum(row["eos_reached"] and row["training_reward"] == 1 for row in completions), "reward_0": sum(row["eos_reached"] and row["training_reward"] == 0 for row in completions)},
                "token_cap": {"reward_1": sum(row["truncated"] and row["training_reward"] == 1 for row in completions), "reward_0": sum(row["truncated"] and row["training_reward"] == 0 for row in completions)},
            },
            "format_failures": {
                "total": failed_count,
                "truncations_among_format_failures": truncated_invalid,
                "truncations_as_fraction_of_format_failures": truncated_invalid / failed_count if failed_count else None,
                "eos_format_failures": sum(row["eos_reached"] and not row["valid_final_box"] for row in completions),
            },
            "group_correct_count_distribution": dict(sorted(group_correct_counts.items(), key=lambda item: int(item[0]))),
        },
        "reward_and_length": {
            "completion_level_descriptive_only": {
                "reward_1_count": len(reward_one),
                "reward_1_mean_length": statistics.mean(row["generated_token_count"] for row in reward_one),
                "reward_0_count": len(reward_zero),
                "reward_0_mean_length": statistics.mean(row["generated_token_count"] for row in reward_zero),
                "eos_mean_length": statistics.mean(row["generated_token_count"] for row in eos_rows),
                "truncated_mean_length": statistics.mean(row["generated_token_count"] for row in truncated_rows),
                "valid_box_mean_length": statistics.mean(row["generated_token_count"] for row in valid_rows),
                "invalid_box_mean_length": statistics.mean(row["generated_token_count"] for row in failed_format),
            },
            "group_level_descriptive_only": {
                "reward_group_types": group_truncation,
                "mixed_group_correct_minus_incorrect_length": {
                    "mixed_groups_with_both_outcomes": len(mixed_with_both_rewards),
                    "per_group_values": mixed_with_both_rewards,
                    "mean_difference": statistics.mean(row["correct_minus_incorrect_length"] for row in mixed_with_both_rewards),
                    "median_difference": statistics.median(row["correct_minus_incorrect_length"] for row in mixed_with_both_rewards),
                    "groups_correct_responses_shorter": sum(row["correct_minus_incorrect_length"] < 0 for row in mixed_with_both_rewards),
                    "groups_correct_responses_longer": sum(row["correct_minus_incorrect_length"] > 0 for row in mixed_with_both_rewards),
                },
            },
        },
        "group_metrics": groups,
        "failure_mode_taxonomy": TAXONOMY,
        "failure_mode_counts": dict(sorted(category_counts.items())),
        "failure_annotations": annotations,
        "counterfactual_feasibility_bound": {
            "observed_valid": len(valid_rows),
            "total": len(completions),
            "observed_rate": len(valid_rows) / len(completions),
            "required_valid_for_at_least_90_percent": required_valid,
            "additional_valid_needed": gap,
            "observed_truncated_completions": remaining_truncated,
            "minimum_fraction_of_truncated_that_would_need_to_become_valid_if_only_they_changed": gap / remaining_truncated if remaining_truncated else None,
            "logical_upper_bound_if_only_truncated_completions_changed": (len(valid_rows) + remaining_truncated) / len(completions),
            "interpretation": "Arithmetic bound only. It is not an observed effect or a prediction that longer generation would create valid answers.",
        },
        "limits": [
            "One post-hoc, fixed cohort of 32 TRAIN prompts; prompt groups, not 128 completions, are the independent sampling units.",
            "No causal intervention varied the token cap; difficulty, response strategy, truncation, reward, and format can co-vary.",
            "A truncated completion having no valid final box does not show that additional tokens would have yielded one.",
            "No policy update was performed; this is not evidence of GRPO learning or capability improvement.",
            "Manual failure-mode labels are qualitative and not independently adjudicated; quotes are short audit cues.",
            "The raw result stores prompt IDs but not the original prompt strings; task-level quote interpretation is therefore limited to the completion's own restatement.",
            "All examples are one selected TRAIN cohort for one 0.5B model and one sampling configuration; no population or model-family generality is established.",
        ],
    }


def audit_artifacts(
    result_path: Path,
    journal_path: Path,
    partial_path: Path,
    protocol_path: Path,
    result: dict[str, Any],
    journal: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    observed_result_sha = sha256_file(result_path)
    if observed_result_sha != SOURCE_RESULT_SHA256:
        raise ValueError(f"source result SHA-256 mismatch: expected {SOURCE_RESULT_SHA256}, found {observed_result_sha}")
    protocol_sha = sha256_file(protocol_path)
    if protocol_sha != EXPECTED_RESULT_IDENTITY["protocol_lock_sha256"]:
        raise ValueError(f"protocol lock SHA-256 mismatch: found {protocol_sha}")
    identity = result["identity"]
    for key in ("protocol_id", "model_id", "model_revision", "model_weights_sha256", "seed"):
        if identity.get(key) != EXPECTED_RESULT_IDENTITY[key]:
            raise ValueError(f"result identity mismatch for {key}: {identity.get(key)}")
    if identity.get("execution_code", {}).get("git_revision") != EXPECTED_RESULT_IDENTITY["git_revision"]:
        raise ValueError("v7 runner revision mismatch")
    if identity.get("protocol_lock_sha256") != protocol_sha:
        raise ValueError("result's protocol lock hash does not match the checked-in v7 lock")
    if identity.get("gate_prompt_ids_sha256") != EXPECTED_RESULT_IDENTITY["gate_prompt_ids_sha256"]:
        raise ValueError("result's fixed prompt fingerprint mismatch")
    if identity.get("deepmath_train_sha256") != EXPECTED_RESULT_IDENTITY["deepmath_train_sha256"]:
        raise ValueError("result's TRAIN data hash mismatch")
    if result.get("decision") != "stop" or result.get("optimizer_updates") != 0 or result.get("math500_loaded") is not False:
        raise ValueError("v7 stop/zero-update/no-MATH500 invariants are not present")
    if result.get("generated_tokens") != result["generation"].get("generated_tokens_from_generated_token_ids"):
        raise ValueError("top-level and generation token totals disagree")
    if journal.get("decision") != "complete" or journal.get("completed_prompt_groups") != 32:
        raise ValueError("journal does not record all 32 completed groups")
    if journal.get("identity") != identity:
        raise ValueError("result and journal identities differ")
    if journal.get("generated_tokens") != result["generated_tokens"]:
        raise ValueError("result and journal token totals differ")
    for key in ("peak_allocated_vram_bytes", "peak_reserved_vram_bytes"):
        if journal.get(key) != result.get(key):
            raise ValueError(f"result and journal {key} differ")
    elapsed_delta = float(journal["elapsed_seconds"]) - float(result["elapsed_seconds"])
    if elapsed_delta < 0 or elapsed_delta > 2:
        raise ValueError(f"unexpected result/journal elapsed-time difference: {elapsed_delta}")
    if elapsed_delta > 0:
        findings.append({
            "type": "finalization_elapsed_offset",
            "severity": "reported_discrepancy",
            "result_json_seconds": result["elapsed_seconds"],
            "terminal_journal_seconds": journal["elapsed_seconds"],
            "difference_seconds": elapsed_delta,
            "explanation": "GateJournal.finalize writes the result JSON before writing the terminal journal state, so the journal elapsed value includes the intervening finalization time. Neither source artifact was changed.",
        })
    partial_records = [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(partial_records) != 32:
        raise ValueError(f"partial journal has {len(partial_records)} records, expected 32")
    if [row.get("prompt_index") for row in partial_records] != list(range(32)):
        raise ValueError("partial JSONL prompt indexes are missing, duplicated, or out of order")
    if partial_records != result["prompt_groups"]:
        raise ValueError("result JSON prompt_groups differ from durable partial JSONL records")
    expected_protocol = protocol
    if expected_protocol.get("protocol_id") != identity["protocol_id"]:
        raise ValueError("protocol ID mismatch")
    if expected_protocol["model"]["revision"] != identity["model_revision"]:
        raise ValueError("protocol model revision mismatch")
    if expected_protocol["model"]["weights_sha256"] != identity["model_weights_sha256"]:
        raise ValueError("protocol model weights mismatch")
    if expected_protocol["training_data"]["sha256"] != identity["deepmath_train_sha256"]:
        raise ValueError("protocol TRAIN hash mismatch")
    if expected_protocol["training_data"]["sample"]["base_gate_id_list_sha256"] != identity["gate_prompt_ids_sha256"]:
        raise ValueError("protocol prompt-ID fingerprint mismatch")
    if expected_protocol["first_gpu_gate"]["generation"]["seed"] != identity["seed"]:
        raise ValueError("protocol seed mismatch")
    if result["generated_tokens"] != 90294:
        raise ValueError("v7 token total differs from its frozen report value")
    completions, groups = _record_rows(result)
    generation = result["generation"]
    derived = {
        "completion_count": len(completions),
        "eos_reached_count": sum(row["eos_reached"] for row in completions),
        "truncation_count": sum(row["truncated"] for row in completions),
        "training_reward_independent_checker_disagreement_count": sum(row["reward_checker_disagreement"] for row in completions),
        "training_reward_group_mixed_count": sum(row["mixed_reward"] for row in groups),
        "training_reward_group_variance_mean": statistics.mean(row["reward_variance"] for row in groups),
        "valid_final_box_rate": sum(row["valid_final_box"] for row in completions) / len(completions),
        "independent_task_success_rate": sum(row["independent_task_success"] for row in completions) / len(completions),
        "completion_tokens_min": min(row["generated_token_count"] for row in completions),
        "completion_tokens_max": max(row["generated_token_count"] for row in completions),
        "completion_tokens_mean": statistics.mean(row["generated_token_count"] for row in completions),
    }
    for key, observed in derived.items():
        saved = generation.get(key)
        if isinstance(observed, float):
            matches = isinstance(saved, (int, float)) and math.isclose(float(saved), observed, rel_tol=0, abs_tol=1e-12)
        else:
            matches = saved == observed
        if not matches:
            raise ValueError(f"saved generation aggregate {key} disagrees: saved={saved}, recounted={observed}")
    if generation.get("eos_reached_rate") != derived["eos_reached_count"] / len(completions):
        raise ValueError("saved EOS rate disagrees with completion records")
    if generation.get("truncation_rate") != derived["truncation_count"] / len(completions):
        raise ValueError("saved truncation rate disagrees with completion records")
    if generation.get("generated_tokens_from_generated_token_ids") != sum(row["generated_token_count"] for row in completions):
        raise ValueError("saved generated-token metric disagrees with completion records")
    return {
        "source_result_sha256": observed_result_sha,
        "source_journal_sha256": sha256_file(journal_path),
        "source_partial_jsonl_sha256": sha256_file(partial_path),
        "checked_in_protocol_sha256": protocol_sha,
        "result_matches_partial_jsonl": True,
        "result_and_journal_identity_match": True,
        "elapsed_seconds_difference": elapsed_delta,
        "findings": findings,
        "raw_inputs_modified": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def run(result_path: Path, journal_path: Path, partial_path: Path, protocol_path: Path, output_json: Path, failure_csv: Path, group_csv: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    audit = audit_artifacts(result_path, journal_path, partial_path, protocol_path, result, journal, protocol)
    summary = analyze_records(result)
    summary["study"]["source_result_sha256"] = audit["source_result_sha256"]
    summary["provenance_audit"] = audit
    summary["elapsed_seconds"] = {
        "result_json": result["elapsed_seconds"],
        "terminal_journal": journal["elapsed_seconds"],
        "authoritative_for_cumulative_gate_budget": journal["elapsed_seconds"],
        "difference_seconds": audit["elapsed_seconds_difference"],
        "throughput_tokens_per_second_from_terminal_journal": result["generated_tokens"] / journal["elapsed_seconds"],
    }
    summary["gate_decision"] = {
        "frozen_v7_decision": result["decision"],
        "valid_final_box_threshold_passed": result["success_criteria"]["format_at_least_90_percent"],
        "all_other_reported_criteria_passed": all(value for key, value in result["success_criteria"].items() if key != "format_at_least_90_percent"),
        "decision_changed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(failure_csv, summary["failure_annotations"])
    _write_csv(group_csv, summary["group_metrics"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="read-only downloaded v7 result JSON")
    parser.add_argument("--journal", type=Path, required=True, help="read-only completed state journal JSON")
    parser.add_argument("--partial", type=Path, required=True, help="read-only durable prompt-group JSONL")
    parser.add_argument("--protocol", type=Path, default=Path("protocols/qwen25_deepmath_grpo_math500_v7.lock.json"))
    parser.add_argument("--output-json", type=Path, default=Path("results/math-grpo-v7-analysis/summary.json"))
    parser.add_argument("--failure-csv", type=Path, default=Path("results/math-grpo-v7-analysis/truncated-failure-annotations.csv"))
    parser.add_argument("--group-csv", type=Path, default=Path("results/math-grpo-v7-analysis/group-metrics.csv"))
    args = parser.parse_args()
    summary = run(args.result, args.journal, args.partial, args.protocol, args.output_json, args.failure_csv, args.group_csv)
    print(json.dumps({
        "output_json": str(args.output_json),
        "failure_csv": str(args.failure_csv),
        "group_csv": str(args.group_csv),
        "integrity_recount": summary["integrity_recount"],
        "failure_mode_counts": summary["failure_mode_counts"],
        "gate_decision": summary["gate_decision"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
