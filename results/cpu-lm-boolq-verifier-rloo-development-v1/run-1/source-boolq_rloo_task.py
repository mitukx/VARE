#!/usr/bin/env python3
"""Prompt-only cohort selection and metrics for BoolQ binary-action RL."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Sequence

try:  # Package import from tests and other Python modules.
    from .boolq_sequence_task import build_sequence_example, digest_question
except ImportError:  # Direct execution with scripts/ on sys.path.
    from boolq_sequence_task import build_sequence_example, digest_question


def hash_rank_indices(dataset: Any, start: int, stop: int) -> list[int]:
    """Rank a split using question/passage only; never request answer labels here."""
    if start < 0 or stop <= start or stop > len(dataset):
        raise ValueError("invalid BoolQ hash-rank range")
    prompt_view = dataset.select_columns(["question", "passage"])
    ranked = sorted(
        range(len(prompt_view)),
        key=lambda i: (digest_question(prompt_view[i]["question"], prompt_view[i]["passage"]), i),
    )
    return ranked[start:stop]


def build_selected_rows(dataset: Any, indices: Sequence[int], split: str) -> list[dict[str, Any]]:
    """Materialize full rows only after cohort indices are fixed by the protocol."""
    rows = []
    for index in indices:
        row = dataset[int(index)]
        rows.append(build_sequence_example(row, int(index), split))
    return rows


def validate_unique_hashes(rows: Sequence[dict[str, Any]], *, label: str) -> None:
    hashes = [str(row["question_sha256"]) for row in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"duplicate prompt hash in {label} cohort")


def validate_disjoint_hashes(cohorts: dict[str, Sequence[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for label, rows in cohorts.items():
        validate_unique_hashes(rows, label=label)
        for row in rows:
            digest = str(row["question_sha256"])
            previous = seen.get(digest)
            if previous is not None:
                raise ValueError(f"prompt hash overlaps {previous} and {label}")
            seen[digest] = label


def balanced_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if len(predictions) != len(labels) or not labels:
        raise ValueError("prediction/label length mismatch or empty cohort")
    recalls = []
    for cls in (0, 1):
        idx = [i for i, y in enumerate(labels) if int(y) == cls]
        if not idx:
            raise ValueError("balanced accuracy requires both classes")
        recalls.append(sum(int(predictions[i]) == cls for i in idx) / len(idx))
    return sum(recalls) / 2.0


def binary_nll(prob_yes: Sequence[float], labels: Sequence[int]) -> float:
    if len(prob_yes) != len(labels) or not labels:
        raise ValueError("probability/label length mismatch or empty cohort")
    eps = 1e-12
    losses = []
    for p, y in zip(prob_yes, labels):
        p = min(1 - eps, max(eps, float(p)))
        losses.append(-math.log(p if int(y) == 1 else 1 - p))
    return sum(losses) / len(losses)


def bernoulli_kl(p: float, q: float) -> float:
    """KL(Bernoulli(p) || Bernoulli(q)) with stable boundary handling."""
    eps = 1e-12
    p = min(1 - eps, max(eps, float(p)))
    q = min(1 - eps, max(eps, float(q)))
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def class_weights(labels: Sequence[int]) -> dict[int, float]:
    if not labels:
        raise ValueError("cannot weight an empty training cohort")
    counts = {cls: sum(int(y) == cls for y in labels) for cls in (0, 1)}
    if min(counts.values()) == 0:
        raise ValueError("training cohort must contain both classes")
    return {cls: 0.5 / (counts[cls] / len(labels)) for cls in (0, 1)}


def stratified_paired_bootstrap(
    per_prompt_a: Sequence[Sequence[float]],
    per_prompt_b: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    """Prompt-paired CI for mean-across-seed metric differences, stratified by gold class."""
    import random

    if len(per_prompt_a) != len(labels) or len(per_prompt_b) != len(labels) or not labels:
        raise ValueError("bootstrap input length mismatch or empty cohort")
    if any(len(a) != len(b) for a, b in zip(per_prompt_a, per_prompt_b)):
        raise ValueError("paired per-prompt seed counts differ")
    class_indices = {cls: [i for i, y in enumerate(labels) if int(y) == cls] for cls in (0, 1)}
    if any(not values for values in class_indices.values()):
        raise ValueError("stratified bootstrap requires both classes")

    def ba(indices: Iterable[int], values: Sequence[Sequence[float]]) -> float:
        recalls = []
        for cls in (0, 1):
            selected = [i for i in indices if int(labels[i]) == cls]
            recalls.append(sum(sum(values[i]) / len(values[i]) for i in selected) / len(selected))
        return sum(recalls) / 2.0

    observed = ba(range(len(labels)), per_prompt_a) - ba(range(len(labels)), per_prompt_b)
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        selected = []
        for cls in (0, 1):
            pool = class_indices[cls]
            selected.extend(rng.choices(pool, k=len(pool)))
        diffs.append(ba(selected, per_prompt_a) - ba(selected, per_prompt_b))
    diffs.sort()
    lo = diffs[max(0, int(0.025 * resamples))]
    hi = diffs[min(resamples - 1, int(0.975 * resamples))]
    return {"difference": observed, "ci95_low": lo, "ci95_high": hi}


def canonical_sha256(value: Any) -> str:
    import json

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
