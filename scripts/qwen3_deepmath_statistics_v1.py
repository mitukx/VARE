"""Paired task/seed uncertainty helpers for the locked Qwen3 study."""

from __future__ import annotations

import random


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values or not 0 <= probability <= 1:
        raise ValueError("quantile needs values and a probability in [0, 1]")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_task_seed_bootstrap(
    candidate: list[list[float]],
    control: list[list[float]],
    *,
    seed: int,
    draws: int,
) -> dict:
    """Report paired task-only, seed-only, and hierarchical percentile intervals.

    Matrices are seed × task and must use the same task order in each arm.
    The hierarchical resample first samples seeds, then tasks within each
    sampled seed, preserving candidate/control pairing at every draw.
    """
    if not candidate or len(candidate) != len(control) or draws <= 0:
        raise ValueError("candidate/control matrices and positive bootstrap draws are required")
    task_count = len(candidate[0])
    if task_count == 0 or any(len(row) != task_count for row in candidate + control):
        raise ValueError("all seed rows must have the same non-zero task count")
    paired = [
        [float(candidate[s][t]) - float(control[s][t]) for t in range(task_count)]
        for s in range(len(candidate))
    ]
    point = mean([value for row in paired for value in row])
    rng = random.Random(seed)
    task_only: list[float] = []
    seed_only: list[float] = []
    hierarchical: list[float] = []
    seed_count = len(paired)
    for _ in range(draws):
        task_indices = [rng.randrange(task_count) for _ in range(task_count)]
        task_only.append(mean([paired[s][t] for s in range(seed_count) for t in task_indices]))
        seed_indices = [rng.randrange(seed_count) for _ in range(seed_count)]
        seed_only.append(mean([value for s in seed_indices for value in paired[s]]))
        selected = []
        for s in seed_indices:
            selected.extend(paired[s][rng.randrange(task_count)] for _ in range(task_count))
        hierarchical.append(mean(selected))

    def interval(values: list[float]) -> list[float]:
        values.sort()
        return [_quantile(values, 0.025), _quantile(values, 0.975)]

    return {
        "mean_candidate_minus_control": point,
        "task_only_95_percentile_interval": interval(task_only),
        "seed_only_95_percentile_interval": interval(seed_only),
        "hierarchical_task_seed_95_percentile_interval": interval(hierarchical),
        "hierarchical_bootstrap_seed": seed,
        "hierarchical_bootstrap_draws": draws,
        "seed_deltas": [mean(row) for row in paired],
    }
