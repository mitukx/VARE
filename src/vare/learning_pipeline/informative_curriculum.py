"""Small, auditable fixed-budget task schedulers for verifiable GRPO groups.

The adaptive policy is a heuristic expected-informative-group-per-cost sampler.
It is not a new RL objective and makes no causal or inferential claim from the
small family-level pilot. It updates its family-level group prior after each
trusted group and selects without replacement within a fixed slate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence


SCHEDULER_VERSION = "vare_informative_group_cost_scheduler_v1"


def independent_bernoulli_informative_probability(pass_probability: float, group_size: int) -> float:
    """Diagnostic P(mixed binary rewards) under iid Bernoulli completions."""
    if isinstance(pass_probability, bool) or not math.isfinite(pass_probability):
        raise ValueError("pass_probability must be finite")
    if not 0.0 <= pass_probability <= 1.0:
        raise ValueError("pass_probability must be between zero and one")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("group_size must be an integer >= 2")
    return 1.0 - pass_probability**group_size - (1.0 - pass_probability) ** group_size


@dataclass
class GroupRate:
    """Beta-smoothed observed mixed-group rate for one task family."""

    informative_groups: int = 0
    observed_groups: int = 0
    successes: int = 0
    completions: int = 0

    @property
    def alpha(self) -> float:
        return 1.0 + self.informative_groups

    @property
    def beta(self) -> float:
        return 1.0 + self.observed_groups - self.informative_groups

    @property
    def expected_informative_group_probability(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def observed_pass_probability(self) -> float | None:
        return self.successes / self.completions if self.completions else None

    def observe(self, rewards: Sequence[float]) -> bool:
        if not rewards:
            raise ValueError("a task group must contain rewards")
        checked = []
        for reward in rewards:
            if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                raise ValueError("rewards must be numeric binary values")
            if reward not in (0, 1):
                raise ValueError("this scheduler expects trusted binary rewards")
            checked.append(int(reward))
        informative = len(set(checked)) > 1
        self.informative_groups += int(informative)
        self.observed_groups += 1
        self.successes += sum(checked)
        self.completions += len(checked)
        return informative


class InformativeGroupScheduler:
    """Select a fixed slate by uniform or observed-info-per-cost probabilities."""

    def __init__(
        self,
        *,
        task_families: Mapping[str, str],
        family_rates: Mapping[str, GroupRate],
        family_cost_seconds: Mapping[str, float],
        seed: int,
        mode: str,
    ) -> None:
        if mode not in {"uniform", "informative_per_cost"}:
            raise ValueError("mode must be 'uniform' or 'informative_per_cost'")
        if not task_families or set(task_families.values()) - set(family_rates):
            raise ValueError("every task family needs an initial rate")
        if set(task_families.values()) - set(family_cost_seconds):
            raise ValueError("every task family needs an estimated cost")
        if any(not math.isfinite(cost) or cost <= 0 for cost in family_cost_seconds.values()):
            raise ValueError("estimated family costs must be positive and finite")
        self.task_families = dict(task_families)
        self.family_rates = {name: GroupRate(**vars(rate)) for name, rate in family_rates.items()}
        self.family_cost_seconds = dict(family_cost_seconds)
        self.mode = mode
        self.rng = random.Random(seed)
        self.history: list[dict[str, object]] = []
        self.remaining = set(task_families)

    def next_task(self) -> str:
        if not self.remaining:
            raise StopIteration("no eligible task remains in this slate")
        eligible = sorted(self.remaining)
        raw: dict[str, float] = {}
        for task_id in eligible:
            family = self.task_families[task_id]
            if self.mode == "uniform":
                raw[task_id] = 1.0
            else:
                expected_rate = self.family_rates[family].expected_informative_group_probability
                cost = self.family_cost_seconds[family]
                raw[task_id] = expected_rate / cost
        total = math.fsum(raw.values())
        probabilities = {task_id: value / total for task_id, value in raw.items()}
        chosen = self.rng.choices(eligible, weights=[probabilities[item] for item in eligible], k=1)[0]
        family = self.task_families[chosen]
        family_rate = self.family_rates[family]
        self.history.append({
            "draw_index": len(self.history),
            "selected_task_id": chosen,
            "selection_probability": probabilities[chosen],
            "conditional_probabilities": probabilities,
            "mode": self.mode,
            "scheduler_version": SCHEDULER_VERSION,
            "family": family,
            "family_alpha_before_draw": family_rate.alpha,
            "family_beta_before_draw": family_rate.beta,
            "expected_informative_group_probability_before_draw": family_rate.expected_informative_group_probability,
            "estimated_group_cost_seconds": self.family_cost_seconds[family],
            "eligibility": "prospective_train_task; selected without replacement",
        })
        self.remaining.remove(chosen)
        return chosen

    def observe(self, task_id: str, rewards: Sequence[float]) -> bool:
        if task_id not in self.task_families:
            raise ValueError("cannot update scheduler from an ineligible task")
        informative = self.family_rates[self.task_families[task_id]].observe(rewards)
        record = self.history[-1]
        if record.get("selected_task_id") != task_id:
            raise ValueError("outcomes must belong to the most recently selected task")
        record["observed_rewards"] = list(rewards)
        record["observed_informative_group"] = informative
        record["family_alpha_after_observation"] = self.family_rates[self.task_families[task_id]].alpha
        record["family_beta_after_observation"] = self.family_rates[self.task_families[task_id]].beta
        return informative

    def select_slate(self, count: int) -> list[str]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count > len(self.remaining):
            raise ValueError("count must be between zero and the number of eligible tasks")
        return [self.next_task() for _ in range(count)]


def family_group_rates(
    observations: Sequence[tuple[str, Sequence[float]]],
    task_families: Mapping[str, str],
) -> dict[str, GroupRate]:
    """Aggregate one or more trusted groups per task into family-level priors."""
    result = {family: GroupRate() for family in set(task_families.values())}
    for task_id, rewards in observations:
        if task_id not in task_families:
            raise ValueError(f"observation for task outside eligible task families: {task_id}")
        result[task_families[task_id]].observe(rewards)
    return result
