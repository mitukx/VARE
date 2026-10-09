"""Independent exact score-update oracle; intentionally imports no VARE code."""
from __future__ import annotations

from fractions import Fraction
from itertools import product
import json


def update_distribution(q_first: Fraction, q_second: Fraction):
    target_probability = Fraction(1, 2)
    distribution = {}
    for first, second in product((0, 1), repeat=2):
        probability = (q_first if first else 1 - q_first) * (
            q_second if second else 1 - q_second
        )
        if first == second:
            update = Fraction(0)
        else:
            # Binary rewards with population-standardized G=2 advantages give
            # A_i in {-1,+1}; the exact target-policy score is a_i - p.
            first_advantage = 1 if first else -1
            second_advantage = 1 if second else -1
            first_score = Fraction(first) - target_probability
            second_score = Fraction(second) - target_probability
            update = (
                first_advantage * first_score
                + second_advantage * second_score
            ) / 2
        distribution[update] = distribution.get(update, Fraction(0)) + probability
    return distribution


def moments(q_first: Fraction, q_second: Fraction):
    distribution = update_distribution(q_first, q_second)
    mean = sum(value * probability for value, probability in distribution.items())
    variance = sum(
        (value - mean) ** 2 * probability
        for value, probability in distribution.items()
    )
    return mean, variance


def main() -> None:
    target, target_variance = moments(Fraction(1, 2), Fraction(1, 2))
    mixed, mixed_variance = moments(Fraction(9, 10), Fraction(1, 10))
    difference = mixed - target
    assert target == Fraction(1, 4)
    assert mixed == Fraction(41, 100)
    assert difference == Fraction(4, 25)
    target_two_group_mse = target_variance / 2
    mixed_two_group_mse = mixed_variance / 2 + difference**2
    assert target_variance == Fraction(1, 16)
    assert mixed_variance == Fraction(369, 10000)
    assert target_two_group_mse == Fraction(1, 32)
    assert mixed_two_group_mse == Fraction(881, 20000)
    print(json.dumps({
        "target_policy_group_expected_update": str(target),
        "target_policy_single_group_variance": str(target_variance),
        "mixed_behavior_group_expected_update_at_target_score": str(mixed),
        "mixed_behavior_single_group_variance": str(mixed_variance),
        "absolute_estimand_difference": str(difference),
        "relative_difference_percent": str(100 * difference / target),
        "two_group_mean_mse_target_policy": str(target_two_group_mse),
        "two_group_mean_mse_mixed_behavior_vs_target": str(mixed_two_group_mse),
        "two_group_mse_ratio_mixed_over_target": str(mixed_two_group_mse / target_two_group_mse),
        "scope": "exact synthetic score-function calculation; not model capability or a universal GRPO correctness claim",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
