"""Reproduce the exact-output-KL / opposite-gradient example for a tiny MLP.

This is a direct autograd check of a parameter symmetry, not a trainer or an
RL experiment. It has no rollout, reward, optimizer, or model-quality claim.
"""

from __future__ import annotations

import json

import torch


def action_probability(parameters: torch.Tensor) -> torch.Tensor:
    a1, w1, a2, w2 = parameters.unbind()
    logit = a1 * torch.sigmoid(w1) + a2 * torch.sigmoid(w2)
    return torch.sigmoid(logit)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    # Each hidden unit is one sigmoid feature followed by a learned output
    # weight. Swapping complete hidden units leaves the network function fixed.
    theta = torch.tensor([10.0, 0.0, -10.0, 0.0], requires_grad=True)
    swap = torch.tensor([2, 3, 0, 1])
    theta_swapped = theta.detach()[swap].clone().requires_grad_(True)

    probability = action_probability(theta)
    swapped_probability = action_probability(theta_swapped)
    gradient = torch.autograd.grad(probability, theta)[0]
    swapped_gradient = torch.autograd.grad(swapped_probability, theta_swapped)[0]
    cosine = torch.nn.functional.cosine_similarity(gradient, swapped_gradient, dim=0)

    epsilons = (1e-3, 1e-2)
    finite_steps = {}
    for epsilon in epsilons:
        stale_delta = action_probability(theta_swapped.detach() + epsilon * gradient) - swapped_probability.detach()
        fresh_delta = action_probability(theta_swapped.detach() + epsilon * swapped_gradient) - swapped_probability.detach()
        finite_steps[str(epsilon)] = {
            "stale_gradient_probability_change": stale_delta.item(),
            "fresh_gradient_probability_change": fresh_delta.item(),
        }

    result = {
        "network": "one-input, two-hidden-unit sigmoid MLP with Bernoulli policy",
        "parameters": theta.detach().tolist(),
        "swapped_parameters": theta_swapped.detach().tolist(),
        "policy_probability_original": probability.item(),
        "policy_probability_swapped": swapped_probability.item(),
        "policy_kl_nats": 0.0,
        "gradient_original": gradient.tolist(),
        "gradient_swapped": swapped_gradient.tolist(),
        "gradient_cosine": cosine.item(),
        "stale_ascent_directional_derivative": torch.dot(swapped_gradient, gradient).item(),
        "fresh_ascent_directional_derivative": torch.dot(swapped_gradient, swapped_gradient).item(),
        "finite_step_checks": finite_steps,
        "scope": "parameter-symmetry diagnostic only; not evidence of a trainer path or model capability effect",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
