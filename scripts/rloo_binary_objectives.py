#!/usr/bin/env python3
"""Binary policy-gradient objectives shared by the CPU RLOO experiment."""
from __future__ import annotations


def _validate_shapes(logits, labels, weights, base_logits):
    import torch

    if logits.ndim != 1 or labels.ndim != 1 or weights.ndim != 1 or base_logits.ndim != 1:
        raise ValueError("binary objectives expect one-dimensional prompt batches")
    if not (len(logits) == len(labels) == len(weights) == len(base_logits)):
        raise ValueError("binary objective batch lengths differ")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("binary labels must be 0 or 1")


def bernoulli_kl_from_logits(policy_logits, reference_logits):
    """Exact forward KL for Bernoulli policies represented by log-odds."""
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(policy_logits)
    log_p = F.logsigmoid(policy_logits)
    log_not_p = F.logsigmoid(-policy_logits)
    log_q = F.logsigmoid(reference_logits)
    log_not_q = F.logsigmoid(-reference_logits)
    return p * (log_p - log_q) + (1 - p) * (log_not_p - log_not_q)


def correct_action_log_probability(logits, labels):
    import torch.nn.functional as F

    return labels * F.logsigmoid(logits) + (1 - labels) * F.logsigmoid(-logits)


def supervised_loss(logits, labels, weights, base_logits, kl_coefficient):
    """Class-weighted action SFT plus the shared analytic KL penalty."""
    import torch

    _validate_shapes(logits, labels, weights, base_logits)
    nll = -weights * correct_action_log_probability(logits, labels)
    kl = bernoulli_kl_from_logits(logits, base_logits)
    return (nll + kl_coefficient * kl).mean(), {"nll": nll.mean(), "kl": kl.mean()}


def exact_expected_reward_loss(logits, labels, weights, base_logits, kl_coefficient):
    """Exact negative verifier reward; its gradient equals expected REINFORCE."""
    import torch

    _validate_shapes(logits, labels, weights, base_logits)
    p_yes = torch.sigmoid(logits)
    p_correct = torch.where(labels == 1, p_yes, 1 - p_yes)
    reward_objective = -weights * p_correct
    kl = bernoulli_kl_from_logits(logits, base_logits)
    return (reward_objective + kl_coefficient * kl).mean(), {
        "negative_expected_reward": reward_objective.mean(),
        "kl": kl.mean(),
    }


def rloo_policy_loss(logits, actions, labels, weights, base_logits, kl_coefficient):
    """K=4-style detached leave-one-out REINFORCE loss for one update batch."""
    import torch
    import torch.nn.functional as F

    _validate_shapes(logits, labels, weights, base_logits)
    if actions.ndim != 2 or actions.shape[0] != logits.shape[0] or actions.shape[1] < 2:
        raise ValueError("actions must have shape [prompts, K] with K >= 2")
    if not torch.all((actions == 0) | (actions == 1)):
        raise ValueError("sampled actions must be 0 or 1")
    group_size = actions.shape[1]
    rewards = (actions == labels[:, None]).to(dtype=logits.dtype) * weights[:, None]
    baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (group_size - 1)
    advantages = (rewards - baseline).detach()
    logp = actions * F.logsigmoid(logits[:, None]) + (1 - actions) * F.logsigmoid(-logits[:, None])
    policy_loss = -(advantages * logp).mean()
    kl = bernoulli_kl_from_logits(logits, base_logits)
    total = policy_loss + kl_coefficient * kl.mean()
    mixed = (rewards.max(dim=1).values != rewards.min(dim=1).values).to(dtype=logits.dtype)
    zero_advantage = (advantages.abs().sum(dim=1) == 0).to(dtype=logits.dtype)
    return total, {
        "policy_loss": policy_loss,
        "kl": kl.mean(),
        "reward_mean": rewards.mean(),
        "reward_variance": rewards.var(unbiased=False),
        "mixed_reward_group_rate": mixed.mean(),
        "zero_advantage_group_rate": zero_advantage.mean(),
        "actions_yes": (actions == 1).to(dtype=logits.dtype).mean(),
    }


def expected_reward_gradient_self_check(tolerance: float = 1e-10):
    """Enumerate binary action groups and verify RLOO's expected reward gradient."""
    import itertools
    import math
    import torch

    checked = 0
    max_abs_error = 0.0
    group_size = 4
    for logit_value in (-1.3, 0.37, 1.8):
        probability_yes = 1 / (1 + math.exp(-logit_value))
        for gold in (0.0, 1.0):
            for weight in (0.6, 1.4):
                logits = torch.tensor([logit_value], dtype=torch.float64, requires_grad=True)
                labels = torch.tensor([gold], dtype=torch.float64)
                weights = torch.tensor([weight], dtype=torch.float64)
                base = torch.tensor([logit_value], dtype=torch.float64)
                exact_loss, _ = exact_expected_reward_loss(logits, labels, weights, base, 0.0)
                exact_gradient = torch.autograd.grad(exact_loss, logits)[0].item()
                enumerated_gradient = 0.0
                for group in itertools.product((0, 1), repeat=group_size):
                    actions = torch.tensor([group], dtype=torch.float64)
                    probability = math.prod(
                        probability_yes if action == 1 else 1 - probability_yes for action in group
                    )
                    sample_logits = torch.tensor([logit_value], dtype=torch.float64, requires_grad=True)
                    loss, _ = rloo_policy_loss(sample_logits, actions, labels, weights, base, 0.0)
                    gradient = torch.autograd.grad(loss, sample_logits)[0].item()
                    enumerated_gradient += probability * gradient
                error = abs(enumerated_gradient - exact_gradient)
                max_abs_error = max(max_abs_error, error)
                checked += 1
                if error > tolerance:
                    raise AssertionError(
                        f"RLOO/exact expected-reward gradient mismatch: {error} > {tolerance}"
                    )
    return {
        "pass": True,
        "enumerated_cases": checked,
        "group_size": group_size,
        "max_abs_gradient_error": max_abs_error,
        "tolerance": tolerance,
        "description": "Full enumeration over binary action groups, both labels, three logits, and two reward weights.",
    }
