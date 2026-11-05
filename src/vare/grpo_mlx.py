"""Group-relative policy optimization (GRPO) for MLX LoRA policies.

This module is the trainable counterpart of the advantage-only stub in
``grpo.py`` (which stays byte-identical because it is hash-frozen by the v1
protocol). It implements one clipped-ratio policy-gradient update on whole
rollout groups:

* Samples are grouped by ``(instance_id, behavior_checkpoint_id)``; a group
  mixes neither tasks nor behavior policies nor verifier versions.
* Advantages are group-relative: ``A_i = r_i - mean(r)`` (optionally divided by
  the population standard deviation). A group whose rewards are all equal has
  no learning signal; its advantages are exactly zero and its tokens are
  excluded from the loss denominator, so it changes neither the gradient nor
  the normalization of the other groups. A step with no informative group
  applies no optimizer update.
* Log-probabilities are computed on policy-generated completion tokens only
  (prompt and padding are masked out) from ``logits / temperature``. A Sample
  may contain multiple independently conditioned trajectory segments; the same
  group-relative terminal advantage is then applied to every action segment.
* The objective is the PPO/GRPO clipped surrogate
  ``min(rho * A, clip(rho, 1 - eps_low, 1 + eps_high) * A)`` with
  ``rho = exp(logp_current - logp_behavior)``, averaged over all completion
  tokens of informative samples in the step. No KL penalty (beta = 0).
* By default the update is strictly on-policy: every sample must come from the
  checkpoint being updated (``max_policy_lag = 0``), so ``rho == 1`` at the
  update point and the gradient equals the REINFORCE-with-group-baseline
  gradient. Behavior log-probs come from the same traced (differentiated)
  computation as the update, so ``rho == 1`` holds numerically, not only in
  exact arithmetic; ``max_abs_log_ratio`` is reported per step. Stale samples raise ``StalePolicyError`` instead of being reused.
* Sampling decision (Option A, exact ratio): the behavior distribution must be
  the untruncated temperature softmax ``softmax(logits / T)`` with ``T > 0``,
  ``top_p == 1`` (no top-k / min-p), because that is the distribution the
  log-probabilities above model. A top-p-truncated sampler renormalizes over
  the nucleus, so ``logp_behavior`` would be wrong and ``rho`` would no longer
  be the exact on-policy ratio. ``check_exact_ratio_sampling`` enforces this;
  runs sampled with ``top_p < 1`` are approximate-ratio evidence only.
* Gradients are accumulated one sample at a time; because each microbatch loss
  is divided by the step-wide token count, the accumulated gradient equals the
  full-batch gradient.

Checkpoints hold the trainable (LoRA) parameters and the full optimizer state,
and require the pinned identity of the frozen base model on save and reload.
MLX 0.32 exposes no setter for its global RNG key, so randomness is instead
derived from an explicit ``(base_seed, step)`` schedule (``reseed``) that is
stored in the checkpoint and re-applied on load; with LoRA dropout 0 the update
itself is deterministic, and a reloaded run continues bit-identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA = "vare.learning_pipeline.grpo_step/v1"
CHECKPOINT_SCHEMA = "vare.learning_pipeline.grpo_checkpoint/v3"


class StalePolicyError(ValueError):
    """A sample's behavior policy is older than the configured lag allows."""


class GroupError(ValueError):
    """Samples cannot form valid GRPO groups."""


class TruncatedSamplingError(ValueError):
    """Sampling settings whose behavior distribution the exact-ratio objective does not model."""


# The exact-ratio objective models behavior log-probs as log_softmax(logits / T)
# over the full vocabulary, so sampling must not truncate the distribution.
EXACT_RATIO_TOP_P = 1.0
ON_POLICY_LOG_RATIO_TOLERANCE = 1e-6


def check_exact_ratio_sampling(*, temperature: float, top_p: float, top_k: int = 0, min_p: float = 0.0) -> None:
    """Raise unless sampling draws from the untruncated ``softmax(logits / temperature)``.

    Option A of the sampling-consistency decision: with ``top_p == 1`` (and no
    top-k or min-p filtering) the sampler's distribution is exactly the one
    ``token_logprobs(..., temperature=T)`` computes, so the clipped-surrogate
    ratio is the exact on-policy ratio. Any truncation would silently mix a
    truncated behavior policy with an untruncated probability model.
    """
    for name, value in (("temperature", temperature), ("top_p", top_p), ("min_p", min_p)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TruncatedSamplingError(f"{name} must be a finite number")
    if temperature <= 0:
        raise TruncatedSamplingError("exact-ratio GRPO needs stochastic sampling (temperature > 0); "
                                     "greedy decoding is not the modeled softmax distribution")
    if top_p != EXACT_RATIO_TOP_P:
        raise TruncatedSamplingError(f"exact-ratio GRPO requires top_p == 1.0 (untruncated); got top_p={top_p!r}. "
                                     "Truncated sampling would make behavior log-probs and ratios approximate")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k != 0:
        raise TruncatedSamplingError(f"exact-ratio GRPO requires top_k == 0 (no truncation); got top_k={top_k!r}")
    if min_p != 0:
        raise TruncatedSamplingError(f"exact-ratio GRPO requires min_p == 0 (no truncation); got min_p={min_p!r}")


@dataclass(frozen=True)
class GRPOConfig:
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.2
    normalize_std: bool = True
    temperature: float = 1.0
    max_policy_lag: int = 0
    max_grad_norm: float | None = 1.0
    group_size: int | None = None  # if set, every group must have exactly this many samples

    def __post_init__(self) -> None:
        for name in ("clip_eps_low", "clip_eps_high", "temperature"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be a finite number")

        if not 0 <= self.clip_eps_low < 1:
            raise ValueError("clip_eps_low must be in [0, 1)")
        if self.clip_eps_high < 0:
            raise ValueError("clip_eps_high must be nonnegative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not isinstance(self.normalize_std, bool):
            raise ValueError("normalize_std must be a bool")
        if (
            isinstance(self.max_policy_lag, bool)
            or not isinstance(self.max_policy_lag, int)
            or self.max_policy_lag < 0
        ):
            raise ValueError("max_policy_lag must be a nonnegative integer")
        if self.max_grad_norm is not None:
            value = self.max_grad_norm
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "max_grad_norm must be a finite nonnegative number or None"
                )
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or value < 0:
                raise ValueError(
                    "max_grad_norm must be a finite nonnegative number or None"
                )
        if self.group_size is not None and (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size < 1
        ):
            raise ValueError("group_size must be a positive integer or None")

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class TrajectorySegment:
    """One policy-generated action with its actual conditioning context."""

    prompt_ids: list[int]
    completion_ids: list[int]
    sampling_behavior_logprobs: list[float] | None = None
    behavior_logprobs: list[float] | None = None


@dataclass
class Sample:
    instance_id: str
    behavior_checkpoint_id: str
    behavior_step: int
    prompt_ids: list[int]
    completion_ids: list[int]
    reward: float
    verifier: str
    sample_index: int = 0
    behavior_logprobs: list[float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # When present, terminal reward is assigned to every policy action token
    # across these independently conditioned turns. The first segment must
    # match prompt_ids/completion_ids for backward-compatible grouping.
    trajectory_segments: list[TrajectorySegment] | None = None

    @property
    def group_key(self) -> tuple[str, str]:
        return (self.instance_id, self.behavior_checkpoint_id)


# ------------------------------------------------------------------ groups
def group_advantages(rewards: Sequence[float], *, normalize_std: bool = True) -> tuple[list[float], bool]:
    """Return (advantages, informative). Zero-variance groups give all zeros."""
    if not rewards:
        raise GroupError("empty group")
    if any(isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) for r in rewards):
        raise GroupError("rewards must be finite numbers")
    n = len(rewards)
    mean = math.fsum(rewards) / n
    centered = [float(r) - mean for r in rewards]
    var = math.fsum(c * c for c in centered) / n
    if var == 0.0:
        return [0.0] * n, False
    if normalize_std:
        std = math.sqrt(var)
        return [c / std for c in centered], True
    return centered, True


def group_samples(samples: Iterable[Sample], *, group_size: int | None = None) -> dict[tuple[str, str], list[Sample]]:
    groups: dict[tuple[str, str], list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.group_key, []).append(s)
    for key, g in groups.items():
        idx = [s.sample_index for s in g]
        if len(set(idx)) != len(idx):
            raise GroupError(f"duplicate sample_index in group {key}")
        if len({tuple(s.prompt_ids) for s in g}) != 1:
            raise GroupError(f"mixed prompts in group {key}")
        if len({s.verifier for s in g}) != 1:
            raise GroupError(f"mixed verifier versions in group {key}")
        if len({s.behavior_step for s in g}) != 1:
            raise GroupError(f"mixed behavior steps in group {key}")
        if len({s.trajectory_segments is not None for s in g}) != 1:
            raise GroupError(f"mixed single-sequence and trajectory samples in group {key}")
        if group_size is not None and len(g) != group_size:
            raise GroupError(f"partial group {key}: {len(g)} != {group_size}")
        for s in g:
            segments = s.trajectory_segments if s.trajectory_segments is not None else [s]
            if not segments:
                raise GroupError(f"empty trajectory in group {key}")
            if s.trajectory_segments is not None and (
                segments[0].prompt_ids != s.prompt_ids
                or segments[0].completion_ids != s.completion_ids
            ):
                raise GroupError(f"first trajectory segment differs from the sample action in group {key}")
            for segment in segments:
                if not segment.completion_ids:
                    raise GroupError(f"empty completion in group {key}")
                if not segment.prompt_ids or any(
                    type(token) is not int for token in (*segment.prompt_ids, *segment.completion_ids)
                ):
                    raise GroupError(f"prompt and completion tokens must be integer IDs in group {key}")
                logprob_sets = []
                if s.trajectory_segments is None:
                    if s.behavior_logprobs is not None:
                        logprob_sets.append(s.behavior_logprobs)
                else:
                    if segment.sampling_behavior_logprobs is not None:
                        logprob_sets.append(segment.sampling_behavior_logprobs)
                    if segment.behavior_logprobs is not None:
                        logprob_sets.append(segment.behavior_logprobs)
                for values in logprob_sets:
                    if len(values) != len(segment.completion_ids):
                        raise GroupError(f"behavior log-probability length mismatch in group {key}")
                    if any(
                        isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value > 0
                        for value in values
                    ):
                        raise GroupError(
                            f"behavior log-probabilities must be finite nonpositive numbers in group {key}"
                        )
    return groups


def check_policy_lag(samples: Iterable[Sample], *, policy_checkpoint_id: str, policy_step: int,
                     max_policy_lag: int) -> int:
    worst = 0
    for s in samples:
        lag = policy_step - s.behavior_step
        if lag < 0:
            raise StalePolicyError(f"sample from future step {s.behavior_step} > {policy_step}")
        if lag == 0 and s.behavior_checkpoint_id != policy_checkpoint_id:
            raise StalePolicyError(
                f"sample checkpoint {s.behavior_checkpoint_id!r} != current {policy_checkpoint_id!r} at lag 0")
        if lag > max_policy_lag:
            raise StalePolicyError(f"policy lag {lag} > max_policy_lag {max_policy_lag}")
        worst = max(worst, lag)
    return worst


def check_on_policy_log_ratio(policy_lag: int, max_abs_log_ratio: float) -> None:
    """Fail closed when a lag-zero sample's behavior log-probs mismatch.

    A checkpoint ID is caller-supplied provenance, not proof that the stored
    behavior log-probabilities came from that checkpoint. The differentiated
    current-policy pass must reproduce each lag-zero sample's stored
    probabilities within floating-point tolerance before any optimizer state
    can change. Stale samples retain the configured importance-ratio behavior.
    """
    if policy_lag != 0:
        return
    if (isinstance(max_abs_log_ratio, bool) or not isinstance(max_abs_log_ratio, (int, float))
            or not math.isfinite(max_abs_log_ratio) or max_abs_log_ratio < 0
            or max_abs_log_ratio > ON_POLICY_LOG_RATIO_TOLERANCE):
        raise StalePolicyError(
            "lag-zero behavior log-probabilities do not match the current policy: "
            f"max_abs_log_ratio={max_abs_log_ratio!r}, "
            f"tolerance={ON_POLICY_LOG_RATIO_TOLERANCE}"
        )


# ---------------------------------------------------------------- tensors
def build_batch(seqs: Sequence[tuple[Sequence[int], Sequence[int]]], pad_id: int):
    """Right-padded ``tokens`` [B, L] and target ``mask`` [B, L-1].

    ``mask[b, t] == 1`` iff target ``tokens[b, t + 1]`` is a completion token;
    prompt targets and padding are 0.
    """
    import mlx.core as mx

    if not seqs:
        raise ValueError("empty batch")
    length = max(len(p) + len(c) for p, c in seqs)
    if length < 2:
        raise ValueError("sequence too short")
    toks, mask = [], []
    for p, c in seqs:
        if not p:
            raise ValueError("prompt must contain at least one token")
        row = list(p) + list(c)
        toks.append(row + [pad_id] * (length - len(row)))
        m = [0.0] * (length - 1)
        for t in range(len(p) - 1, len(p) - 1 + len(c)):
            m[t] = 1.0
        mask.append(m)
    return mx.array(toks, dtype=mx.int32), mx.array(mask, dtype=mx.float32)


def _logits(model, inputs, start: int):
    """Logits for input positions >= ``start``; skips the vocab projection on
    prompt positions for mlx-lm decoder models (``model.model`` + head)."""
    inner = getattr(model, "model", None)
    args = getattr(model, "args", None)
    if start > 0 and inner is not None and args is not None and hasattr(inner, "embed_tokens"):
        h = inner(inputs)[:, start:]
        if getattr(args, "tie_word_embeddings", False):
            return inner.embed_tokens.as_linear(h)
        return model.lm_head(h)
    return model(inputs)[:, start:]


def token_logprobs(model, tokens, *, temperature: float = 1.0, start: int = 0):
    """log pi(tokens[:, t+1] | tokens[:, :t+1]) for t >= start, float32 [B, L-1-start]."""
    import mlx.core as mx

    logits = _logits(model, tokens[:, :-1], start).astype(mx.float32)
    if temperature != 1.0:
        logits = logits / temperature
    targets = tokens[:, 1 + start:]
    lse = mx.logsumexp(logits, axis=-1)
    picked = mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return picked - lse


def clipped_surrogate(logp, behavior_logp, advantages, mask, *, eps_low: float, eps_high: float,
                      n_tokens: float):
    """Negative clipped surrogate summed over masked tokens and divided by
    ``n_tokens``. ``advantages`` is per sequence [B]. Returns (loss, stats)."""
    import mlx.core as mx

    ratio = mx.exp(logp - behavior_logp)
    adv = advantages[:, None]
    unclipped = ratio * adv
    clipped = mx.clip(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    obj = mx.minimum(unclipped, clipped)
    loss = -(obj * mask).sum() / n_tokens
    is_clipped = (clipped < unclipped).astype(mx.float32) * mask
    stats = {
        "clipped_tokens": is_clipped.sum(),
        "ratio_sum": (ratio * mask).sum(),
        "ratio_max": mx.max(mx.where(mask > 0, ratio, 0.0)),
        # k3 estimator of KL(behavior || current) on behavior samples
        "kl_sum": ((mx.exp(behavior_logp - logp) - (behavior_logp - logp) - 1.0) * mask).sum(),
        "max_abs_log_ratio": mx.max(mx.abs(logp - behavior_logp) * mask),
    }
    return loss, stats


def sample_tensors(s: Sample | TrajectorySegment, pad_id: int):
    tokens, mask = build_batch([(s.prompt_ids, s.completion_ids)], pad_id)
    start = len(s.prompt_ids) - 1
    return tokens, mask[:, start:], start


def informative_samples(samples: Sequence[Sample], cfg: GRPOConfig) -> list[Sample]:
    """Samples in groups with reward variance (the only ones ``train_step`` uses)."""
    out: list[Sample] = []
    for g in group_samples(samples, group_size=cfg.group_size).values():
        if group_advantages([s.reward for s in g], normalize_std=cfg.normalize_std)[1]:
            out.extend(g)
    return out


def _sample_segments(sample: Sample) -> list[Sample | TrajectorySegment]:
    return sample.trajectory_segments if sample.trajectory_segments is not None else [sample]


def forward_logprobs(model, s: Sample, *, temperature: float, pad_id: int = 0) -> list[float]:
    """Plain (untraced) forward log-probs of ``s``'s completion tokens. Not used
    for the ratio: on Metal/bf16 it can differ from the differentiated forward
    by ~0.1 nats on single tokens (see ``compute_behavior_logprobs``)."""
    import mlx.core as mx

    result: list[float] = []
    for segment in _sample_segments(s):
        tokens, _, start = sample_tensors(segment, pad_id)
        lp = token_logprobs(model, tokens, temperature=temperature, start=start)
        mx.eval(lp)
        result.extend(float(x) for x in lp[0, : len(segment.completion_ids)].tolist())
    return result


def compute_behavior_logprobs(model, samples: Iterable[Sample], cfg: GRPOConfig | None = None, *,
                              temperature: float | None = None, pad_id: int = 0,
                              after_sample: Callable[[], None] | None = None) -> None:
    """Fill ``behavior_logprobs`` with the current (behavior) parameters using
    the exact traced computation that ``policy_gradient`` differentiates, so a
    lag-0 ratio is exactly 1. Call before any update that uses these samples,
    under the same execution context (e.g. gradient checkpointing) as the
    update. Measured on the 4B MLX bf16 model, an untraced forward differs from
    the differentiated one by up to 0.13 nats per token; the traced pass is
    deterministic, so the gradients it also produces are discarded."""
    import mlx.core as mx

    if cfg is None:
        if temperature is None:
            raise ValueError("pass cfg or temperature")
        cfg = GRPOConfig(temperature=temperature)
    value_and_grad = _value_and_grad(model, cfg, n_tokens=1)
    for s in samples:
        for index, segment in enumerate(_sample_segments(s)):
            tokens, mask, start = sample_tensors(segment, pad_id)
            zeros = mx.zeros(mask.shape, dtype=mx.float32)
            (_, (_, lp)), g = value_and_grad(tokens, mask, start, mx.zeros((1,), dtype=mx.float32), zeros)
            mx.eval(lp, g)
            del g
            values = [float(x) for x in lp[0, : len(segment.completion_ids)].tolist()]
            if s.trajectory_segments is not None:
                segment.behavior_logprobs = values
                if index == 0:
                    s.behavior_logprobs = values
            else:
                s.behavior_logprobs = values
            if after_sample is not None:
                after_sample()


# ------------------------------------------------------------------ update
def _trainable_digest_from_flat(items: Iterable[tuple[str, Any]]) -> str:
    import mlx.core as mx
    import numpy as np

    h = hashlib.sha256()
    for k, v in sorted(items):
        h.update(k.encode())
        h.update(np.asarray(v.astype(mx.float32)).tobytes())
    return h.hexdigest()


def trainable_digest(model) -> str:
    from mlx.utils import tree_flatten

    return _trainable_digest_from_flat(tree_flatten(model.trainable_parameters()))


def _value_and_grad(model, cfg: GRPOConfig, *, n_tokens: int):
    """The one traced loss used both for behavior log-probs and for the update."""
    import mlx.nn as nn

    def loss_fn(tokens, mask, start, adv, blp):
        lp = token_logprobs(model, tokens, temperature=cfg.temperature, start=start)
        loss, stats = clipped_surrogate(lp, blp, adv, mask, eps_low=cfg.clip_eps_low, eps_high=cfg.clip_eps_high,
                                        n_tokens=float(n_tokens))
        return loss, (stats, lp)

    return nn.value_and_grad(model, loss_fn)


def policy_gradient(model, work: Sequence[tuple[Sample, float]], cfg: GRPOConfig, *, n_tokens: int,
                    policy_step: int, pad_id: int = 0,
                    after_sample: Callable[[], None] | None = None) -> tuple[Any, dict[str, float]]:
    """Gradient of the step loss w.r.t. trainable parameters, accumulated one
    sample at a time; equals the gradient of the full-batch token-mean loss.

    ``after_sample`` (optional) runs after each microbatch is evaluated, e.g. to
    release MLX's buffer cache or check a resource limit; it cannot change the
    gradient, and an exception from it aborts the step before any update."""
    import mlx.core as mx
    from mlx.utils import tree_map

    value_and_grad = _value_and_grad(model, cfg, n_tokens=n_tokens)
    grads: Any = None
    loss_total = clipped = ratio_sum = kl_sum = ratio_max = max_lr = 0.0
    for s, a in work:
        for segment in _sample_segments(s):
            behavior_logprobs = (
                segment.behavior_logprobs
                if s.trajectory_segments is not None
                else s.behavior_logprobs
            )
            if behavior_logprobs is None:
                raise ValueError("missing behavior log-probs")
            tokens, mask, start = sample_tensors(segment, pad_id)
            blp = mx.array([behavior_logprobs], dtype=mx.float32)
            (loss, (st, _lp)), g = value_and_grad(
                tokens, mask, start, mx.array([a], dtype=mx.float32), blp
            )
            mx.eval(loss, st, g)
            check_on_policy_log_ratio(policy_step - s.behavior_step, float(st["max_abs_log_ratio"]))
            grads = g if grads is None else tree_map(lambda x, y: x + y, grads, g)
            mx.eval(grads)
            loss_total += float(loss)
            clipped += float(st["clipped_tokens"])
            ratio_sum += float(st["ratio_sum"])
            kl_sum += float(st["kl_sum"])
            ratio_max = max(ratio_max, float(st["ratio_max"]))
            max_lr = max(max_lr, float(st["max_abs_log_ratio"]))
            if after_sample is not None:
                after_sample()
    diag = {"loss": loss_total, "clip_fraction": clipped / n_tokens, "ratio_mean": ratio_sum / n_tokens,
            "ratio_max": ratio_max, "approx_kl": kl_sum / n_tokens,
            "max_abs_log_ratio": max_lr}
    return grads, diag


def train_step(model, optimizer, samples: Sequence[Sample], cfg: GRPOConfig, *, policy_checkpoint_id: str,
               policy_step: int, pad_id: int = 0, after_sample: Callable[[], None] | None = None) -> dict[str, Any]:
    """One GRPO update on whole groups. Returns a JSON-serializable record."""
    import mlx.core as mx
    import mlx.optimizers as optim

    groups = group_samples(samples, group_size=cfg.group_size)
    lag = check_policy_lag(samples, policy_checkpoint_id=policy_checkpoint_id, policy_step=policy_step,
                           max_policy_lag=cfg.max_policy_lag)
    work: list[tuple[Sample, float]] = []
    group_rows = []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda s: s.sample_index)
        adv, informative = group_advantages([s.reward for s in g], normalize_std=cfg.normalize_std)
        group_rows.append({"instance_id": key[0], "behavior_checkpoint_id": key[1], "size": len(g),
                           "rewards": [s.reward for s in g], "advantages": adv, "informative": informative})
        if informative:
            work.extend(zip(g, adv))
    record: dict[str, Any] = {
        "schema": SCHEMA, "policy_checkpoint_id": policy_checkpoint_id, "policy_step": policy_step,
        "max_observed_lag": lag, "config": cfg.to_dict(), "groups": group_rows,
        "n_groups": len(group_rows), "n_informative_groups": sum(r["informative"] for r in group_rows),
        "n_samples": len(samples), "n_trained_samples": len(work),
        "n_trained_action_segments": sum(len(_sample_segments(s)) for s, _ in work),
        "credit_assignment": (
            "full_trajectory_terminal_reward"
            if samples and samples[0].trajectory_segments is not None
            else "single_sequence_terminal_reward"
        ),
    }
    for s, _ in work:  # zero-advantage samples need no log-probs
        for segment in _sample_segments(s):
            behavior_logprobs = (
                segment.behavior_logprobs
                if s.trajectory_segments is not None
                else s.behavior_logprobs
            )
            if behavior_logprobs is None or len(behavior_logprobs) != len(segment.completion_ids):
                raise ValueError(f"missing behavior log-probs for {s.instance_id}#{s.sample_index}")
    n_tokens = sum(
        len(segment.completion_ids)
        for s, _ in work
        for segment in _sample_segments(s)
    )
    record["n_tokens"] = n_tokens
    if not work:
        record.update(update_applied=False, reason="no_informative_group")
        return record

    grads, diag = policy_gradient(model, work, cfg, n_tokens=n_tokens, policy_step=policy_step, pad_id=pad_id,
                                  after_sample=after_sample)
    grad_norm = None
    if cfg.max_grad_norm is not None:
        grads, gn = optim.clip_grad_norm(grads, cfg.max_grad_norm)
        grad_norm = float(gn)
    before = trainable_digest(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    after = trainable_digest(model)
    record.update(
        update_applied=True, grad_norm_preclip=grad_norm, **diag, params_digest_before=before, params_digest_after=after,
        params_changed=before != after, optimizer_step=int(optimizer.state["step"].item()),
    )
    return record


# -------------------------------------------------------------- checkpoint
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def step_seed(base_seed: int, step: int) -> int:
    return int(hashlib.sha256(f"{base_seed}:{step}".encode()).hexdigest()[:8], 16)


def reseed(base_seed: int, step: int) -> int:
    """Seed MLX's global RNG for ``step``; called at the start of every step."""
    import mlx.core as mx

    seed = step_seed(base_seed, step)
    mx.random.seed(seed)
    return seed


def _validate_base_model_id(base_model_id: str) -> str:
    if (not isinstance(base_model_id, str) or not base_model_id.strip()
            or len(base_model_id) > 512):
        raise ValueError("base_model_id must be a nonempty string of at most 512 characters")
    return base_model_id


def _optimizer_identity(optimizer) -> dict[str, Any]:
    optimizer_type = type(optimizer)
    configuration = {
        key: value for key, value in vars(optimizer).items()
        if not key.startswith("_") and key != "state"
    }
    try:
        normalized_configuration = json.loads(json.dumps(configuration, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("optimizer configuration is not JSON-serializable") from exc
    return {
        "class": f"{optimizer_type.__module__}.{optimizer_type.__qualname__}",
        "configuration": normalized_configuration,
    }


def save_checkpoint(path: Path, model, optimizer, *, base_model_id: str, step: int, base_seed: int,
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Atomically write adapters, optimizer state and the RNG schedule to
    ``path``. ``base_model_id`` binds adapters to their frozen base model;
    ``step`` is the number of completed updates."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    base_model_id = _validate_base_model_id(base_model_id)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"checkpoint destination already exists: {path}")
    optimizer_step = int(optimizer.state["step"].item())
    if type(step) is not int or step < 0 or optimizer_step != step:
        raise ValueError(f"checkpoint step {step!r} does not match optimizer step {optimizer_step}")
    optimizer_identity = _optimizer_identity(optimizer)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    adapters = dict(tree_flatten(model.trainable_parameters()))
    opt = dict(tree_flatten(optimizer.state))
    if any(not isinstance(v, mx.array) for v in opt.values()):
        raise TypeError("optimizer state contains non-array entries")
    mx.save_safetensors(str(tmp / "adapters.safetensors"), adapters)
    mx.save_safetensors(str(tmp / "optimizer.safetensors"), opt)
    files = {n: _sha256(tmp / n) for n in ("adapters.safetensors", "optimizer.safetensors")}
    manifest = {"schema": CHECKPOINT_SCHEMA, "base_model_id": base_model_id,
                "optimizer_identity": optimizer_identity,
                "step": step, "optimizer_step": optimizer_step, "files": files,
                "rng": {"base_seed": base_seed, "step": step, "derivation": "sha256(f'{base_seed}:{step}')[:8]"},
                "trainable_digest": trainable_digest(model), "extra": dict(extra or {})}
    with (tmp / "checkpoint.json").open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        descriptor = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass
    if path.exists():
        raise FileExistsError(f"checkpoint destination appeared during write: {path}")
    os.replace(tmp, path)
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass
    return manifest


def load_checkpoint(path: Path, model, optimizer, *, base_model_id: str) -> dict[str, Any]:
    """Restore a checkpoint written by ``save_checkpoint``; verifies hashes,
    base model identity, exact trainable key set and restored parameter digest.

    Validate every persisted state component before mutating the model or optimizer,
    so an internally inconsistent checkpoint cannot leave a partial restore behind.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    base_model_id = _validate_base_model_id(base_model_id)
    path = Path(path)
    manifest = json.loads((path / "checkpoint.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be an object")
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unknown checkpoint schema")
    if manifest.get("base_model_id") != base_model_id:
        raise ValueError("checkpoint base model identity mismatch")
    if manifest.get("optimizer_identity") != _optimizer_identity(optimizer):
        raise ValueError("checkpoint optimizer identity mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"adapters.safetensors", "optimizer.safetensors"}:
        raise ValueError("checkpoint file set is incomplete or unexpected")
    allowed_files = {"checkpoint.json", "adapters.safetensors", "optimizer.safetensors"}
    actual_files = {item.name for item in path.iterdir()}
    if actual_files != allowed_files or any(not (path / name).is_file() for name in allowed_files):
        raise ValueError("checkpoint directory contains missing or unexpected files")
    for name, digest in files.items():
        if _sha256(path / name) != digest:
            raise ValueError(f"checkpoint file hash mismatch: {name}")

    adapters = mx.load(str(path / "adapters.safetensors"))
    expected = dict(tree_flatten(model.trainable_parameters()))
    if set(adapters) != set(expected):
        raise ValueError("checkpoint trainable keys differ from model")

    for key, value in adapters.items():
        if not isinstance(value, mx.array):
            raise ValueError(f"checkpoint parameter is not an MLX array: {key}")
        target = expected[key]
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"checkpoint parameter shape or dtype differs from model: {key}")

    optimizer_arrays = mx.load(str(path / "optimizer.safetensors"))
    if not optimizer_arrays or any(not isinstance(value, mx.array) for value in optimizer_arrays.values()):
        raise ValueError("checkpoint optimizer state contains invalid values")
    restored_optimizer_state = tree_unflatten(list(optimizer_arrays.items()))
    try:
        optimizer_step = int(restored_optimizer_state["step"].item())
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint optimizer state has no valid step") from exc

    step = manifest.get("step")
    manifest_optimizer_step = manifest.get("optimizer_step")
    if (type(step) is not int or step < 0 or type(manifest_optimizer_step) is not int
            or manifest_optimizer_step != optimizer_step or step != optimizer_step):
        raise ValueError("checkpoint update count differs from restored optimizer state")

    rng = manifest.get("rng")
    if (not isinstance(rng, dict) or type(rng.get("base_seed")) is not int
            or rng.get("step") != step
            or rng.get("derivation") != "sha256(f'{base_seed}:{step}')[:8]"):
        raise ValueError("checkpoint RNG state is invalid")
    expected_digest = manifest.get("trainable_digest")
    if not isinstance(expected_digest, str) or _trainable_digest_from_flat(adapters.items()) != expected_digest:
        raise ValueError("restored parameters do not match checkpoint digest")

    mx.eval(adapters, restored_optimizer_state)
    original_parameters = dict(tree_flatten(model.trainable_parameters()))
    original_optimizer_state = optimizer.state
    try:
        model.load_weights(list(adapters.items()), strict=False)
        optimizer.state = restored_optimizer_state
        mx.eval(model.parameters(), optimizer.state)
        if trainable_digest(model) != expected_digest:
            raise ValueError("restored parameters do not match checkpoint digest")
    except Exception:
        model.load_weights(list(original_parameters.items()), strict=False)
        optimizer.state = original_optimizer_state
        mx.eval(model.parameters(), optimizer.state)
        raise

    reseed(rng["base_seed"], rng["step"])
    return manifest


def make_lora_policy(model, *, num_layers: int, rank: int, scale: float, dropout: float = 0.0,
                     seed: int = 0) -> None:
    """Freeze ``model`` and attach mlx-lm LoRA layers (same recipe as the v1
    SFT/RFT arms)."""
    import mlx.core as mx
    from mlx_lm.tuner.utils import linear_to_lora_layers

    mx.random.seed(seed)
    model.freeze()
    linear_to_lora_layers(model, num_layers, {"rank": rank, "scale": scale, "dropout": dropout})
