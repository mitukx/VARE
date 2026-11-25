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
* Log-probabilities are computed on completion tokens only (prompt and padding
  are masked out). The legacy FP32 softmax is the default. An explicit
  MLX-LM-native score mode models the collector's finite-precision sampler
  arithmetic for recorded, lag-zero groups.
* A multi-turn sample may contain independently conditioned action segments.
  Its terminal group advantage is applied to every policy-generated token;
  tool observations are context in the next segment's prompt, never targets.
* The objective is the PPO/GRPO clipped surrogate
  ``min(rho * A, clip(rho, 1 - eps_low, 1 + eps_high) * A)`` with
  ``rho = exp(logp_current - logp_behavior)``, averaged over all completion
  tokens of informative samples in the step. No KL penalty (beta = 0).
* By default the update is strictly on-policy: every sample must come from the
  checkpoint being updated (``max_policy_lag = 0``), so ``rho == 1`` at the
  update point and the gradient equals the REINFORCE-with-group-baseline
  gradient. Recorded sampling log-probs are preserved. Each lag-zero action
  must agree with the differentiated current-policy pass within the declared
  tolerance before an optimizer update; a numerical mismatch is rejected,
  never repaired by replacing the recorded denominator. Legacy samples without
  scores may be scored from their exact behavior checkpoint by the caller.
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

Checkpoints hold the trainable (LoRA) parameters and the full optimizer state.
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
CHECKPOINT_SCHEMA = "vare.learning_pipeline.grpo_checkpoint/v1"


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
LEGACY_SCORE_MODE = "legacy_fp32"
NATIVE_SAMPLER_SCORE_MODE = "mlx_lm_native_v1"


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
    sampling_score_mode: str = LEGACY_SCORE_MODE

    def __post_init__(self) -> None:
        if self.sampling_score_mode not in (LEGACY_SCORE_MODE, NATIVE_SAMPLER_SCORE_MODE):
            raise ValueError("unknown GRPO sampling score mode")
        if self.sampling_score_mode == NATIVE_SAMPLER_SCORE_MODE and self.max_policy_lag != 0:
            raise ValueError("native sampler scoring is currently restricted to lag-zero groups")

    def to_dict(self) -> dict[str, Any]:
        values = {k: getattr(self, k) for k in self.__dataclass_fields__}
        # Preserve existing serialized configuration for legacy experiments.
        if self.sampling_score_mode == LEGACY_SCORE_MODE:
            values.pop("sampling_score_mode")
        return values


@dataclass
class TrajectorySegment:
    """One policy-generated action and the exact context it followed.

    ``prompt_ids`` must include any tool or environment observations preceding
    this action. ``completion_ids`` contains policy-generated tokens only.
    """

    prompt_ids: list[int]
    completion_ids: list[int]
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
    # One terminal reward may credit several independently conditioned turns.
    # The first segment mirrors the legacy prompt/action fields above.
    trajectory_segments: list[TrajectorySegment] | None = None

    @property
    def group_key(self) -> tuple[str, str]:
        return (self.instance_id, self.behavior_checkpoint_id)


def _sample_segments(sample: Sample) -> Sequence[Sample | TrajectorySegment]:
    return sample.trajectory_segments if sample.trajectory_segments is not None else (sample,)


def _training_action_segments(
    work: Sequence[tuple[Sample, float]],
) -> Iterable[tuple[Sample, float, Sample | TrajectorySegment]]:
    """Enumerate every policy action with the trajectory's terminal advantage."""
    for sample, advantage in work:
        for segment in _sample_segments(sample):
            yield sample, advantage, segment


def _credit_assignment(samples: Sequence[Sample]) -> str:
    modes = {sample.trajectory_segments is not None for sample in samples}
    if not modes:
        return "no_samples"
    if modes == {True}:
        return "full_trajectory_terminal_reward"
    if modes == {False}:
        return "single_sequence_terminal_reward"
    return "mixed_single_sequence_and_full_trajectory_terminal_reward"


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


def _validate_behavior_scores(values: list[float] | None, count: int, key: Any) -> None:
    if values is None:
        return
    if len(values) != count:
        raise GroupError(f"behavior log-probability length mismatch in group {key}")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value > 0 for value in values):
        raise GroupError(f"behavior log-probabilities must be finite nonpositive numbers in group {key}")


def group_samples(samples: Iterable[Sample], *, group_size: int | None = None) -> dict[tuple[str, str], list[Sample]]:
    groups: dict[tuple[str, str], list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.group_key, []).append(s)
    for key, g in groups.items():
        idx = [s.sample_index for s in g]
        if len(set(idx)) != len(idx):
            raise GroupError(f"duplicate sample_index in group {key}")
        if any(s.meta.get("behavior_logprob_source") == "sampling_distribution" for s in g):
            sizes = [s.meta.get("behavior_group_size") for s in g]
            if (any(s.meta.get("behavior_logprob_source") != "sampling_distribution" for s in g)
                    or any(type(size) is not int or size < 2 for size in sizes)
                    or any(size != sizes[0] for size in sizes)
                    or len(g) != sizes[0]
                    or any(type(index) is not int for index in idx)
                    or sorted(idx) != list(range(len(g)))):
                raise GroupError(f"partial or inconsistent recorded trajectory group {key}")
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
            segments = _sample_segments(s)
            if not segments:
                raise GroupError(f"empty trajectory in group {key}")
            if s.trajectory_segments is not None and (
                segments[0].prompt_ids != s.prompt_ids
                or segments[0].completion_ids != s.completion_ids
            ):
                raise GroupError(f"first trajectory segment differs from the sample action in group {key}")
            if s.trajectory_segments is not None and s.behavior_logprobs is not None:
                _validate_behavior_scores(s.behavior_logprobs, len(s.completion_ids), key)
                if (
                    segments[0].behavior_logprobs is not None
                    and s.behavior_logprobs != segments[0].behavior_logprobs
                ):
                    raise GroupError(f"first segment behavior log-probabilities differ from sample in group {key}")
            for segment in segments:
                if not segment.completion_ids:
                    raise GroupError(f"empty completion in group {key}")
                if not segment.prompt_ids or any(
                    type(token) is not int or token < 0
                    for token in (*segment.prompt_ids, *segment.completion_ids)
                ):
                    raise GroupError(f"prompt and completion tokens must be integer IDs in group {key}")
                _validate_behavior_scores(segment.behavior_logprobs, len(segment.completion_ids), key)
    return groups


def check_policy_lag(samples: Iterable[Sample], *, policy_checkpoint_id: str, policy_step: int,
                     max_policy_lag: int) -> int:
    if (type(policy_step) is not int or policy_step < 0
            or type(max_policy_lag) is not int or max_policy_lag < 0
            or not isinstance(policy_checkpoint_id, str) or not policy_checkpoint_id):
        raise StalePolicyError("current checkpoint, policy step, and maximum lag must be valid")
    worst = 0
    for s in samples:
        if type(s.behavior_step) is not int or s.behavior_step < 0:
            raise StalePolicyError("behavior step must be a nonnegative integer")
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
    """Reject invalid ratios at every lag, and mismatched lag-zero scores.

    Checkpoint labels alone do not establish the sampling distribution. This
    check uses each action segment's differentiated current-policy scores.
    Numerical differences in a real backend require a separate investigation;
    they do not authorize replacing recorded sampling log-probabilities.
    """
    if type(policy_lag) is not int or policy_lag < 0:
        raise StalePolicyError("policy lag must be a nonnegative integer")
    if (isinstance(max_abs_log_ratio, bool) or not isinstance(max_abs_log_ratio, (int, float))
            or not math.isfinite(max_abs_log_ratio) or max_abs_log_ratio < 0):
        raise StalePolicyError("behavior log-ratio must be finite and nonnegative")
    if policy_lag == 0 and max_abs_log_ratio > ON_POLICY_LOG_RATIO_TOLERANCE:
        raise StalePolicyError(
            "lag-zero behavior log-probabilities do not match the current policy: "
            f"max_abs_log_ratio={max_abs_log_ratio!r}, tolerance={ON_POLICY_LOG_RATIO_TOLERANCE}"
        )


def _check_behavior_sampling(samples: Iterable[Sample], cfg: GRPOConfig) -> None:
    """Bind a recorded collector's sampler to the distribution in the loss."""
    check_exact_ratio_sampling(temperature=cfg.temperature, top_p=1.0)
    for sample in samples:
        if (cfg.sampling_score_mode == NATIVE_SAMPLER_SCORE_MODE
                and sample.meta.get("behavior_logprob_source") != "sampling_distribution"):
            raise GroupError("native sampler scoring requires recorded sampling-distribution scores")
        if ("behavior_temperature" not in sample.meta
                and "behavior_top_p" not in sample.meta
                and sample.meta.get("behavior_logprob_source") != "sampling_distribution"):
            continue  # Legacy manually constructed samples have no collector metadata.
        temperature, top_p = sample.meta.get("behavior_temperature"), sample.meta.get("behavior_top_p")
        if not isinstance(temperature, (int, float)) or not isinstance(top_p, (int, float)):
            raise TruncatedSamplingError("recorded behavior sampler metadata is missing or nonnumeric")
        check_exact_ratio_sampling(
            temperature=temperature, top_p=top_p,
            top_k=sample.meta.get("behavior_top_k", 0), min_p=sample.meta.get("behavior_min_p", 0.0),
        )
        if temperature != cfg.temperature:
            raise TruncatedSamplingError("recorded behavior temperature differs from the GRPO loss")
        if sample.meta.get("behavior_logprob_source") == "sampling_distribution" and any(
            segment.behavior_logprobs is None for segment in _sample_segments(sample)
        ):
            raise GroupError("recorded sampling behavior log-probs are missing")


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


def token_logprobs(model, tokens, *, temperature: float = 1.0, start: int = 0,
                   sampling_score_mode: str = LEGACY_SCORE_MODE):
    """Completion log-probs [B, L-1-start], with explicit numerical score semantics.

    The native mode mirrors the MLX-LM 0.32.2 untruncated sampler used by
    TraceableMLXPolicy: center in the model's output dtype, multiply by
    reciprocal temperature in that dtype, promote to FP32, then normalize.
    It does not bypass the differentiated lag-zero guard or replace any
    behavior denominator. The legacy path is unchanged.
    """
    import mlx.core as mx

    logits = _logits(model, tokens[:, :-1], start)
    if sampling_score_mode == NATIVE_SAMPLER_SCORE_MODE:
        check_exact_ratio_sampling(temperature=temperature, top_p=1.0)
        native_logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logits = (native_logprobs * (1.0 / temperature)).astype(mx.float32)
    elif sampling_score_mode == LEGACY_SCORE_MODE:
        logits = logits.astype(mx.float32)
        if temperature != 1.0:
            logits = logits / temperature
    else:
        raise ValueError("unknown GRPO sampling score mode")
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


def forward_logprobs(model, s: Sample, *, temperature: float, pad_id: int = 0,
                     sampling_score_mode: str = LEGACY_SCORE_MODE) -> list[float]:
    """Plain forward log-probs for every policy action token in ``s``.

    Not used for the ratio: on Metal/bf16 it can differ from the differentiated
    forward by ~0.1 nats on single tokens (see ``compute_behavior_logprobs``).
    """
    import mlx.core as mx

    result: list[float] = []
    for segment in _sample_segments(s):
        tokens, _, start = sample_tensors(segment, pad_id)
        lp = token_logprobs(model, tokens, temperature=temperature, start=start,
                            sampling_score_mode=sampling_score_mode)
        mx.eval(lp)
        result.extend(float(x) for x in lp[0, : len(segment.completion_ids)].tolist())
    return result



def _alignment_metric(values: Sequence[float], *, tolerance: float) -> dict[str, float | int]:
    """Conservative nearest-rank quantiles for absolute log-probability differences."""
    ordered = sorted(values)
    if not ordered or any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("alignment differences must be finite nonnegative values")
    return {
        "max": ordered[-1],
        "p50": ordered[math.ceil(len(ordered) * 0.50) - 1],
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "n_over_tolerance": sum(value > tolerance for value in ordered),
    }


def audit_on_policy_sampling_alignment(
    model,
    samples: Sequence[Sample],
    cfg: GRPOConfig,
    *,
    policy_checkpoint_id: str,
    policy_step: int,
    max_action_segments: int = 32,
    max_completion_tokens: int = 2048,
    max_sequence_tokens: int = 8192,
    pad_id: int = 0,
    tolerance: float = ON_POLICY_LOG_RATIO_TOLERANCE,
) -> dict[str, Any]:
    """Read-only comparison of recorded sampling scores to both forward paths.

    Uses the same differentiated forward as policy_gradient but never invokes
    optimizer.update. No parameter, checkpoint, or sample is changed. Callers
    must load the exact behavior checkpoint and preserve its runtime mode; the
    checkpoint ID and step labels alone do not authenticate those bytes.

    This is an integrity diagnostic, not authorization for training and not a
    model/task-success evaluation. All members of every recorded group are
    required even though the diagnostic does not use terminal rewards.
    """
    limits = {
        "max_action_segments": max_action_segments,
        "max_completion_tokens": max_completion_tokens,
        "max_sequence_tokens": max_sequence_tokens,
    }
    if any(type(value) is not int or value <= 0 for value in limits.values()):
        raise ValueError("alignment budgets must be positive integers")
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError("alignment tolerance must be finite and nonnegative")

    groups = group_samples(samples, group_size=cfg.group_size)
    if not groups:
        raise GroupError("alignment requires a complete nonempty group")
    check_policy_lag(
        samples, policy_checkpoint_id=policy_checkpoint_id,
        policy_step=policy_step, max_policy_lag=0,
    )
    _check_behavior_sampling(samples, cfg)
    if any(sample.meta.get("behavior_logprob_source") != "sampling_distribution" for sample in samples):
        raise GroupError("alignment requires sampled, not recomputed, behavior log-probabilities")
    actions = [(sample, segment) for sample in samples for segment in _sample_segments(sample)]
    token_count = sum(len(segment.completion_ids) for _, segment in actions)
    if (len(actions) > max_action_segments or token_count > max_completion_tokens
            or any(len(segment.prompt_ids) + len(segment.completion_ids) > max_sequence_tokens
                   for _, segment in actions)):
        raise ValueError("alignment action, completion, or sequence budget exceeded")
    if any(segment.behavior_logprobs is None for _, segment in actions):
        raise GroupError("alignment requires recorded behavior scores for every action")

    import mlx.core as mx

    value_and_grad = _value_and_grad(model, cfg, n_tokens=1)
    sample_to_plain: list[float] = []
    sample_to_traced: list[float] = []
    plain_to_traced: list[float] = []
    for _, segment in actions:
        tokens, mask, start = sample_tensors(segment, pad_id)
        plain = token_logprobs(model, tokens, temperature=cfg.temperature, start=start,
                               sampling_score_mode=cfg.sampling_score_mode)
        zeros = mx.zeros(mask.shape, dtype=mx.float32)
        (_, (_, traced)), gradients = value_and_grad(
            tokens, mask, start, mx.zeros((1,), dtype=mx.float32), zeros,
        )
        mx.eval(plain, traced, gradients)
        n = len(segment.completion_ids)
        plain_values = [float(value) for value in plain[0, :n].tolist()]
        traced_values = [float(value) for value in traced[0, :n].tolist()]
        recorded_values = segment.behavior_logprobs
        if recorded_values is None or len(plain_values) != n or len(traced_values) != n:
            raise ValueError("alignment score vector length mismatch")
        for recorded, forward, differentiated in zip(recorded_values, plain_values, traced_values):
            if not all(math.isfinite(value) for value in (recorded, forward, differentiated)):
                raise FloatingPointError("nonfinite on-policy alignment score")
            sample_to_plain.append(abs(recorded - forward))
            sample_to_traced.append(abs(recorded - differentiated))
            plain_to_traced.append(abs(forward - differentiated))

    metrics = {
        "sampled_vs_forward": _alignment_metric(sample_to_plain, tolerance=tolerance),
        "sampled_vs_differentiated": _alignment_metric(sample_to_traced, tolerance=tolerance),
        "forward_vs_differentiated": _alignment_metric(plain_to_traced, tolerance=tolerance),
    }
    return {
        "schema": "vare.grpo.sampling_alignment/v1",
        "policy_checkpoint_id": policy_checkpoint_id,
        "policy_step": policy_step,
        "temperature": cfg.temperature,
        "sampling_score_mode": cfg.sampling_score_mode,
        "tolerance": tolerance,
        "n_groups": len(groups),
        "n_samples": len(samples),
        "n_action_segments": len(actions),
        "n_completion_tokens": token_count,
        "model_training_mode": getattr(model, "training", None),
        "limits": limits,
        "metrics": metrics,
        "would_reject_lag_zero_update": metrics["sampled_vs_differentiated"]["n_over_tolerance"] > 0,
        "optimizer_updated": False,
        "task_success_measured": False,
    }

def compute_behavior_logprobs(model, samples: Iterable[Sample], cfg: GRPOConfig | None = None, *,
                              temperature: float | None = None, pad_id: int = 0,
                              after_sample: Callable[[], None] | None = None) -> None:
    """Fill missing legacy scores without replacing recorded behavior scores.

    The caller must own the exact behavior checkpoint while scoring missing
    values. Recorded sampling scores are retained and checked in the update
    path, not replaced with current-policy scores to force a ratio of one.
    A historical 4B bf16 run found forward-path differences up to 0.13 nats;
    that observation does not justify silently changing a sampled denominator.
    Missing values are staged so a later failure cannot leave a partial fill.
    """
    if cfg is None:
        if temperature is None:
            raise ValueError("pass cfg or temperature")
        cfg = GRPOConfig(temperature=temperature)
    samples = list(samples)
    group_samples(samples, group_size=cfg.group_size)
    _check_behavior_sampling(samples, cfg)
    missing = [(s, index, segment) for s in samples
               for index, segment in enumerate(_sample_segments(s))
               if segment.behavior_logprobs is None]
    staged: list[tuple[Sample | TrajectorySegment, list[float]]] = []
    if missing:
        import mlx.core as mx

        value_and_grad = _value_and_grad(model, cfg, n_tokens=1)
        for s, index, segment in missing:
            # The legacy first-action mirror is also a recorded value.
            if index == 0 and isinstance(segment, TrajectorySegment) and s.behavior_logprobs is not None:
                staged.append((segment, list(s.behavior_logprobs)))
                continue
            tokens, mask, start = sample_tensors(segment, pad_id)
            zeros = mx.zeros(mask.shape, dtype=mx.float32)
            (_, (_, lp)), g = value_and_grad(tokens, mask, start, mx.zeros((1,), dtype=mx.float32), zeros)
            mx.eval(lp, g)
            values = [float(x) for x in lp[0, : len(segment.completion_ids)].tolist()]
            _validate_behavior_scores(values, len(segment.completion_ids), s.group_key)
            staged.append((segment, values))
            if after_sample is not None:
                after_sample()
    for segment, values in staged:
        segment.behavior_logprobs = values
    for s in samples:
        if s.trajectory_segments is not None and s.behavior_logprobs is None:
            first = s.trajectory_segments[0].behavior_logprobs
            s.behavior_logprobs = list(first) if first is not None else None


# ------------------------------------------------------------------ update
def trainable_digest(model) -> str:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    h = hashlib.sha256()
    for k, v in sorted(tree_flatten(model.trainable_parameters())):
        h.update(k.encode())
        h.update(np.asarray(v.astype(mx.float32)).tobytes())
    return h.hexdigest()


def _value_and_grad(model, cfg: GRPOConfig, *, n_tokens: int):
    """The one traced loss used both for behavior log-probs and for the update."""
    import mlx.nn as nn

    def loss_fn(tokens, mask, start, adv, blp):
        lp = token_logprobs(model, tokens, temperature=cfg.temperature, start=start,
                            sampling_score_mode=cfg.sampling_score_mode)
        loss, stats = clipped_surrogate(lp, blp, adv, mask, eps_low=cfg.clip_eps_low, eps_high=cfg.clip_eps_high,
                                        n_tokens=float(n_tokens))
        return loss, (stats, lp)

    return nn.value_and_grad(model, loss_fn)


def policy_gradient(model, work: Sequence[tuple[Sample, float]], cfg: GRPOConfig, *, n_tokens: int,
                    policy_checkpoint_id: str, policy_step: int, pad_id: int = 0,
                    after_sample: Callable[[], None] | None = None) -> tuple[Any, dict[str, float]]:
    """Gradient of the step loss, accumulated one action segment at a time.

    For trajectories, each segment receives its episode's terminal advantage.
    The accumulated result equals the full-batch token-mean loss gradient.

    ``after_sample`` (optional) runs after each microbatch is evaluated, e.g. to
    release MLX's buffer cache or check a resource limit; it cannot change the
    gradient, and an exception from it aborts the step before any update."""
    samples = [sample for sample, _ in work]
    group_samples(samples, group_size=cfg.group_size)
    check_policy_lag(samples, policy_checkpoint_id=policy_checkpoint_id,
                     policy_step=policy_step, max_policy_lag=cfg.max_policy_lag)
    _check_behavior_sampling(samples, cfg)
    if type(n_tokens) is not int or n_tokens <= 0:
        raise ValueError("n_tokens must be a positive integer")
    import mlx.core as mx
    from mlx.utils import tree_map

    value_and_grad = _value_and_grad(model, cfg, n_tokens=n_tokens)
    grads: Any = None
    loss_total = clipped = ratio_sum = kl_sum = ratio_max = max_lr = 0.0
    for s, a, segment in _training_action_segments(work):
        behavior_logprobs = (
            segment.behavior_logprobs if isinstance(segment, TrajectorySegment)
            else s.behavior_logprobs
        )
        if behavior_logprobs is None:
            raise ValueError("missing behavior log-probs")
        tokens, mask, start = sample_tensors(segment, pad_id)
        blp = mx.array([behavior_logprobs], dtype=mx.float32)
        (loss, (st, _lp)), g = value_and_grad(tokens, mask, start, mx.array([a], dtype=mx.float32), blp)
        mx.eval(loss, st, g)
        check_on_policy_log_ratio(policy_step - s.behavior_step, float(st["max_abs_log_ratio"]))
        if not math.isfinite(float(loss)) or any(not math.isfinite(float(value)) for value in st.values()):
            raise FloatingPointError("non-finite GRPO loss or importance-ratio diagnostics")
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
    groups = group_samples(samples, group_size=cfg.group_size)
    lag = check_policy_lag(samples, policy_checkpoint_id=policy_checkpoint_id, policy_step=policy_step,
                           max_policy_lag=cfg.max_policy_lag)
    _check_behavior_sampling(samples, cfg)
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
        "n_trained_action_segments": sum(1 for _ in _training_action_segments(work)),
        "credit_assignment": _credit_assignment(samples),
    }
    action_segments = list(_training_action_segments(work))
    for sample, _, segment in action_segments:  # zero-advantage samples need no log-probs
        behavior_logprobs = (
            segment.behavior_logprobs if isinstance(segment, TrajectorySegment)
            else sample.behavior_logprobs
        )
        if behavior_logprobs is None or len(behavior_logprobs) != len(segment.completion_ids):
            raise ValueError(f"missing behavior log-probs for {sample.instance_id}#{sample.sample_index}")
    n_tokens = sum(len(segment.completion_ids) for _, _, segment in action_segments)
    record["n_tokens"] = n_tokens
    if not work:
        record.update(update_applied=False, reason="no_informative_group")
        return record

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    grads, diag = policy_gradient(
        model, work, cfg, n_tokens=n_tokens, policy_checkpoint_id=policy_checkpoint_id,
        policy_step=policy_step, pad_id=pad_id, after_sample=after_sample,
    )
    if any(not bool(mx.all(mx.isfinite(value)).item()) for _, value in tree_flatten(grads)):
        raise FloatingPointError("non-finite GRPO gradient")
    grad_norm = None
    if cfg.max_grad_norm is not None:
        grads, gn = optim.clip_grad_norm(grads, cfg.max_grad_norm)
        grad_norm = float(gn)
        if (not math.isfinite(grad_norm)
                or any(not bool(mx.all(mx.isfinite(value)).item()) for _, value in tree_flatten(grads))):
            raise FloatingPointError("non-finite clipped GRPO gradient")
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


def save_checkpoint(path: Path, model, optimizer, *, step: int, base_seed: int,
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Atomically write adapters, optimizer state and the RNG schedule to
    ``path``. ``step`` is the number of completed updates."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"checkpoint destination already exists: {path}")
    optimizer_step = int(optimizer.state["step"].item())
    if type(step) is not int or step < 0 or optimizer_step != step:
        raise ValueError(f"checkpoint step {step!r} does not match optimizer step {optimizer_step}")
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
    manifest = {"schema": CHECKPOINT_SCHEMA, "step": step, "optimizer_step": optimizer_step, "files": files,
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


def load_checkpoint(path: Path, model, optimizer) -> dict[str, Any]:
    """Restore a checkpoint written by ``save_checkpoint``; verifies hashes,
    the exact trainable key set and the restored parameter digest."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    path = Path(path)
    manifest = json.loads((path / "checkpoint.json").read_text())
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unknown checkpoint schema")
    if set(manifest.get("files", {})) != {"adapters.safetensors", "optimizer.safetensors"}:
        raise ValueError("checkpoint file set is incomplete or unexpected")
    allowed_files = {"checkpoint.json", "adapters.safetensors", "optimizer.safetensors"}
    actual_files = {item.name for item in path.iterdir()}
    if actual_files != allowed_files or any(not (path / name).is_file() for name in allowed_files):
        raise ValueError("checkpoint directory contains missing or unexpected files")
    for name, digest in manifest["files"].items():
        if _sha256(path / name) != digest:
            raise ValueError(f"checkpoint file hash mismatch: {name}")
    adapters = mx.load(str(path / "adapters.safetensors"))
    expected = {k for k, _ in tree_flatten(model.trainable_parameters())}
    if set(adapters) != expected:
        raise ValueError("checkpoint trainable keys differ from model")
    model.load_weights(list(adapters.items()), strict=False)
    optimizer.state = tree_unflatten(list(mx.load(str(path / "optimizer.safetensors")).items()))
    mx.eval(model.parameters(), optimizer.state)
    optimizer_step = int(optimizer.state["step"].item())
    if manifest.get("optimizer_step") != optimizer_step or manifest.get("step") != optimizer_step:
        raise ValueError("checkpoint update count differs from restored optimizer state")
    reseed(manifest["rng"]["base_seed"], manifest["rng"]["step"])
    if trainable_digest(model) != manifest["trainable_digest"]:
        raise ValueError("restored parameters do not match checkpoint digest")
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
