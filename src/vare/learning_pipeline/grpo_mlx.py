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
  are masked out) from ``logits / temperature``.
* The objective is the PPO/GRPO clipped surrogate
  ``min(rho * A, clip(rho, 1 - eps_low, 1 + eps_high) * A)`` with
  ``rho = exp(logp_current - logp_behavior)``, averaged over all completion
  tokens of informative samples in the step. No KL penalty (beta = 0).
* By default the update is strictly on-policy: every sample must come from the
  checkpoint being updated (``max_policy_lag = 0``), so ``rho == 1`` at the
  update point and the gradient equals the REINFORCE-with-group-baseline
  gradient. Stale samples raise ``StalePolicyError`` instead of being reused.
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
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "vare.learning_pipeline.grpo_step/v1"
CHECKPOINT_SCHEMA = "vare.learning_pipeline.grpo_checkpoint/v1"


class StalePolicyError(ValueError):
    """A sample's behavior policy is older than the configured lag allows."""


class GroupError(ValueError):
    """Samples cannot form valid GRPO groups."""


@dataclass(frozen=True)
class GRPOConfig:
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.2
    normalize_std: bool = True
    temperature: float = 1.0
    max_policy_lag: int = 0
    max_grad_norm: float | None = 1.0
    group_size: int | None = None  # if set, every group must have exactly this many samples

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


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

    @property
    def group_key(self) -> tuple[str, str]:
        return (self.instance_id, self.behavior_checkpoint_id)


# ------------------------------------------------------------------ groups
def group_advantages(rewards: Sequence[float], *, normalize_std: bool = True) -> tuple[list[float], bool]:
    """Return (advantages, informative). Zero-variance groups give all zeros."""
    if not rewards:
        raise GroupError("empty group")
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
        if len({s.verifier for s in g}) != 1:
            raise GroupError(f"mixed verifier versions in group {key}")
        if len({s.behavior_step for s in g}) != 1:
            raise GroupError(f"mixed behavior steps in group {key}")
        if group_size is not None and len(g) != group_size:
            raise GroupError(f"partial group {key}: {len(g)} != {group_size}")
        for s in g:
            if not s.completion_ids:
                raise GroupError(f"empty completion in group {key}")
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
    }
    return loss, stats


def sample_tensors(s: Sample, pad_id: int):
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


def compute_behavior_logprobs(model, samples: Iterable[Sample], *, temperature: float, pad_id: int = 0) -> None:
    """Fill ``behavior_logprobs`` with the current (behavior) parameters,
    without gradients. Call before any update that uses these samples."""
    import mlx.core as mx

    for s in samples:
        tokens, mask, start = sample_tensors(s, pad_id)
        lp = mx.stop_gradient(token_logprobs(model, tokens, temperature=temperature, start=start))
        mx.eval(lp)
        s.behavior_logprobs = [float(x) for x in lp[0, : len(s.completion_ids)].tolist()]


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


def policy_gradient(model, work: Sequence[tuple[Sample, float]], cfg: GRPOConfig, *, n_tokens: int,
                    pad_id: int = 0) -> tuple[Any, dict[str, float]]:
    """Gradient of the step loss w.r.t. trainable parameters, accumulated one
    sample at a time; equals the gradient of the full-batch token-mean loss."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map

    def loss_fn(tokens, mask, start, adv, blp):
        lp = token_logprobs(model, tokens, temperature=cfg.temperature, start=start)
        return clipped_surrogate(lp, blp, adv, mask, eps_low=cfg.clip_eps_low, eps_high=cfg.clip_eps_high,
                                 n_tokens=float(n_tokens))

    value_and_grad = nn.value_and_grad(model, loss_fn)
    grads: Any = None
    loss_total = clipped = ratio_sum = kl_sum = ratio_max = 0.0
    for s, a in work:
        if s.behavior_logprobs is None:
            raise ValueError("missing behavior log-probs")
        tokens, mask, start = sample_tensors(s, pad_id)
        blp = mx.array([s.behavior_logprobs], dtype=mx.float32)
        (loss, st), g = value_and_grad(tokens, mask, start, mx.array([a], dtype=mx.float32), blp)
        grads = g if grads is None else tree_map(lambda x, y: x + y, grads, g)
        mx.eval(loss, st, grads)
        loss_total += float(loss)
        clipped += float(st["clipped_tokens"])
        ratio_sum += float(st["ratio_sum"])
        kl_sum += float(st["kl_sum"])
        ratio_max = max(ratio_max, float(st["ratio_max"]))
    diag = {"loss": loss_total, "clip_fraction": clipped / n_tokens, "ratio_mean": ratio_sum / n_tokens,
            "ratio_max": ratio_max, "approx_kl": kl_sum / n_tokens}
    return grads, diag


def train_step(model, optimizer, samples: Sequence[Sample], cfg: GRPOConfig, *, policy_checkpoint_id: str,
               policy_step: int, pad_id: int = 0) -> dict[str, Any]:
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
    }
    for s, _ in work:  # zero-advantage samples need no log-probs
        if s.behavior_logprobs is None or len(s.behavior_logprobs) != len(s.completion_ids):
            raise ValueError(f"missing behavior log-probs for {s.instance_id}#{s.sample_index}")
    n_tokens = sum(len(s.completion_ids) for s, _ in work)
    record["n_tokens"] = n_tokens
    if not work:
        record.update(update_applied=False, reason="no_informative_group")
        return record

    grads, diag = policy_gradient(model, work, cfg, n_tokens=n_tokens, pad_id=pad_id)
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


def save_checkpoint(path: Path, model, optimizer, *, step: int, base_seed: int,
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Atomically write adapters, optimizer state and the RNG schedule to
    ``path``. ``step`` is the number of completed updates."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    path = Path(path)
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
    manifest = {"schema": CHECKPOINT_SCHEMA, "step": step, "files": files,
                "rng": {"base_seed": base_seed, "step": step, "derivation": "sha256(f'{base_seed}:{step}')[:8]"},
                "trainable_digest": trainable_digest(model), "extra": dict(extra or {})}
    (tmp / "checkpoint.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if path.exists():
        shutil.rmtree(path)
    os.replace(tmp, path)
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
