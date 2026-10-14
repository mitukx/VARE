"""MLX LoRA GRPO step for a chat model. Imported only when a run starts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minigrpo.grpo import LOG_RATIO_LIMIT, LORA_KEYS
from minigrpo.loop import Rollout, RunConfig


def _ints(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]
    return [int(token) for token in value]


class MlxTrainer:
    def __init__(self, config: RunConfig) -> None:
        import mlx.core as mx  # type: ignore[import-not-found]
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]

        self.config = config
        self.mx = mx
        self.local = (
            snapshot_download(config.model, revision=config.revision) if config.revision else config.model
        )
        self.sampler = make_sampler(temp=config.temperature)
        self.model: Any = None
        self.tokenizer: Any = None
        self.optimizer: Any = None
        self.ref_model: Any = None

    def activate(self, adapter: Path | None) -> None:
        """Load the base model, optionally with saved LoRA weights, for generation."""
        self._reload_base()
        if adapter is not None:
            self._load_adapter(adapter, load_optimizer=False)
        self.model.eval()

    def prepare(self, resume: Path | None) -> None:
        self._reload_base()
        if resume is not None:
            self._load_adapter(resume, load_optimizer=True)
        elif self.config.init_adapter:
            self._load_adapter(Path(self.config.init_adapter), load_optimizer=False)
        else:
            self._init_fresh()
        if self.config.kl_coef > 0.0 and self.ref_model is None:
            self._load_reference()

    def _reload_base(self) -> None:
        from mlx_lm.utils import load  # type: ignore[import-not-found]

        self.model, self.tokenizer = load(self.local)
        if not getattr(self.tokenizer, "chat_template", None):
            raise RuntimeError("model tokenizer has no chat template")
        self.model.freeze()
        self.optimizer = None

    def _lora_config(self) -> dict[str, object]:
        return {
            "dropout": self.config.lora_dropout,
            "keys": list(LORA_KEYS),
            "rank": self.config.lora_rank,
            "scale": self.config.lora_scale,
        }

    def _init_fresh(self) -> None:
        import mlx.optimizers as optim  # type: ignore[import-not-found]
        from mlx_lm.tuner.utils import linear_to_lora_layers  # type: ignore[import-not-found]

        self.mx.random.seed(self.config.seed)
        linear_to_lora_layers(self.model, self.config.num_layers, self._lora_config())
        self.optimizer = optim.Adam(learning_rate=self.config.learning_rate)

    def _load_reference(self) -> None:
        from mlx_lm.utils import load  # type: ignore[import-not-found]

        reference, _tokenizer = load(self.local)
        reference.freeze()
        reference.eval()
        self.ref_model = reference

    def _load_adapter(self, directory: Path, *, load_optimizer: bool) -> None:
        import mlx.optimizers as optim  # type: ignore[import-not-found]
        from mlx.utils import tree_unflatten  # type: ignore[import-not-found]
        from mlx_lm.tuner.utils import load_adapters  # type: ignore[import-not-found]

        load_adapters(self.model, str(directory))
        self.optimizer = optim.Adam(learning_rate=self.config.learning_rate)
        optimizer_path = directory / "optimizer.safetensors"
        if load_optimizer:
            if not optimizer_path.exists():
                raise RuntimeError(f"missing optimizer state in {directory}")
            state = self.mx.load(str(optimizer_path))
            self.optimizer.state = tree_unflatten(list(state.items()))

    def _prompt_text(self, prompt: str) -> str:
        return str(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def _encode_pair(self, prompt: str, completion: str) -> tuple[tuple[int, ...], int]:
        full = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
            add_generation_prompt=False,
            return_dict=False,
        )
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_dict=False,
        )
        tokens = _ints(full)
        offset = len(_ints(prompt_ids))
        if offset < 1 or offset >= len(tokens) or len(tokens) > self.config.max_seq_length:
            raise RuntimeError("prompt mask leaves no completion tokens inside the length cap")
        return tuple(tokens), offset

    def _token_logp(self, model: Any, token_ids: tuple[int, ...], prompt_length: int, *, entropy: bool) -> tuple[Any, Any, Any]:
        import mlx.nn as nn  # type: ignore[import-not-found]

        ids = self.mx.array(list(token_ids))[None]
        logits = model(ids)[:, :-1, :]
        targets = ids[:, 1:]
        log_probs = nn.log_softmax(logits, axis=-1)
        token_logp = self.mx.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)[0]
        positions = self.mx.arange(targets.shape[1])
        mask = (positions >= (prompt_length - 1)).astype(token_logp.dtype)
        if not entropy:
            return token_logp, mask, None
        probs = self.mx.softmax(logits, axis=-1)
        token_entropy = -self.mx.sum(probs * log_probs, axis=-1)[0]
        return token_logp, mask, token_entropy

    def _tolist(self, array: Any) -> tuple[float, ...]:
        self.mx.eval(array)
        return tuple(float(value) for value in array.tolist())

    def generate(self, prompt: str, seed: int) -> str:
        from mlx_lm import generate  # type: ignore[import-not-found]

        if self.model is None:
            raise RuntimeError("trainer is not prepared")
        self.model.eval()
        self.mx.random.seed(seed)
        text = str(
            generate(
                self.model,
                self.tokenizer,
                prompt=self._prompt_text(prompt),
                max_tokens=self.config.max_new_tokens,
                sampler=self.sampler,
                verbose=False,
            )
        )
        if not text.strip():
            raise RuntimeError("empty completion; refusing to drop a group member")
        return text

    def encode(self, prompt: str, completion: str) -> Rollout:
        if self.model is None:
            raise RuntimeError("trainer is not prepared")
        tokens, offset = self._encode_pair(prompt, completion)
        self.model.eval()
        logp, mask, entropy = self._token_logp(self.model, tokens, offset, entropy=True)
        ref = None
        if self.ref_model is not None:
            ref_logp, _ref_mask, _ref_entropy = self._token_logp(self.ref_model, tokens, offset, entropy=False)
            ref = self._tolist(ref_logp)
        return Rollout(
            text=completion,
            old_logp=self._tolist(logp),
            mask=tuple(int(bit) for bit in self._tolist(mask)),
            entropy=self._tolist(entropy),
            ref_logp=ref,
            token_ids=tokens,
            prompt_length=offset,
        )

    def update(self, rollouts: list[Rollout], advantages: list[float], *, epochs: int) -> float:
        import mlx.nn as nn  # type: ignore[import-not-found]

        if self.optimizer is None:
            raise RuntimeError("trainer is not prepared")
        last = 0.0
        for _epoch in range(epochs):
            def loss_fn(model: Any) -> Any:
                total = self.mx.array(0.0)
                for rollout, advantage in zip(rollouts, advantages, strict=True):
                    logp, mask, _entropy = self._token_logp(
                        model, rollout.token_ids, rollout.prompt_length, entropy=False
                    )
                    old = self.mx.array(rollout.old_logp)
                    delta = self.mx.clip(logp - old, -LOG_RATIO_LIMIT, LOG_RATIO_LIMIT)
                    ratio = self.mx.exp(delta)
                    adv = self.mx.array(advantage)
                    unclipped = ratio * adv
                    clipped = self.mx.clip(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv
                    per_token = -self.mx.minimum(unclipped, clipped)
                    if rollout.ref_logp is not None and self.config.kl_coef > 0.0:
                        ref = self.mx.array(rollout.ref_logp)
                        kl_delta = self.mx.clip(ref - logp, -LOG_RATIO_LIMIT, LOG_RATIO_LIMIT)
                        per_token = per_token + self.config.kl_coef * (self.mx.exp(kl_delta) - kl_delta - 1.0)
                    active = per_token * mask
                    if self.config.dr_grpo:
                        total = total + self.mx.sum(active)
                    else:
                        total = total + self.mx.sum(active) / self.mx.maximum(self.mx.sum(mask), self.mx.array(1.0))
                return total / len(rollouts)

            self.model.train()
            value_and_grad = nn.value_and_grad(self.model, loss_fn)
            loss, grads = value_and_grad(self.model)
            self.optimizer.update(self.model, grads)
            self.mx.eval(self.model.parameters(), self.optimizer.state, loss)
            last = float(loss.item())
        return last

    def save(self, directory: Path) -> None:
        from mlx.utils import tree_flatten  # type: ignore[import-not-found]

        if self.optimizer is None:
            raise RuntimeError("trainer is not prepared")
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "fine_tune_type": "lora",
            "lora_parameters": {
                "dropout": self.config.lora_dropout,
                "keys": list(LORA_KEYS),
                "rank": self.config.lora_rank,
                "scale": self.config.lora_scale,
            },
            "num_layers": self.config.num_layers,
        }
        (directory / "adapter_config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        weights = dict(tree_flatten(self.model.trainable_parameters()))
        self.mx.eval(*weights.values())
        self.mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
        flat = dict(tree_flatten(self.optimizer.state))
        self.mx.eval(*flat.values())
        self.mx.save_safetensors(str(directory / "optimizer.safetensors"), flat)
