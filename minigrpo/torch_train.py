"""PyTorch LoRA GRPO step for a Colab GPU. Imported only when a run starts.

The rank, scale, and last-layer q/v targets match the MLX run. The two
backends are not numerical replicates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minigrpo.grpo import LOG_RATIO_LIMIT
from minigrpo.loop import Rollout, RunConfig


def _load_causal(repo: str, revision: str, dtype: Any) -> Any:
    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    kwargs: dict[str, object] = {"revision": revision} if revision else {}
    try:
        return AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, **kwargs)


class TorchTrainer:
    def __init__(self, config: RunConfig) -> None:
        import torch  # type: ignore[import-not-found]

        self.config = config
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
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
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        kwargs: dict[str, object] = {"revision": self.config.revision} if self.config.revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, **kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not getattr(self.tokenizer, "chat_template", None):
            raise RuntimeError("model tokenizer has no chat template")
        self.model = _load_causal(self.config.model, self.config.revision, self.dtype).to(self.device)
        self.optimizer = None

    def _layer_indexes(self) -> list[int]:
        count = int(self.model.config.num_hidden_layers)
        if not 1 <= self.config.num_layers <= count:
            raise RuntimeError("num_layers is outside the model")
        return list(range(count - self.config.num_layers, count))

    def _init_fresh(self) -> None:
        from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]

        self.torch.manual_seed(self.config.seed)
        spec = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_scale,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            layers_to_transform=self._layer_indexes(),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, spec)
        self._new_optimizer()

    def _load_reference(self) -> None:
        reference = _load_causal(self.config.model, self.config.revision, self.dtype).to(self.device)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        self.ref_model = reference

    def _new_optimizer(self) -> None:
        params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = self.torch.optim.Adam(params, lr=self.config.learning_rate)

    def _load_adapter(self, directory: Path, *, load_optimizer: bool) -> None:
        from peft import PeftModel  # type: ignore[import-not-found]

        self.model = PeftModel.from_pretrained(self.model, str(directory)).to(self.device)
        self._new_optimizer()
        optimizer_path = directory / "optimizer.pt"
        if load_optimizer:
            if not optimizer_path.exists():
                raise RuntimeError(f"missing optimizer state in {directory}")
            state = self.torch.load(optimizer_path, map_location=self.device)
            self.optimizer.load_state_dict(state)

    def _encode_pair(self, prompt: str, completion: str) -> tuple[tuple[int, ...], int]:
        full = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
            add_generation_prompt=False,
            return_dict=True,
        )
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_dict=True,
        )
        tokens = [int(token) for token in full["input_ids"]]
        offset = len(list(prompt_ids["input_ids"]))
        if offset < 1 or offset >= len(tokens) or len(tokens) > self.config.max_seq_length:
            raise RuntimeError("prompt mask leaves no completion tokens inside the length cap")
        return tuple(tokens), offset

    def _token_logp(self, model: Any, token_ids: tuple[int, ...], prompt_length: int, *, entropy: bool) -> tuple[Any, Any, Any]:
        ids = self.torch.tensor([token_ids], dtype=self.torch.long, device=self.device)
        logits = model(input_ids=ids).logits[:, :-1, :]
        targets = ids[:, 1:]
        log_probs = self.torch.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        positions = self.torch.arange(targets.shape[1], device=self.device)
        mask = (positions >= (prompt_length - 1)).to(token_logp.dtype)
        if not entropy:
            return token_logp, mask, None
        probs = self.torch.softmax(logits, dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1)[0]
        return token_logp, mask, token_entropy

    def _tuple(self, array: Any) -> tuple[float, ...]:
        return tuple(float(value) for value in array.detach().float().cpu().tolist())

    def generate(self, prompt: str, seed: int) -> str:
        if self.model is None:
            raise RuntimeError("trainer is not prepared")
        self.model.eval()
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention = encoded.get("attention_mask")
        if attention is not None:
            attention = attention.to(self.device)
        self.torch.manual_seed(seed)
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(seed)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
        if not str(text).strip():
            raise RuntimeError("empty completion; refusing to drop a group member")
        return str(text)

    def encode(self, prompt: str, completion: str) -> Rollout:
        if self.model is None:
            raise RuntimeError("trainer is not prepared")
        tokens, offset = self._encode_pair(prompt, completion)
        self.model.eval()
        with self.torch.inference_mode():
            logp, mask, entropy = self._token_logp(self.model, tokens, offset, entropy=True)
            ref = None
            if self.ref_model is not None:
                ref_logp, _mask, _entropy = self._token_logp(self.ref_model, tokens, offset, entropy=False)
                ref = self._tuple(ref_logp)
        return Rollout(
            text=completion,
            old_logp=self._tuple(logp),
            mask=tuple(int(bit) for bit in self._tuple(mask)),
            entropy=self._tuple(entropy),
            ref_logp=ref,
            token_ids=tokens,
            prompt_length=offset,
        )

    def update(self, rollouts: list[Rollout], advantages: list[float], *, epochs: int) -> float:
        if self.optimizer is None:
            raise RuntimeError("trainer is not prepared")
        last = 0.0
        for _epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            def loss_fn() -> Any:
                total = self.torch.zeros((), device=self.device, dtype=self.torch.float32)
                for rollout, advantage in zip(rollouts, advantages, strict=True):
                    logp, mask, _entropy = self._token_logp(
                        self.model, rollout.token_ids, rollout.prompt_length, entropy=False
                    )
                    old = self.torch.tensor(rollout.old_logp, device=self.device)
                    ratio = self.torch.exp((logp.float() - old).clamp(-LOG_RATIO_LIMIT, LOG_RATIO_LIMIT))
                    adv = self.torch.tensor(advantage, device=self.device)
                    unclipped = ratio * adv
                    clipped = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv
                    per_token = -self.torch.minimum(unclipped, clipped)
                    if rollout.ref_logp is not None and self.config.kl_coef > 0.0:
                        ref = self.torch.tensor(rollout.ref_logp, device=self.device)
                        kl_delta = (ref - logp.float()).clamp(-LOG_RATIO_LIMIT, LOG_RATIO_LIMIT)
                        per_token = per_token + self.config.kl_coef * (self.torch.exp(kl_delta) - kl_delta - 1.0)
                    active = per_token * mask.float()
                    if self.config.dr_grpo:
                        total = total + active.sum()
                    else:
                        total = total + active.sum() / self.torch.clamp(mask.float().sum(), min=1.0)
                return total / len(rollouts)

            loss = loss_fn()
            loss.backward()
            self.optimizer.step()
            last = float(loss.detach().float().cpu())
        return last

    def save(self, directory: Path) -> None:
        if self.optimizer is None:
            raise RuntimeError("trainer is not prepared")
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.torch.save(self.optimizer.state_dict(), directory / "optimizer.pt")
        note = {"backend": "torch", "num_layers": self.config.num_layers, "rank": self.config.lora_rank}
        (directory / "trainer.json").write_text(json.dumps(note, sort_keys=True) + "\n", encoding="utf-8")
