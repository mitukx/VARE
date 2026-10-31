"""Generation backends.

``MLXPolicy`` runs mlx-lm with an optional LoRA adapter. ``MockPolicy`` is a
synthetic stand-in for plumbing tests: it reads the reference fix (which a
real policy never sees), so its outputs are not model behavior and every
record it produces is labeled ``synthetic``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import time
from typing import Any, Mapping, Protocol

from .edit_format import reference_fix_blocks, render_blocks


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    finish: str  # "stop" | "length" | "prompt_too_long"


class Policy(Protocol):
    policy_id: str
    synthetic: bool

    def generate(self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float,
                 top_p: float, seed: int, task: Mapping[str, Any] | None = None) -> Generation: ...

    def close(self) -> None: ...

    def peak_memory_bytes(self) -> int | None: ...


class MLXPolicy:
    synthetic = False

    def __init__(self, model: str, adapter_path: str | None = None, *, revision: str | None = None,
                 max_prompt_tokens: int = 4096, synthetic: bool = False):
        import mlx.core as mx
        from mlx_lm import load

        self._mx = mx
        self.model, self.tokenizer = load(model, adapter_path=adapter_path, revision=revision)
        self.policy_id = f"{model}@{revision or 'local'}+{adapter_path or 'no-adapter'}"
        self.max_prompt_tokens = max_prompt_tokens
        self.synthetic = synthetic
        mx.reset_peak_memory()

    def prompt_ids(self, messages):
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    def generate(self, messages, *, max_tokens, temperature, top_p, seed, task=None):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        ids = self.prompt_ids(messages)
        if len(ids) > self.max_prompt_tokens:
            return Generation("", len(ids), 0, 0.0, "prompt_too_long")
        self._mx.random.seed(seed)
        sampler = make_sampler(temp=temperature, top_p=top_p if temperature > 0 else 0.0)
        t0 = time.time()
        text, n, finish = "", 0, "length"
        for resp in stream_generate(self.model, self.tokenizer, ids, max_tokens=max_tokens, sampler=sampler):
            text += resp.text
            n = resp.generation_tokens
            if resp.finish_reason:
                finish = "stop" if resp.finish_reason == "stop" else "length"
        return Generation(text, len(ids), n, round(time.time() - t0, 3), finish)

    def peak_memory_bytes(self):
        return int(self._mx.get_peak_memory())

    def close(self):
        del self.model
        self._mx.clear_cache()


class MockPolicy:
    """Synthetic: emits the reference fix with probability ``p_correct``."""

    synthetic = True

    def __init__(self, p_correct: float, policy_id: str = "mock"):
        self.p_correct = p_correct
        self.policy_id = f"mock:{policy_id}:p={p_correct}"

    def generate(self, messages, *, max_tokens, temperature, top_p, seed, task=None):
        assert task is not None
        h = hashlib.sha256(f"{self.policy_id}:{task['instance_id']}:{seed}:{temperature}".encode())
        rng = random.Random(int.from_bytes(h.digest()[:8], "big"))
        if rng.random() < self.p_correct:
            _, blocks = reference_fix_blocks(task["_patch"])
            text = render_blocks(blocks)
        else:
            text = "<<<<<<< SEARCH\nthis line does not exist\n=======\npass\n>>>>>>> REPLACE"
        return Generation(text, 0, len(text) // 4, 0.0, "stop")

    def peak_memory_bytes(self):
        return None

    def close(self):
        pass
