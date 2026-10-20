"""Local GGUF completion for the coding-agent loop.

The import of llama-cpp is deferred so harness tests do not require the
optional runtime. This module does not download weights.
"""
from __future__ import annotations

from pathlib import Path

from .coding_agent_loop import Generation


class GGUFGenerator:
    def __init__(self, model_path: Path, *, seed: int, n_ctx: int, n_threads: int, n_batch: int = 128) -> None:
        from llama_cpp import Llama

        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            seed=seed,
            n_batch=min(n_batch, n_ctx),
            verbose=False,
            use_mmap=True,
            use_mlock=False,
            logits_all=False,
        )

    def __call__(self, prompt: str, max_tokens: int) -> Generation:
        encoded = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        room = self.n_ctx - len(encoded) - 8
        if room < 16:
            return Generation("context full; call a tool on the next turn if one remains.", 0)
        limit = min(max_tokens, room)
        out = self.llm.create_completion(
            prompt,
            max_tokens=limit,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            repeat_penalty=1.0,
            stop=["<|im_end|>", "<|im_start|>"],
        )
        text = out["choices"][0]["text"]
        usage = out.get("usage") or {}
        tokens = int(usage.get("completion_tokens") or 0)
        if tokens <= 0 and text:
            tokens = len(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))
        return Generation(text, tokens)
