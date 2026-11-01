"""Exact same-prompt prefix reuse for independent MLX completions.

The caller retains one cache containing only the shared prompt prefix. Each
sample receives a deep copy before its final prompt token is processed, so
generated tokens never flow into a sibling completion's cache.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any, Mapping, Sequence


def _messages_key(messages: Sequence[Mapping[str, str]]) -> str:
    data = json.dumps(list(messages), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _tokenize(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    return [int(token) for token in tokenizer.apply_chat_template(
        list(messages), add_generation_prompt=True, tokenize=True,
    )]


def _cache_bytes(cache: Sequence[Any]) -> int:
    return sum(int(item.nbytes) for item in cache)


def _cache_cursor_signature(cache: Sequence[Any]) -> list[dict[str, Any]]:
    """Return ownership-sensitive cursors without copying cache tensors."""
    rows = []
    for item in cache:
        row = {"type": type(item).__name__, "nbytes": int(item.nbytes)}
        for name in ("offset", "_idx", "_offset"):
            value = getattr(item, name, None)
            if isinstance(value, (int, float, str, bool)) or value is None:
                row[name] = value
            elif hasattr(value, "tolist"):
                row[name] = value.tolist()
        try:
            row["size"] = item.size()
        except (AttributeError, TypeError, ValueError):
            pass
        rows.append(row)
    return rows


def build_shared_prefix(model: Any, prompt_ids: Sequence[int]) -> dict[str, Any]:
    """Prefill all but the final prompt token once for a completion group."""
    if len(prompt_ids) < 2:
        raise ValueError("prefix reuse requires at least two prompt tokens")
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    inputs = mx.array(list(prompt_ids[:-1]), dtype=mx.int32)[None]
    started = time.perf_counter()
    logits = model(inputs, cache=cache)
    mx.eval(logits, [item.state for item in cache])
    seconds = time.perf_counter() - started
    mx.clear_cache()
    return {
        "prompt_ids": list(prompt_ids),
        "cache": cache,
        "prefill_seconds": seconds,
        "cache_bytes": _cache_bytes(cache),
        "cursor_signature": _cache_cursor_signature(cache),
    }


def generate_profiled(
    model: Any,
    tokenizer: Any,
    *,
    messages: Sequence[Mapping[str, str]],
    seed: int,
    temperature: float,
    max_tokens: int,
    prompt_ids: Sequence[int] | None = None,
    shared_prefix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one completion and retain tokenization, first-token, and decode timing."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    tokenization_started = time.perf_counter()
    if prompt_ids is None:
        prompt_ids = _tokenize(tokenizer, messages)
        tokenization_seconds = time.perf_counter() - tokenization_started
    else:
        prompt_ids = [int(token) for token in prompt_ids]
        tokenization_seconds = 0.0
    if not prompt_ids:
        raise ValueError("empty prompt")
    mx.random.seed(seed)
    sampler = make_sampler(temp=temperature, top_p=1.0 if temperature > 0 else 0.0,
                           min_p=0.0, top_k=0)
    prompt_cache = None
    clone_seconds = 0.0
    stream_prompt: Sequence[int] = prompt_ids
    if shared_prefix is not None:
        if list(prompt_ids) != list(shared_prefix["prompt_ids"]):
            raise ValueError("shared prefix does not match the exact tokenized prompt")
        clone_started = time.perf_counter()
        prompt_cache = copy.deepcopy(shared_prefix["cache"])
        clone_seconds = time.perf_counter() - clone_started
        stream_prompt = [int(prompt_ids[-1])]

    prompt_progress: dict[str, float] = {}

    def record_progress(processed: int, total: int) -> None:
        now = time.perf_counter()
        if "started" not in prompt_progress:
            prompt_progress["started"] = now
        if processed == total:
            prompt_progress["complete"] = now

    generation_started = time.perf_counter()
    token_ids: list[int] = []
    sampler_logprobs: list[float] = []
    text = ""
    finish = "length"
    first_yield_seconds = None
    last_item = None
    kwargs = {
        "max_tokens": max_tokens,
        "sampler": sampler,
        "prompt_progress_callback": record_progress,
    }
    if prompt_cache is not None:
        kwargs["prompt_cache"] = prompt_cache
    for item in stream_generate(model, tokenizer, list(stream_prompt), **kwargs):
        if first_yield_seconds is None:
            first_yield_seconds = time.perf_counter() - generation_started
        last_item = item
        text += item.text
        token_ids.append(int(item.token))
        if getattr(item, "logprobs", None) is not None:
            sampler_logprobs.append(float(item.logprobs[item.token]))
        if item.finish_reason:
            finish = str(item.finish_reason)
    generation_seconds = time.perf_counter() - generation_started
    prefix_seconds = 0.0
    if shared_prefix is not None:
        prefix_seconds = float(shared_prefix["prefill_seconds"])
        if _cache_cursor_signature(shared_prefix["cache"]) != shared_prefix["cursor_signature"]:
            raise RuntimeError("shared prompt cache was mutated by a completion")
    first_token_sample_seconds = None
    if "started" in prompt_progress and "complete" in prompt_progress:
        first_token_sample_seconds = prompt_progress["complete"] - prompt_progress["started"]
    return {
        "prompt_ids": list(prompt_ids),
        "completion_ids": token_ids,
        "text": text,
        "finish": finish,
        "sampler_logprobs": sampler_logprobs,
        "seconds": generation_seconds + clone_seconds,
        "generation_seconds": generation_seconds,
        "cache_clone_seconds": clone_seconds,
        "tokenization_seconds": tokenization_seconds,
        "shared_prefix_seconds": prefix_seconds,
        "shared_prefix_cache_bytes": int(shared_prefix["cache_bytes"]) if shared_prefix else 0,
        "prompt_tokens": len(prompt_ids),
        "prompt_prefill_and_first_sample_seconds": first_token_sample_seconds,
        "first_yield_seconds": first_yield_seconds,
        "generation_tokens_per_second": getattr(last_item, "generation_tps", None),
        "framework_prompt_tokens_per_second": getattr(last_item, "prompt_tps", None),
        "peak_memory_gib": getattr(last_item, "peak_memory", None),
        "finish_reason": finish,
    }


class CachedGroupGenerator:
    """Drop-in replacement for the runner generator within one same-prompt group."""

    def __init__(self, *, max_prompt_tokens: int) -> None:
        self.max_prompt_tokens = max_prompt_tokens
        self._key: str | None = None
        self._model_id: int | None = None
        self._tokenizer_id: int | None = None
        self._prompt_ids: list[int] | None = None
        self._messages: list[Mapping[str, str]] | None = None
        self._prefix: dict[str, Any] | None = None
        self._setup_tokenization_seconds = 0.0
        self.records: list[dict[str, Any]] = []

    def __call__(self, model: Any, tokenizer: Any, messages: Sequence[Mapping[str, str]], *,
                 seed: int, temperature: float, max_tokens: int) -> dict[str, Any]:
        key = _messages_key(messages)
        setup_attributed = 0.0
        if self._key is None:
            started = time.perf_counter()
            self._prompt_ids = _tokenize(tokenizer, messages)
            self._setup_tokenization_seconds = time.perf_counter() - started
            self._key = key
            self._model_id = id(model)
            self._tokenizer_id = id(tokenizer)
            self._messages = list(messages)
            if len(self._prompt_ids) > self.max_prompt_tokens:
                result: dict[str, Any] = {
                    "prompt_ids": self._prompt_ids,
                    "completion_ids": [],
                    "text": "",
                    "finish": "prompt_too_long",
                    "sampler_logprobs": [],
                    "seconds": 0.0,
                    "generation_seconds": 0.0,
                    "tokenization_seconds": self._setup_tokenization_seconds,
                    "shared_prefix_seconds": 0.0,
                    "shared_prefix_cache_bytes": 0,
                    "prompt_tokens": len(self._prompt_ids),
                }
                self.records.append({"seed": seed, "profile": result})
                return result
            prefix_started = time.perf_counter()
            self._prefix = build_shared_prefix(model, self._prompt_ids)
            self._prefix["prefill_seconds"] = time.perf_counter() - prefix_started
            setup_attributed = self._setup_tokenization_seconds + self._prefix["prefill_seconds"]
        elif key != self._key or id(model) != self._model_id or id(tokenizer) != self._tokenizer_id:
            raise RuntimeError("one CachedGroupGenerator may serve only one model, tokenizer, and prompt")

        if self._prefix is None or self._prompt_ids is None:
            raise RuntimeError("shared prefix was not initialized")
        result = generate_profiled(
            model, tokenizer, messages=self._messages or messages,
            seed=seed, temperature=temperature, max_tokens=max_tokens,
            prompt_ids=self._prompt_ids, shared_prefix=self._prefix,
        )
        if setup_attributed:
            result["seconds"] += setup_attributed
            result["tokenization_seconds"] = self._setup_tokenization_seconds
        self.records.append({
            "seed": seed,
            "prompt_tokens": len(self._prompt_ids),
            "completion_tokens": len(result["completion_ids"]),
            "profile": {key: value for key, value in result.items()
                        if key not in {"prompt_ids", "completion_ids", "text", "sampler_logprobs"}},
        })
        return result
