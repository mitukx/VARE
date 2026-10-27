#!/usr/bin/env python3
"""Run the existing VARE agent through a plain-chat tool-template adapter."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mlx_lm import generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from vare.flat_tool_chat import flatten_tool_messages  # noqa: E402
import run_local_agent as agent  # noqa: E402


def generate_turn(
    backend: str, model: Any, tokenizer: Any, messages: list[dict[str, Any]],
    tools: list[dict[str, Any]], *, limit: int, args: Any,
) -> tuple[str, int, int]:
    if backend != "mlx":
        raise ValueError("SmolLM tool-template adapter supports only MLX")
    rendered = flatten_tool_messages(messages, tools)
    prompt = tokenizer.apply_chat_template(
        rendered, tokenize=False, add_generation_prompt=True,
    )
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_tokens >= args.max_context_tokens:
        raise ValueError("context_budget_exhausted")
    limit = min(limit, args.max_context_tokens - prompt_tokens)
    sampler = make_sampler(
        temp=0.0 if not args.do_sample else args.temperature,
        top_p=0.0 if not args.do_sample else args.top_p,
        top_k=0 if not args.do_sample else args.top_k,
    )
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=limit,
                   sampler=sampler, verbose=False)
    return raw, len(tokenizer.encode(raw, add_special_tokens=False)), prompt_tokens


agent.generate_turn = generate_turn

if __name__ == "__main__":
    raise SystemExit(agent.main())
