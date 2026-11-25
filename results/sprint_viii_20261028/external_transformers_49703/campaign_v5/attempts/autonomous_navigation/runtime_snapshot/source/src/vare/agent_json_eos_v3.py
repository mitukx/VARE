"""Raw JSON tool generation with marker and premature-EOS constraints, v3."""
from __future__ import annotations

from typing import Any, Callable

from .agent_json_eos_v1 import has_complete_json_object
from .agent_json_eos_v2 import _chat_start_token_id

PROTOCOL_ID = "raw_json_eos_v3"


def make_json_object_eos_v3_logits_processor(
    tokenizer: Any,
    *,
    prompt_token_count: int,
    eos_ids: set[int],
    mx: Any,
) -> tuple[Callable[[Any, Any], Any], dict[str, int | bool]]:
    """Block chat markers/EOS until one complete JSON object has been emitted."""
    if prompt_token_count < 0:
        raise ValueError("prompt token count must be nonnegative")
    if not eos_ids:
        raise ValueError("tokenizer has no EOS token for JSON-boundary termination")
    marker_id = _chat_start_token_id(tokenizer)
    state: dict[str, int | bool] = {
        "calls": 0,
        "completion_seen": False,
        "chat_start_marker_blocked": False,
        "premature_eos_blocked": False,
    }

    def processor(tokens: Any, logits: Any) -> Any:
        generated = [int(token) for token in tokens[prompt_token_count:].tolist()]
        raw = tokenizer.decode(generated, skip_special_tokens=False)
        state["calls"] = int(state["calls"]) + 1
        vocab_size = int(logits.shape[-1])
        blocked = {marker_id, *eos_ids}
        if any(token < 0 or token >= vocab_size for token in blocked):
            raise ValueError("control token is outside the model vocabulary")
        token_ids = mx.arange(vocab_size)
        if has_complete_json_object(raw):
            state["completion_seen"] = True
            eos_array = mx.array(sorted(eos_ids))
            is_eos = mx.any(token_ids[:, None] == eos_array[None, :], axis=1)
            return mx.where(is_eos[None, :], logits, -float("inf"))
        state["chat_start_marker_blocked"] = True
        state["premature_eos_blocked"] = True
        allowed = token_ids != marker_id
        for eos_id in eos_ids:
            allowed = allowed & (token_ids != eos_id)
        return mx.where(allowed[None, :], logits, -float("inf"))

    return processor, state
