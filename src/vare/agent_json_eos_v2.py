"""Generation-time JSON termination and chat-marker exclusion, protocol v2."""
from __future__ import annotations

from typing import Any, Callable

from .agent_json_eos_v1 import has_complete_json_object

PROTOCOL_ID = "raw_json_eos_v2"
CHAT_START_MARKER = "<|im_start|>"


def _chat_start_token_id(tokenizer: Any) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise ValueError("tokenizer cannot resolve the chat-start special token")
    token_id = convert(CHAT_START_MARKER)
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise ValueError("tokenizer returned an invalid chat-start token id")
    unknown = getattr(tokenizer, "unk_token_id", None)
    if unknown is not None and token_id == int(unknown):
        raise ValueError("chat-start marker resolves to the unknown token")
    decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    if decoded != CHAT_START_MARKER:
        raise ValueError("chat-start token does not decode to its declared marker")
    return token_id


def make_json_object_eos_v2_logits_processor(
    tokenizer: Any,
    *,
    prompt_token_count: int,
    eos_ids: set[int],
    mx: Any,
) -> tuple[Callable[[Any, Any], Any], dict[str, int | bool]]:
    """Block generated chat-start markers and stop at a complete JSON object.

    Marker handling occurs only in logits selection. Generated tokens and raw
    text are never stripped or rewritten; the normal strict parser remains the
    authority for accepting a tool call.
    """
    if prompt_token_count < 0:
        raise ValueError("prompt token count must be nonnegative")
    if not eos_ids:
        raise ValueError("tokenizer has no EOS token for JSON-boundary termination")
    marker_id = _chat_start_token_id(tokenizer)
    state: dict[str, int | bool] = {
        "calls": 0,
        "completion_seen": False,
        "chat_start_marker_blocked": False,
    }

    def processor(tokens: Any, logits: Any) -> Any:
        generated = [int(token) for token in tokens[prompt_token_count:].tolist()]
        raw = tokenizer.decode(generated, skip_special_tokens=False)
        state["calls"] = int(state["calls"]) + 1
        vocab_size = int(logits.shape[-1])
        if marker_id < 0 or marker_id >= vocab_size:
            raise ValueError("chat-start token is outside the model vocabulary")
        if any(token < 0 or token >= vocab_size for token in eos_ids):
            raise ValueError("EOS token is outside the model vocabulary")
        token_ids = mx.arange(vocab_size)
        if has_complete_json_object(raw):
            state["completion_seen"] = True
            eos_array = mx.array(sorted(eos_ids))
            is_eos = mx.any(token_ids[:, None] == eos_array[None, :], axis=1)
            return mx.where(is_eos[None, :], logits, -float("inf"))
        state["chat_start_marker_blocked"] = True
        not_chat_marker = token_ids != marker_id
        return mx.where(not_chat_marker[None, :], logits, -float("inf"))

    return processor, state
