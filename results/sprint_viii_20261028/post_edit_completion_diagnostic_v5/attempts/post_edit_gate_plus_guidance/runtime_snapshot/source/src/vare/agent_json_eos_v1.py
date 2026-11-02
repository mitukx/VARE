"""Generation-time termination for one complete JSON tool-call object."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

PROTOCOL_ID = "raw_json_eos_v1"

_ALLOWED_JSON_SUFFIX = re.compile(
    r"(?:\s|```|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>)*"
)


def has_complete_json_object(raw: str) -> bool:
    """Return true only when one JSON object is complete at a parser-safe boundary."""
    candidate = raw.strip()
    candidate = re.sub(r"^<\|im_start\|>\s*(?:assistant)?\s*", "", candidate)
    if candidate.startswith("```json"):
        candidate = re.sub(r"^```json\s*", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.lstrip()
    if not candidate:
        return False
    try:
        value, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return False
    if not isinstance(value, dict):
        return False
    return _ALLOWED_JSON_SUFFIX.fullmatch(candidate[end:]) is not None


def make_json_object_eos_logits_processor(
    tokenizer: Any,
    *,
    prompt_token_count: int,
    eos_ids: set[int],
    mx: Any,
) -> tuple[Callable[[Any, Any], Any], dict[str, int | bool]]:
    """Force a declared EOS immediately after a complete JSON object is generated.

    The processor never removes or rewrites generated text. Before the JSON
    object is complete it leaves model logits unchanged; once complete it
    permits only EOS tokens, leaving strict parsing and schema validation to
    the normal tool-call parser.
    """
    if prompt_token_count < 0:
        raise ValueError("prompt token count must be nonnegative")
    if not eos_ids:
        raise ValueError("tokenizer has no EOS token for JSON-boundary termination")
    state: dict[str, int | bool] = {"calls": 0, "completion_seen": False}

    def processor(tokens: Any, logits: Any) -> Any:
        generated = [int(token) for token in tokens[prompt_token_count:].tolist()]
        raw = tokenizer.decode(generated, skip_special_tokens=False)
        state["calls"] = int(state["calls"]) + 1
        if not has_complete_json_object(raw):
            return logits
        state["completion_seen"] = True
        vocab_size = int(logits.shape[-1])
        if any(token < 0 or token >= vocab_size for token in eos_ids):
            raise ValueError("EOS token is outside the model vocabulary")
        eos_array = mx.array(sorted(eos_ids))
        token_ids = mx.arange(vocab_size)
        is_eos = mx.any(token_ids[:, None] == eos_array[None, :], axis=1)
        return mx.where(is_eos[None, :], logits, -float("inf"))

    return processor, state
