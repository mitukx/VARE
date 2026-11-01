"""Versioned MLX action selection with schema-defined typed argument slots.

The model chooses a tool from a finite, token-constrained vocabulary. Required
argument names and JSON types come from the exposed tool schema; string slots
are emitted as raw text and integer slots as one strict JSON integer scalar.
The controller serializes those typed values into the tool envelope. It never
repairs malformed model output or invents argument values.
"""

from __future__ import annotations

import json
from typing import Any, Callable


PROTOCOL_ID = "vare.schema-slots.v1"
PROTOCOL_ID_V2 = "vare.schema-slots.v2"
INTERNAL_GENERATION_FAILURES = frozenset({
    "per_turn_token_budget_exhausted",
    "argument_slot_token_budget_exhausted",
    "schema_slot_validation_failed",
})


class SchemaSlotError(ValueError):
    def __init__(self, message: str, *, raw_value: str | None = None):
        super().__init__(message)
        self.raw_value = raw_value


def generation_protocol_failure(detail: dict[str, Any] | None) -> str | None:
    """Return an internal schema-generation failure without parsing its sentinel."""
    if not isinstance(detail, dict):
        return None
    failure = detail.get("stop_reason")
    if detail.get("protocol_id") in {PROTOCOL_ID, PROTOCOL_ID_V2} and failure in INTERNAL_GENERATION_FAILURES:
        return str(failure)
    return None


def tool_choice_token_sequences(tokenizer: Any, tools: list[dict[str, Any]]) -> dict[str, tuple[int, ...]]:
    """Return exact tokenizer sequences for the tools exposed in this turn."""
    names = [tool["function"]["name"] for tool in tools]
    if not names or len(names) != len(set(names)):
        raise ValueError("tool names must be nonempty and unique")
    sequences: dict[str, tuple[int, ...]] = {}
    for name in names:
        ids = tuple(int(token) for token in tokenizer.encode(name, add_special_tokens=False))
        if not ids or tokenizer.decode(list(ids), skip_special_tokens=False) != name:
            raise ValueError(f"tool name {name!r} does not round-trip through the tokenizer")
        sequences[name] = ids
    return sequences


def allowed_choice_tokens(
    prefix: list[int], sequences: dict[str, tuple[int, ...]], eos_ids: set[int],
) -> set[int] | None:
    """Compute the exact next token IDs permitted by a finite-choice trie."""
    prefix_tuple = tuple(prefix)
    # mlx_lm.generate_step eagerly computes logits one token past the yielded
    # token. Once stream_generate sees EOS it stops and never exposes that
    # look-ahead sample, so leave its logits untouched.
    if any(prefix_tuple[:-1] == sequence and prefix_tuple[-1] in eos_ids
           for sequence in sequences.values()):
        return None
    next_ids = {
        sequence[len(prefix)]
        for sequence in sequences.values()
        if len(sequence) > len(prefix) and sequence[: len(prefix)] == prefix_tuple
    }
    complete = any(sequence == prefix_tuple for sequence in sequences.values())
    if complete:
        next_ids.update(eos_ids)
    if not next_ids:
        raise ValueError(
            "generated tool-choice prefix is outside the allowed vocabulary "
            f"(length={len(prefix)}, tail={prefix[-12:]!r})"
        )
    return next_ids


def make_choice_logits_processor(
    tokenizer: Any,
    tools: list[dict[str, Any]],
    *,
    prompt_token_count: int,
    mx: Any,
) -> tuple[Callable[[Any, Any], Any], dict[str, tuple[int, ...]]]:
    """Create an MLX logits processor that masks every non-schema tool token."""
    sequences = tool_choice_token_sequences(tokenizer, tools)
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = {int(eos)} if eos is not None else set()
    eos_ids.update(int(token) for token in (getattr(tokenizer, "eos_token_ids", None) or []))
    if not eos_ids:
        raise ValueError("tokenizer has no EOS token for constrained tool selection")

    def processor(tokens: Any, logits: Any) -> Any:
        generated = [int(token) for token in tokens[prompt_token_count:].tolist()]
        allowed = allowed_choice_tokens(generated, sequences, eos_ids)
        if allowed is None:
            return logits
        vocab_size = int(logits.shape[-1])
        if any(token < 0 or token >= vocab_size for token in allowed):
            raise ValueError("tool-choice token is outside the model vocabulary")
        ids = mx.array(sorted(allowed))
        token_ids = mx.arange(vocab_size)
        keep = mx.any(token_ids[:, None] == ids[None, :], axis=1)
        return mx.where(keep[None, :], logits, -float("inf"))

    return processor, sequences


def terminal_newline_is_present(
    generated_token_ids: list[int], tokenizer: Any,
) -> bool:
    """Check whether a generated raw string currently ends in a real newline."""
    if not generated_token_ids:
        return False
    return tokenizer.decode(generated_token_ids, skip_special_tokens=False).endswith("\n")


def make_terminal_newline_logits_processor(
    tokenizer: Any,
    *,
    prompt_token_count: int,
    eos_ids: set[int],
    mx: Any,
) -> Callable[[Any, Any], Any]:
    """Disallow EOS until the model-generated raw string ends with a newline.

    This is a prospective output constraint, not a post-generation edit. The
    string remains the exact decoded model output; when the source lease
    requires a final newline, the slot cannot terminate before it has one.
    """
    if not eos_ids:
        raise ValueError("tokenizer has no EOS token for terminal-newline constraint")

    def processor(tokens: Any, logits: Any) -> Any:
        generated = [int(token) for token in tokens[prompt_token_count:].tolist()]
        if terminal_newline_is_present(generated, tokenizer):
            return logits
        vocab_size = int(logits.shape[-1])
        if any(token < 0 or token >= vocab_size for token in eos_ids):
            raise ValueError("EOS token is outside the model vocabulary")
        eos_array = mx.array(sorted(eos_ids))
        token_ids = mx.arange(vocab_size)
        is_eos = mx.any(token_ids[:, None] == eos_array[None, :], axis=1)
        return mx.where(is_eos[None, :], -float("inf"), logits)

    return processor


def parse_slot_value(raw: str, property_schema: dict[str, Any]) -> Any:
    """Interpret only the value form declared by a schema slot."""
    kind = property_schema.get("type")
    if kind == "string":
        return raw
    if kind == "integer":
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaSlotError("integer slot was not one JSON integer", raw_value=raw) from exc
        if not isinstance(value, int) or isinstance(value, bool):
            raise SchemaSlotError("integer slot was not one JSON integer", raw_value=raw)
        return value
    raise SchemaSlotError(f"unsupported schema-slot type: {kind!r}", raw_value=raw)


def serialize_slot_action(
    tool_name: str,
    raw_values: dict[str, str],
    tools: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Build the canonical tool envelope from complete, strictly typed slots."""
    tool = next((item for item in tools if item["function"]["name"] == tool_name), None)
    if tool is None:
        raise SchemaSlotError("selected tool is not exposed by this protocol")
    schema = tool["function"]["parameters"]
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if schema.get("additionalProperties") is not False:
        raise SchemaSlotError("schema-slot protocol requires additionalProperties=false")
    if set(required) != set(properties):
        raise SchemaSlotError("schema-slot protocol requires every declared property")
    if set(raw_values) != set(required):
        raise SchemaSlotError("generated slots do not exactly match the required properties")

    arguments = {
        key: parse_slot_value(raw_values[key], properties[key])
        for key in required
    }
    payload = {"name": tool_name, "arguments": arguments}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), payload
