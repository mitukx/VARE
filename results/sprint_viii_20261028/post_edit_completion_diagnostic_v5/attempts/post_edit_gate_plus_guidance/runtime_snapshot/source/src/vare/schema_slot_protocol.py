"""Versioned MLX action selection with schema-defined typed argument slots.

The model chooses a tool from a finite, token-constrained vocabulary. Required
argument names and JSON types come from the exposed tool schema; string slots
are emitted as raw text and integer slots as one strict JSON integer scalar.
The controller serializes those typed values into the tool envelope. It never
repairs malformed model output or invents argument values.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable


PROTOCOL_ID = "vare.schema-slots.v1"
PROTOCOL_ID_V2 = "vare.schema-slots.v2"
PROTOCOL_ID_V3 = "vare.schema-slots.v3"
PROTOCOL_ID_V4 = "vare.schema-slots.v4"
PROTOCOL_ID_V5 = "vare.schema-slots.v5"
PROTOCOL_ID_V6 = "vare.schema-slots.v6"
PROTOCOL_ID_V7 = "vare.schema-slots.v7"
PROTOCOL_ID_V8 = "vare.schema-slots.v8"
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
    if detail.get("protocol_id") in {
        PROTOCOL_ID, PROTOCOL_ID_V2, PROTOCOL_ID_V3, PROTOCOL_ID_V4,
        PROTOCOL_ID_V5, PROTOCOL_ID_V6, PROTOCOL_ID_V7, PROTOCOL_ID_V8,
    } and failure in INTERNAL_GENERATION_FAILURES:
        return str(failure)
    return None


def function_body_normalization_choices(
    raw_body: str, *, original_ends_with_newline: bool,
) -> tuple[str, ...]:
    """Constrain declared formatting choices to those compatible with raw text.

    This affects generation-time enum tokens only. It never changes the body;
    the selected enum value remains an explicit model-generated argument.
    """
    if not isinstance(raw_body, str):
        return ()
    is_single_fence = re.fullmatch(
        r"```(?:python|py)?\n(.*?)\n?```\n?", raw_body, flags=re.DOTALL,
    ) is not None
    if is_single_fence:
        if original_ends_with_newline and not raw_body.endswith("\n"):
            return ("strip_single_python_fence_append_final_newline",)
        return ("strip_single_python_fence",)
    if original_ends_with_newline and not raw_body.endswith("\n"):
        return ("append_final_newline",)
    return ("none",)


def dynamic_function_slot_choices(
    tool_name: str, argument_name: str, messages: list[dict[str, Any]],
) -> tuple[list[str], str] | None:
    """Return exact prior-tool values for function navigation and edit leases.

    These are controller-selected finite choices from already observed tool
    results. Missing or malformed provenance returns an empty choice set so
    the caller can fail closed instead of asking the model to copy a value.
    """
    source_tool: str | None = None
    if tool_name == "read_function" and argument_name in {"path", "qualified_name"}:
        source_tool = "list_functions"
    elif tool_name == "replace_function_body_transactional" and argument_name == "lease":
        source_tool = "read_function"
    else:
        return None

    source_result: dict[str, Any] | None = None
    for message in reversed(messages):
        if message.get("role") != "tool" or message.get("name") != source_tool:
            continue
        try:
            parsed = json.loads(message.get("content", ""))
        except (TypeError, json.JSONDecodeError):
            return ([], f"dynamic_choice_source_invalid:{source_tool}")
        if not isinstance(parsed, dict):
            return ([], f"dynamic_choice_source_invalid:{source_tool}")
        source_result = parsed
        break
    if source_result is None:
        return ([], f"dynamic_choice_source_missing:{source_tool}")

    if source_tool == "list_functions":
        if source_result.get("status") != "functions_listed":
            return ([], "dynamic_choice_source_not_successful:list_functions")
        if argument_name == "path":
            path = source_result.get("path")
            if isinstance(path, str) and path:
                return ([path], "prior_list_functions_path_v1")
            return ([], "dynamic_choice_value_missing:list_functions.path")
        functions = source_result.get("functions")
        if not isinstance(functions, list):
            return ([], "dynamic_choice_value_invalid:list_functions.functions")
        names: list[str] = []
        for item in functions:
            if not isinstance(item, dict):
                continue
            name = item.get("qualified_name")
            if isinstance(name, str) and name:
                names.append(name)
        names = list(dict.fromkeys(names))
        return (names, "prior_list_functions_names_v1") if names else (
            [], "dynamic_choice_value_missing:list_functions.qualified_name",
        )

    if source_result.get("status") != "read_complete":
        return ([], "dynamic_choice_source_not_successful:read_function")
    lease = source_result.get("lease")
    return ([lease], "prior_read_function_lease_v1") if isinstance(lease, str) and lease else (
        [], "dynamic_choice_value_missing:read_function.lease",
    )


def prefix_token_sequences(tokenizer: Any, prefix: str) -> tuple[int, ...]:
    """Encode and verify an exact forced prefix for constrained generation."""
    ids = tuple(int(token) for token in tokenizer.encode(prefix, add_special_tokens=False))
    if tokenizer.decode(list(ids), skip_special_tokens=False) != prefix:
        raise ValueError("forced generation prefix does not round-trip through the tokenizer")
    return ids


def allowed_prefix_tokens(generated: list[int], prefix: tuple[int, ...]) -> set[int] | None:
    """Return the next forced-prefix token, or None after the prefix completes."""
    if not prefix:
        return None
    if len(generated) >= len(prefix):
        if tuple(generated[:len(prefix)]) != prefix:
            raise ValueError("generated tokens diverged from the forced output prefix")
        return None
    if tuple(generated) != prefix[:len(generated)]:
        raise ValueError("generated tokens diverged from the forced output prefix")
    return {prefix[len(generated)]}


def make_prefix_logits_processor(
    tokenizer: Any,
    prefix: str,
    *,
    prompt_token_count: int,
    mx: Any,
) -> tuple[Callable[[Any, Any], Any], tuple[int, ...]]:
    """Constrain decoding to emit an exact declared prefix before free text."""
    sequence = prefix_token_sequences(tokenizer, prefix)
    if not sequence:
        raise ValueError("forced generation prefix must be nonempty")

    def processor(tokens: Any, logits: Any) -> Any:
        generated = [int(token) for token in tokens[prompt_token_count:].tolist()]
        allowed = allowed_prefix_tokens(generated, sequence)
        if allowed is None:
            return logits
        vocab_size = int(logits.shape[-1])
        if any(token < 0 or token >= vocab_size for token in allowed):
            raise ValueError("forced prefix token is outside the model vocabulary")
        ids = mx.array(sorted(allowed))
        token_ids = mx.arange(vocab_size)
        keep = mx.any(token_ids[:, None] == ids[None, :], axis=1)
        return mx.where(keep[None, :], logits, -float("inf"))

    return processor, sequence


def single_line_termination_state(text: str) -> str:
    """Describe whether one physical line is complete or invalid."""
    newlines = text.count("\n")
    if newlines == 0:
        return "continue"
    if newlines == 1 and text.endswith("\n"):
        return "force_eos"
    return "invalid"


def single_line_token_piece_is_safe(piece: str) -> bool:
    """Allow tokens with no newline or one newline as their final character."""
    return "\n" not in piece or (piece.count("\n") == 1 and piece.endswith("\n"))


def make_single_line_termination_logits_processor(
    tokenizer: Any,
    *,
    prompt_token_count: int,
    eos_ids: set[int],
    mx: Any,
) -> Callable[[Any, Any], Any]:
    """Permit EOS only after one newline, then force EOS as the next token."""
    if not eos_ids:
        raise ValueError("tokenizer has no EOS token for single-line body termination")
    safe_token_mask = None
    eos_token_mask = None

    def processor(tokens: Any, logits: Any) -> Any:
        nonlocal safe_token_mask, eos_token_mask
        generated = [int(token) for token in tokens[prompt_token_count:].tolist()]
        state = single_line_termination_state(
            tokenizer.decode(generated, skip_special_tokens=False),
        )
        if state == "invalid":
            raise ValueError("single-line function body generated an interior or repeated newline")
        vocab_size = int(logits.shape[-1])
        if any(token < 0 or token >= vocab_size for token in eos_ids):
            raise ValueError("termination token is outside the model vocabulary")
        if safe_token_mask is None or eos_token_mask is None:
            safe_token_mask = mx.array([
                token not in eos_ids and single_line_token_piece_is_safe(
                    tokenizer.decode([token], skip_special_tokens=False),
                )
                for token in range(vocab_size)
            ])
            eos_token_mask = mx.array([token in eos_ids for token in range(vocab_size)])
        keep = eos_token_mask if state == "force_eos" else safe_token_mask
        return mx.where(keep[None, :], logits, -float("inf"))

    return processor


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


def string_choice_token_sequences(tokenizer: Any, choices: list[str]) -> dict[str, tuple[int, ...]]:
    """Return round-tripping token sequences for a finite string enum."""
    if not choices or len(choices) != len(set(choices)):
        raise ValueError("string choices must be nonempty and unique")
    sequences: dict[str, tuple[int, ...]] = {}
    for choice in choices:
        ids = tuple(int(token) for token in tokenizer.encode(choice, add_special_tokens=False))
        if not ids or tokenizer.decode(list(ids), skip_special_tokens=False) != choice:
            raise ValueError(f"string choice {choice!r} does not round-trip through the tokenizer")
        sequences[choice] = ids
    return sequences


def make_string_choice_logits_processor(
    tokenizer: Any,
    choices: list[str],
    *,
    prompt_token_count: int,
    mx: Any,
) -> tuple[Callable[[Any, Any], Any], dict[str, tuple[int, ...]]]:
    """Constrain a string argument slot to one declared enum value."""
    sequences = string_choice_token_sequences(tokenizer, choices)
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = {int(eos)} if eos is not None else set()
    eos_ids.update(int(token) for token in (getattr(tokenizer, "eos_token_ids", None) or []))
    if not eos_ids:
        raise ValueError("tokenizer has no EOS token for enum-constrained string generation")

    def processor(tokens: Any, logits: Any) -> Any:
        generated = [int(token) for token in tokens[prompt_token_count:].tolist()]
        allowed = allowed_choice_tokens(generated, sequences, eos_ids)
        if allowed is None:
            return logits
        vocab_size = int(logits.shape[-1])
        if any(token < 0 or token >= vocab_size for token in allowed):
            raise ValueError("enum choice token is outside the model vocabulary")
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
        choices = property_schema.get("enum")
        if choices is not None and raw not in choices:
            raise SchemaSlotError("string slot was not one declared enum value", raw_value=raw)
        return raw
    if kind == "integer":
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaSlotError("integer slot was not one JSON integer", raw_value=raw) from exc
        if not isinstance(value, int) or isinstance(value, bool):
            raise SchemaSlotError("integer slot was not one JSON integer", raw_value=raw)
        return value
    if kind == "array":
        items_schema = property_schema.get("items")
        min_items = property_schema.get("minItems", 0)
        max_items = property_schema.get("maxItems")
        if (not isinstance(items_schema, dict) or items_schema.get("type") != "string"
                or not isinstance(min_items, int) or isinstance(min_items, bool) or min_items < 0
                or not isinstance(max_items, int) or isinstance(max_items, bool) or max_items < 0):
            raise SchemaSlotError("array slot requires string items and a nonnegative maxItems", raw_value=raw)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaSlotError("array slot was not one JSON array", raw_value=raw) from exc
        if (not isinstance(value, list) or not min_items <= len(value) <= max_items
                or any(not isinstance(item, str) for item in value)):
            raise SchemaSlotError("array slot was not a bounded JSON string array", raw_value=raw)
        max_length = items_schema.get("maxLength")
        if max_length is not None and any(len(item) > max_length for item in value):
            raise SchemaSlotError("array slot item exceeded maxLength", raw_value=raw)
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
