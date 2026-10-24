"""Candidate-visible v2 bridge; the trusted grader owns all golden outputs."""

from __future__ import annotations

from typing import Any


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"unsupported cache state value: {type(value).__name__}")


def _cached_trace(model, tokens, mx):
    cache = model.make_cache()
    step_logits = []
    for index in range(tokens.shape[1]):
        logits = model(tokens[:, index : index + 1], cache=cache)
        mx.eval(logits)
        step_logits.append(logits.tolist())
    state = []
    for layer_cache in cache:
        values = layer_cache.state
        state.append(_jsonable(values))
    return {"step_logits": step_logits, "cache_state": state}


def run(case: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models import mimo_v2_flash

    mx.random.seed(int(case["seed"]))
    config = case["config"]
    scale = float(case["value_scale"])
    scaled = mimo_v2_flash.Model(
        mimo_v2_flash.ModelArgs.from_dict(
            {**config, "attention_value_scale": scale}
        )
    )
    reference = mimo_v2_flash.Model(mimo_v2_flash.ModelArgs.from_dict(config))
    weights = tree_flatten(scaled.parameters())
    reference.load_weights(
        [
            (key, value * scale if key.endswith("v_proj.weight") else value)
            for key, value in weights
        ]
    )
    tokens = mx.array(case["inputs"])
    full_reference = reference(tokens)
    full_configured = scaled(tokens)
    mx.eval(full_reference, full_configured)
    return {
        "full_reference": full_reference.tolist(),
        "full_configured": full_configured.tolist(),
        "incremental_reference": _cached_trace(reference, tokens, mx),
        "incremental_configured": _cached_trace(scaled, tokens, mx),
    }
