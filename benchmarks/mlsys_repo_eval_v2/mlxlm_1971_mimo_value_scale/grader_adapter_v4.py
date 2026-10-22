"""Candidate-visible bridge with independent full and cached comparisons."""

from __future__ import annotations

from typing import Any


def _models(case: dict[str, Any]):
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
    return mx, reference, scaled


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
        state.append(_jsonable(layer_cache.state))
    return {"step_logits": step_logits, "cache_state": state}


def run_full(case: dict[str, Any]) -> dict[str, Any]:
    mx, reference, scaled = _models(case)
    tokens = mx.array(case["inputs"])
    expected = reference(tokens)
    actual = scaled(tokens)
    mx.eval(expected, actual)
    return {"full_reference": expected.tolist(), "full_configured": actual.tolist()}


def run_cached(case: dict[str, Any]) -> dict[str, Any]:
    mx, reference, scaled = _models(case)
    tokens = mx.array(case["inputs"])
    return {
        "incremental_reference": _cached_trace(reference, tokens, mx),
        "incremental_configured": _cached_trace(scaled, tokens, mx),
    }


def run(case: dict[str, Any]) -> dict[str, Any]:
    """Expose a single session function while keeping result paths separate."""
    mode = case.get("mode")
    clean_case = {key: value for key, value in case.items() if key != "mode"}
    if mode == "full":
        return run_full(clean_case)
    if mode == "cached":
        return run_cached(clean_case)
    raise ValueError("unknown evaluation mode")
