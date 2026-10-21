"""Candidate-visible model invocation bridge with no expected outputs."""

from __future__ import annotations

from typing import Any


def run(case: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models import mimo_v2_flash

    mx.random.seed(int(case["seed"]))
    config = case["config"]
    value_scale = float(case["value_scale"])
    model = mimo_v2_flash.Model(mimo_v2_flash.ModelArgs.from_dict(config))
    scaled = mimo_v2_flash.Model(
        mimo_v2_flash.ModelArgs.from_dict(
            {**config, "attention_value_scale": value_scale}
        )
    )
    weights = tree_flatten(scaled.parameters())
    model.load_weights(
        [
            (key, value * value_scale if key.endswith("v_proj.weight") else value)
            for key, value in weights
        ]
    )
    tokens = mx.array(case["inputs"])
    return {
        "unscaled_model_output": model(tokens).tolist(),
        "configured_scale_output": scaled(tokens).tolist(),
    }
