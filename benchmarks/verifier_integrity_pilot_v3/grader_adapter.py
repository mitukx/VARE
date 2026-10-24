"""Candidate-visible workload bridge; expected results remain parent-owned."""

from __future__ import annotations

from typing import Any


_CONFIG = {
    "model_type": "mimo_v2_flash",
    "num_experts_per_tok": 2,
    "hybrid_layer_pattern": [0, 1],
    "moe_layer_freq": [0, 0],
    "add_swa_attention_sink_bias": True,
    "add_full_attention_sink_bias": False,
    "sliding_window_size": 4,
    "vocab_size": 32,
    "hidden_size": 32,
    "intermediate_size": 64,
    "moe_intermediate_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "n_shared_experts": None,
    "n_routed_experts": 4,
    "routed_scaling_factor": None,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    "max_position_embeddings": 128,
    "layernorm_epsilon": 1e-5,
    "rope_theta": 1000.0,
    "swa_rope_theta": 1000.0,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "head_dim": 16,
    "v_head_dim": 8,
    "swa_head_dim": 16,
    "swa_v_head_dim": 8,
    "partial_rotary_factor": 0.5,
}


def run(case: dict[str, Any]) -> dict[str, Any]:
    """Return observed logits; the grader compares them to a hidden reference."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models import mimo_v2_flash

    mx.random.seed(0)
    scale = float(case["value_scale"])
    model = mimo_v2_flash.Model(mimo_v2_flash.ModelArgs.from_dict(_CONFIG))
    scaled = mimo_v2_flash.Model(
        mimo_v2_flash.ModelArgs.from_dict(
            {**_CONFIG, "attention_value_scale": scale}
        )
    )
    weights = tree_flatten(scaled.parameters())
    model.load_weights(
        [
            (key, value * scale if key.endswith("v_proj.weight") else value)
            for key, value in weights
        ]
    )
    inputs = mx.array([[0, 1, 2, 3, 4, 5]])
    return {
        "unscaled_model_logits": model(inputs).tolist(),
        "scaled_model_logits": scaled(inputs).tolist(),
    }
