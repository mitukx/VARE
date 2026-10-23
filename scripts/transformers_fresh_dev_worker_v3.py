"""Small candidate-side probes for the frozen v6 Transformers cohort."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def watermark_repeated_ngrams() -> dict[str, Any]:
    import torch
    from types import SimpleNamespace
    from transformers import WatermarkDetector

    detector = WatermarkDetector(
        SimpleNamespace(vocab_size=128, bos_token_id=127, is_encoder_decoder=False, decoder_start_token_id=None),
        "cpu",
        {"greenlist_ratio": 0.25, "bias": 2.0, "seeding_scheme": "selfhash", "context_width": 1},
        ignore_repeated_ngrams=True,
    )
    result = detector(torch.tensor([[3] * 10], dtype=torch.long), return_dict=True)
    return {
        "num_tokens_scored": result.num_tokens_scored.tolist(),
        "num_green_tokens": result.num_green_tokens.tolist(),
    }


def rope_legacy_setter() -> dict[str, Any]:
    import warnings
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    config = LlamaConfig(rope_theta=500000.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.rope_scaling = {
            "type": "yarn",
            "factor": 4.0,
            "rope_theta": 500000.0,
            "original_max_position_embeddings": 4096,
        }
        rotary = LlamaRotaryEmbedding(config)
        normalized = dict(config.rope_parameters)
    with warnings.catch_warnings(record=True) as none_caught:
        warnings.simplefilter("always")
        config.rope_scaling = None
    return {
        "rope_type": normalized.get("rope_type"),
        "factor": normalized.get("factor"),
        "future_warning": any(issubclass(item.category, FutureWarning) for item in caught),
        "none_future_warning": any(issubclass(item.category, FutureWarning) for item in none_caught),
        "none_preserved": config.rope_parameters is None,
    }


def move_to_device_preserves_output_type() -> dict[str, Any]:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from transformers.pytorch_utils import move_to_device

    value = BaseModelOutput(
        last_hidden_state=torch.zeros(1, 2, 4),
        hidden_states=(torch.zeros(2, 4), torch.ones(2, 4)),
    )
    moved = move_to_device(value, torch.device("cpu"))
    return {
        "output_type": type(moved).__name__,
        "hidden_states_type": type(moved.hidden_states).__name__ if hasattr(moved, "hidden_states") else None,
        "last_hidden_state_shape": list(moved.last_hidden_state.shape)
        if hasattr(moved, "last_hidden_state") else None,
        "nested_values_preserved": bool(
            hasattr(moved, "hidden_states")
            and torch.equal(moved.hidden_states[0], value.hidden_states[0])
            and torch.equal(moved.hidden_states[1], value.hidden_states[1])
        ),
    }


def janus_background_color() -> dict[str, Any]:
    from transformers.models.janus.image_processing_janus import JanusImageProcessor

    default = JanusImageProcessor()
    custom = JanusImageProcessor(image_mean=(0.5, 0.25, 0.0))
    return {
        "default_mean": list(default.image_mean),
        "default_background": list(default.background_color),
        "custom_background": list(custom.background_color),
    }
