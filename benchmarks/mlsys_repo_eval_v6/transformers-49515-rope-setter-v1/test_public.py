"""Public regression for legacy RoPE assignment compatibility."""

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


def test_legacy_rope_scaling_initializes_rotary_embedding():
    config = LlamaConfig(rope_theta=500000.0)
    config.rope_scaling = {
        "type": "yarn",
        "factor": 4.0,
        "rope_theta": 500000.0,
        "original_max_position_embeddings": 4096,
    }
    assert LlamaRotaryEmbedding(config) is not None
    assert config.rope_parameters["rope_type"] == "yarn"
