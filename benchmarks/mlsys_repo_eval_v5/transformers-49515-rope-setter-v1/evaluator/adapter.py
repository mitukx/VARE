from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


def grade():
    config = LlamaConfig(
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
    )
    config.rope_scaling = {"type": "linear", "factor": 2.0}
    LlamaRotaryEmbedding(config)
    return config.rope_parameters.get("rope_type") == "linear"
