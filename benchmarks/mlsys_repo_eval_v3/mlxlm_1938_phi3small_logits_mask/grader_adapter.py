"""Candidate-visible model call shim; trusted expectations stay in the parent."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType


def _namespace_package(name: str, path: Path) -> ModuleType:
    package = ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules[name] = package
    return package


def run(case: dict) -> dict:
    import mlx.core as mx

    root = Path("/subject/mlx_lm")
    _namespace_package("mlx_lm", root)
    _namespace_package("mlx_lm.models", root / "models")
    from mlx_lm.models import phi3small

    mx.random.seed(int(case["seed"]))
    args = phi3small.ModelArgs(
        model_type="phi3small",
        hidden_size=128,
        dense_attention_every_n_layers=2,
        ff_intermediate_size=256,
        gegelu_limit=1.0,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_norm_epsilon=1e-4,
        vocab_size=100352,
    )
    model = phi3small.Model(args)
    logits = model(mx.array([[1, 2, 3]]))
    dummy_logits = logits[..., model._dummy_tokenizer_ids]
    return {
        "logits_shape": list(logits.shape),
        "dummy_logits_shape": list(dummy_logits.shape),
        "all_dummy_logits_negative_infinity": bool(
            mx.all(dummy_logits == -float("inf")).item()
        ),
        "layer_count": len(model.layers),
    }
