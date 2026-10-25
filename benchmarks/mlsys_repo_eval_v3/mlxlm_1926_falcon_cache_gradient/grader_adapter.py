"""Candidate-visible model call shim; no expected values or tests."""

from __future__ import annotations

import importlib.util
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
    import mlx.nn as nn

    root = Path("/subject/mlx_lm")
    _namespace_package("mlx_lm", root)
    _namespace_package("mlx_lm.models", root / "models")
    target = root / "models/falcon_h1.py"
    spec = importlib.util.spec_from_file_location(
        "mlx_lm.models.falcon_h1", target
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate Falcon-H1 module is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    mx.random.seed(int(case["seed"]))
    args = module.ModelArgs(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        vocab_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        mamba_d_ssm=32,
        mamba_n_heads=2,
        mamba_d_head=16,
        mamba_d_state=8,
    )
    model = module.Model(args)
    inputs = mx.array([[0, 1, 2], [2, 1, 0]])
    targets = mx.array([[1, 2, 3], [1, 0, 3]])
    logits = model(inputs)
    cached_logits = model(inputs, cache=model.make_cache())

    def loss_fn(candidate_model):
        return nn.losses.cross_entropy(candidate_model(inputs), targets).mean()

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    layers = []
    for layer_grads in grads["model"]["layers"]:
        grad = layer_grads["feed_forward"]["down_proj"]["weight"]
        layers.append(
            {
                "finite": bool(mx.all(mx.isfinite(grad)).item()),
                "nonzero": bool(mx.max(mx.abs(grad)).item() > 0),
            }
        )
    return {
        "cache_equivalent": bool(mx.allclose(logits, cached_logits, atol=1e-5)),
        "loss_finite": bool(mx.isfinite(loss).item()),
        "layers": layers,
    }
