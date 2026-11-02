#!/usr/bin/env python3
"""Run one bounded MLX model check without loading trusted test code."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any


def _array_payload(value: Any, *, np: Any) -> tuple[list[Any], bool]:
    array = np.asarray(value)
    values = array.tolist()
    finite = bool(np.isfinite(array).all())
    return values, finite


def _gradient_payload(
    gradients: Any, *, mx: Any, np: Any, flatten: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, value in flatten(gradients):
        array = value.astype(mx.float32)
        mx.eval(array)
        materialized = np.asarray(array)
        rows.append(
            {
                "name": str(name),
                "shape": list(materialized.shape),
                "finite": bool(np.isfinite(materialized).all()),
                "nonzero_count": int(np.count_nonzero(materialized)),
                "max_abs": float(np.abs(materialized).max(initial=0.0)),
            }
        )
    return rows


def main() -> int:
    # Capture trusted reporting helpers before importing candidate model code.
    dumps = json.dumps
    write = sys.stdout.write
    request = json.load(sys.stdin)
    model_name = request["model"]
    module_name = f"mlx_lm.models.{model_name}"

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten

    trusted_eval = mx.eval
    trusted_array = mx.array
    trusted_value_and_grad = nn.value_and_grad
    trusted_flatten = tree_flatten
    seed = int(request["seed"])
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    from importlib import import_module

    module = import_module(module_name)
    module_origin = getattr(module, "__file__", None)
    if not isinstance(module_origin, str):
        raise RuntimeError("candidate model module has no filesystem origin")
    model_args = module.ModelArgs(**request["model_args"])
    model = module.Model(model_args)
    tokens = trusted_array(request["tokens"], dtype=mx.int32)
    logits = model(tokens)
    trusted_eval(logits)
    logits_list, logits_finite = _array_payload(logits, np=np)
    result: dict[str, Any] = {
        "schema_version": 1,
        "model": model_name,
        "nonce": request["nonce"],
        "module_origin": str(Path(module_origin).resolve()),
        "logits": logits_list,
        "logits_finite": logits_finite,
    }

    if request.get("compute_gradients", False):

        def loss_fn(current_model: Any, inputs: Any) -> Any:
            return current_model(inputs).mean()

        loss, gradients = trusted_value_and_grad(model, loss_fn)(model, tokens)
        trusted_eval(loss, gradients)
        result["loss"] = float(loss)
        result["loss_finite"] = math.isfinite(result["loss"])
        result["gradients"] = _gradient_payload(
            gradients, mx=mx, np=np, flatten=trusted_flatten
        )

    write(dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
