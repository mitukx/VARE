#!/usr/bin/env python3
"""Emit candidate-side observations for three pinned MLX core contracts.

The trusted parent evaluates this JSON after the process exits. No expected
values or task grader code are imported into this process.
"""
from __future__ import annotations

import argparse
import json


def capture(call):
    try:
        value = call()
        return {"value": value}
    except Exception as exc:  # capture backend errors as candidate output
        return {"error_type": type(exc).__name__, "error": str(exc)}


def observe() -> dict[str, object]:
    import mlx.core as mx
    import numpy as np

    mx.set_default_device(mx.cpu)
    take_input = mx.array([[0, 1, 2], [3, 4, 5]])
    take_cases = {
        "negative_axis0": capture(lambda: mx.take(take_input, -1, axis=0).tolist()),
        "negative_flat": capture(lambda: mx.take(take_input, -1).item()),
        "negative_axis1": capture(lambda: mx.take(take_input, -2, axis=1).tolist()),
        "positive_control": capture(lambda: mx.take(take_input, 1, axis=1).tolist()),
    }

    def normal_invalid_loc():
        value = mx.random.normal(shape=(3,), loc=mx.zeros((2, 3)))
        mx.eval(value)
        return list(value.shape)

    def normal_invalid_scale():
        value = mx.random.normal(shape=(3,), scale=mx.ones((2, 3)))
        mx.eval(value)
        return list(value.shape)

    def normal_valid_broadcast():
        value = mx.random.normal(
            shape=(2, 3), loc=mx.zeros((3,)), scale=mx.ones((2, 1))
        )
        mx.eval(value)
        array = np.asarray(value)
        return {"shape": list(array.shape), "all_finite": bool(np.isfinite(array).all())}

    normal_cases = {
        "invalid_loc": capture(normal_invalid_loc),
        "invalid_scale": capture(normal_invalid_scale),
        "valid_broadcast": capture(normal_valid_broadcast),
    }

    return {
        "mlx_version": str(mx.__version__),
        "mlx_core_module_path": str(mx.__file__),
        "default_device_is_cpu": mx.default_device() == mx.cpu,
        "take_negative_index": take_cases,
        "random_normal_shape": normal_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = observe()
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
