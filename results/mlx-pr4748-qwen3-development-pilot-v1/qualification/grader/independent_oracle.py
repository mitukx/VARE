"""Independent CPU oracle for MLX PR #4748; expected values use NumPy only."""
from __future__ import annotations

import json
import sys
import time

import mlx.core as mx
import numpy as np


def main() -> int:
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(20261025)
    inputs = rng.normal(size=(3, 4, 4)).astype(np.float32)
    inputs += np.eye(4, dtype=np.float32)[None, :, :] * 5.0
    results: dict[str, object] = {
        "case_seed": 20261025,
        "input_shape": list(inputs.shape),
        "device": str(mx.default_device()),
        "cases": {},
    }
    all_passed = True
    for upper in (False, True):
        tri = np.triu(inputs) if upper else np.tril(inputs)
        expected = np.stack([np.linalg.inv(matrix) for matrix in tri], axis=0)
        fn = lambda matrix: mx.linalg.tri_inv(matrix, upper=upper, stream=mx.cpu)
        direct = np.asarray(fn(mx.array(inputs)))
        axis0 = np.asarray(mx.vmap(fn, in_axes=(0,))(mx.array(inputs)))
        axis1 = np.asarray(
            mx.vmap(fn, in_axes=(1,))(mx.array(inputs.swapaxes(0, 1)))
        )
        for mode, actual in (("direct", direct), ("vmap_axis0", axis0), ("vmap_axis1", axis1)):
            max_abs = float(np.max(np.abs(actual - expected)))
            passed = bool(actual.shape == expected.shape and np.allclose(
                actual, expected, rtol=5e-4, atol=5e-5
            ))
            results["cases"][f"{'upper' if upper else 'lower'}_{mode}"] = {
                "passed": passed,
                "shape": list(actual.shape),
                "max_abs_error": max_abs,
            }
            all_passed = all_passed and passed
    results["passed"] = all_passed
    results["completed_unix_ns"] = time.time_ns()
    print(json.dumps(results, sort_keys=True))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
