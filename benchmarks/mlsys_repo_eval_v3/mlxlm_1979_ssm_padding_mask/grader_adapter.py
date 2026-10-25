"""Candidate-visible source call shim; trusted expected outcomes stay in parent."""

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
    from mlx_lm.models.ssm import ssm_attn

    mx.random.seed(int(case["seed"]))
    batch_size = 1
    n_group = 1
    num_heads = 48
    head_dim = 64
    state_dim = 128
    seq_len = 4
    pad = 2

    hidden_states = mx.random.normal(
        shape=(batch_size, seq_len + pad, num_heads, head_dim)
    )
    B = mx.random.normal(shape=(batch_size, seq_len + pad, n_group, state_dim))
    C = mx.random.normal(shape=(batch_size, seq_len + pad, n_group, state_dim))
    dt = mx.random.normal(shape=(batch_size, seq_len + pad, num_heads))
    dt_bias = mx.random.normal(shape=(num_heads,))
    A_log = mx.random.normal(shape=(num_heads,))
    D = mx.random.normal(shape=(num_heads,))
    state = mx.random.normal(shape=(batch_size, num_heads, head_dim, state_dim))

    reference, reference_state = ssm_attn(
        hidden_states[:, :-pad],
        A_log,
        B[:, :-pad],
        C[:, :-pad],
        D,
        dt[:, :-pad],
        dt_bias,
        state,
    )
    mask = mx.array([[True] * seq_len + [False] * pad])
    lengths = mx.array([seq_len])
    masked, masked_state = ssm_attn(
        hidden_states,
        A_log,
        B,
        C,
        D,
        dt,
        dt_bias,
        state,
        mask=mask,
        lengths=lengths,
    )
    return {
        "valid_outputs_match_unpadded": bool(
            mx.allclose(reference, masked[:, :-pad], atol=1e-4, rtol=1e-4)
        ),
        "final_state_matches_unpadded": bool(
            mx.allclose(reference_state, masked_state, atol=1e-4, rtol=1e-4)
        ),
        "outputs_finite": bool(mx.all(mx.isfinite(masked)).item()),
        "state_finite": bool(mx.all(mx.isfinite(masked_state)).item()),
    }
