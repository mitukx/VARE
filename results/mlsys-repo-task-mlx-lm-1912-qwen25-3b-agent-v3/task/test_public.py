"""Public regression from MLX-LM #1912, isolated to the candidate workspace."""

import importlib.util
from pathlib import Path

import mlx.core as mx

MODULE_PATH = Path(__file__).parent / "mlx_lm" / "sample_utils.py"
SPEC = importlib.util.spec_from_file_location("candidate_sample_utils", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("candidate sample_utils.py is unavailable")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
apply_top_p = MODULE.apply_top_p


def test_apply_top_p_keeps_top_token_at_tiny_thresholds() -> None:
    probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
    logits = mx.log(probs)
    for top_p in (1e-8, 1e-6, 1e-4):
        actual_probs = mx.softmax(apply_top_p(logits, top_p).squeeze())
        assert actual_probs.tolist() == [1.0, 0.0, 0.0, 0.0]

    mx.random.seed(0)
    values = mx.random.normal((2, 4096)) * 3
    logprobs = values - mx.logsumexp(values, axis=-1, keepdims=True)
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        current = logprobs.astype(dtype)
        for top_p in (1e-8, 1e-4, 1e-3):
            filtered = apply_top_p(current, top_p)
            kept = (filtered > -float("inf")).sum(axis=-1)
            assert kept.min().item() > 0
            assert filtered.max(axis=-1).tolist() == current.max(axis=-1).tolist()


if __name__ == "__main__":
    test_apply_top_p_keeps_top_token_at_tiny_thresholds()
    print("PASS: public top-p regression")
