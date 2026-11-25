"""Contract and small analytic controls for the Metal diagnostic."""
import importlib.util
import math

import pytest

from scripts import diagnose_grpo_metal_path as M

HAS_MLX = importlib.util.find_spec("mlx") is not None


def test_path_protocol_fixes_distinct_contexts_and_complete_matrix():
    p = M.protocol()
    assert p["arms"] == ["fp32", "bf16"]
    assert p["modes"] == ["stock_selected", "stock_full", "explicit_selected", "explicit_full"]
    assert [len(s["prompt_ids"]) for s in p["sequences"]] == [16, 48]
    assert p["tolerance"] == 1e-6
    assert len(p["sequences"]) * len(p["modes"]) == p["limits"]["maximum_model_path_measurements_per_arm"]


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_reference_attention_causal_and_grouped_query_cpu():
    import mlx.core as mx

    with mx.stream(mx.cpu):
        q = mx.zeros((1, 2, 2, 2))
        k = mx.zeros((1, 1, 2, 2))
        v = mx.array([[[[2.0, 4.0], [6.0, 8.0]]]])
        out = M.explicit_attention(q, k, v, scale=1.0, mask="causal")
        mx.eval(out)
        expected = [[2.0, 4.0], [4.0, 6.0]]
        assert out.tolist() == [[expected, expected]]
        with pytest.raises(ValueError, match="unsupported attention mask"):
            M.explicit_attention(q, k, v, scale=1.0, mask="unexpected")


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_native_score_matches_categorical_reference_cpu():
    import mlx.core as mx

    with mx.stream(mx.cpu):
        logits = mx.array([[[1.0, 2.0, 3.0]]])
        result = M.native_logprob(logits, 1, 0.7)
        expected = 2.0 / 0.7 - math.log(sum(math.exp(x / 0.7) for x in (1.0, 2.0, 3.0)))
        assert abs(result.item() - expected) <= 1e-6
