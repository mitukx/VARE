import mlx.core as mx
from mlx_lm.models import falcon_h1

def _model():
    mx.random.seed(0)
    args = falcon_h1.ModelArgs(
        model_type="falcon_h1", hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, vocab_size=64, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, mamba_d_ssm=32,
        mamba_n_heads=2, mamba_d_head=16, mamba_d_state=8,
    )
    return falcon_h1.Model(args)

def test_uncached_logits_match_explicit_cache():
    model = _model()
    inputs = mx.array([[0, 1, 2], [2, 1, 0]])
    uncached = model(inputs)
    cached = model(inputs, cache=model.make_cache())
    mx.eval(uncached, cached)
    assert mx.allclose(uncached, cached, atol=1e-5).item()
