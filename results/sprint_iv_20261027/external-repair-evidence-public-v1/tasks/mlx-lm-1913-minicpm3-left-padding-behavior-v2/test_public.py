import mlx.core as mx
from mlx_lm.generate import BatchGenerator, generate_step
from mlx_lm.models import minicpm3

def _model():
    stream = mx.default_stream(mx.cpu)
    mx.random.seed(0)
    args = minicpm3.ModelArgs(
        model_type="minicpm3", hidden_size=64, dim_model_base=64,
        num_hidden_layers=2, intermediate_size=128, num_attention_heads=4,
        num_key_value_heads=4, rms_norm_eps=1e-5, vocab_size=100,
        q_lora_rank=32, qk_nope_head_dim=8, qk_rope_head_dim=8,
        kv_lora_rank=16, scale_depth=1.4, scale_emb=1.0,
        max_position_embeddings=512,
        rope_scaling={"type":"longrope", "long_factor":[1.0]*4,
                      "short_factor":[1.0]*4, "original_max_position_embeddings":512},
    )
    return minicpm3.Model(args), stream

def test_left_padded_batch_first_token_matches_individual():
    model, stream = _model()
    prompts = [list(range(1, 13)), [7, 8, 9]]
    reference = [next(generate_step(mx.array(prompt), model, stream=stream, max_tokens=1))
                 for prompt in prompts]
    gen = BatchGenerator(model, stream=stream, max_tokens=1, prefill_step_size=4)
    try:
        uids = gen.insert(prompts)
        responses = []
        while batch := gen.next_generated():
            responses.extend(batch)
    finally:
        gen.close()
    by_uid = {item.uid:item for item in responses}
    for uid, (token, logprobs) in zip(uids, reference):
        assert by_uid[uid].token == token
        assert mx.allclose(by_uid[uid].logprobs, logprobs, rtol=1e-4, atol=1e-4).item()
