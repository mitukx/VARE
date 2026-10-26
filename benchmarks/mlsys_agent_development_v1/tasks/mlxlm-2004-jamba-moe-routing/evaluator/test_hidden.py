from __future__ import annotations

import unittest

import mlx.core as mx

from mlx_lm.models import jamba


def _args():
    return jamba.ModelArgs(
        model_type="jamba",
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        attn_layer_offset=1,
        attn_layer_period=2,
        expert_layer_offset=1,
        expert_layer_period=2,
        mamba_d_conv=4,
        mamba_d_state=4,
        mamba_expand=2,
        num_experts=8,
        num_experts_per_tok=2,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        vocab_size=32,
    )


class TestJambaRouting(unittest.TestCase):
    def setUp(self):
        self._device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self._device)

    def test_sparse_moe_uses_full_router_distribution_without_topk_renormalization(self):
        args = _args()
        block = jamba.JambaSparseMoeBlock(args)
        # The input selects a controlled, strictly ordered set of router logits.
        logits = [3, 2, 1, 0, -1, -2, -3, -4]
        block.router.weight = mx.array([
            [float(logit)] + [0.0] * (args.hidden_size - 1) for logit in logits
        ])
        block.switch_mlp = lambda x, inds: (
            inds[..., None] == mx.arange(args.num_experts)
        ).astype(x.dtype)
        x = mx.zeros((1, 3, args.hidden_size))
        x[..., 0] = 1

        observed = block(x)
        gates = block.router(x)
        probabilities = mx.softmax(gates.astype(mx.float32), axis=-1)
        cutoff = mx.sort(probabilities, axis=-1)[..., -args.num_experts_per_tok]
        expected = mx.where(probabilities >= cutoff[..., None], probabilities, 0).astype(x.dtype)

        self.assertTrue(mx.allclose(observed, expected, atol=1e-6).item())
        self.assertTrue(mx.all(observed.sum(axis=-1) < 1).item())


if __name__ == "__main__":
    unittest.main()
