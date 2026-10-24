import dataclasses
import os
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import utils
from mlx_lm.models import llama


def make_model():
    args = llama.ModelArgs(model_type="llama", hidden_size=64, num_hidden_layers=3,
                           intermediate_size=96, num_attention_heads=4,
                           rms_norm_eps=1e-5, vocab_size=64)
    return utils.quantize_model(llama.Model(args), dataclasses.asdict(args),
                                group_size=32, bits=4)


def load_from_shard(weights, config, folder):
    mx.save_safetensors(os.path.join(folder, "model.safetensors"), weights)
    utils.save_config(config, os.path.join(folder, "config.json"))
    return utils.load_model(Path(folder), lazy=True, strict=False)[0]


def check_clean_stage():
    model, config = make_model()
    weights = dict(tree_flatten(model.parameters()))
    with tempfile.TemporaryDirectory() as folder:
        loaded = load_from_shard(weights, config, folder)
        for layer in range(3):
            name = f"model.layers.{layer}.mlp.down_proj.weight"
            assert mx.array_equal(loaded.model.layers[layer].mlp.down_proj.weight,
                                  weights[name]).item(), name


def check_orphan_stage(orphan_layer):
    model, config = make_model()
    weights = dict(tree_flatten(model.parameters()))
    orphan = f"model.layers.{orphan_layer}.mlp.down_proj"
    shard = {key: value for key, value in weights.items()
             if not key.startswith(f"model.layers.{orphan_layer}.") or key.startswith(orphan + ".")}
    del shard[orphan + ".weight"]
    assert orphan + ".scales" in shard
    with tempfile.TemporaryDirectory() as folder:
        loaded = load_from_shard(shard, config, folder)
        for layer in range(3):
            if layer == orphan_layer:
                continue
            name = f"model.layers.{layer}.mlp.down_proj.weight"
            assert mx.array_equal(loaded.model.layers[layer].mlp.down_proj.weight,
                                  weights[name]).item(), name


check_clean_stage()
check_orphan_stage(0)
check_orphan_stage(1)
