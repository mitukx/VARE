import dataclasses
import os
import tempfile
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import utils
from mlx_lm.models import llama

def test_split_shard_loads_complete_local_stage():
    args = llama.ModelArgs(model_type="llama", hidden_size=64, num_hidden_layers=2,
                           intermediate_size=128, num_attention_heads=4,
                           rms_norm_eps=1e-5, vocab_size=64)
    model, config = utils.quantize_model(llama.Model(args), dataclasses.asdict(args),
                                         group_size=32, bits=4)
    weights = dict(tree_flatten(model.parameters()))
    orphan = "model.layers.0.mlp.down_proj"
    shard = {key: value for key, value in weights.items()
             if not key.startswith("model.layers.0.") or key.startswith(orphan + ".")}
    del shard[orphan + ".weight"]
    with tempfile.TemporaryDirectory() as folder:
        mx.save_safetensors(os.path.join(folder, "model.safetensors"), shard)
        utils.save_config(config, os.path.join(folder, "config.json"))
        loaded, _ = utils.load_model(Path(folder), lazy=True, strict=False)
        assert mx.array_equal(loaded.model.layers[1].mlp.down_proj.weight,
                              weights["model.layers.1.mlp.down_proj.weight"]).item()

test_split_shard_loads_complete_local_stage()
