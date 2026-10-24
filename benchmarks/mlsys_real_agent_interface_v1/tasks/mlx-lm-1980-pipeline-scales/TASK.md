# Task: pipeline shards with split quantization metadata

In `mlx_lm/utils.py`, make `load_model` robust when a shard selected for one pipeline stage also contains quantization scales for a layer owned by another stage, but not that layer’s weight tensor. Loading should continue for the local stage and preserve its complete quantized layer weights. Do not change the treatment of layers whose local weights and scales are both present.

Only edit `mlx_lm/utils.py`. The public test constructs a tiny model and a split shard; the independent grader repeats this with a different layer count and boundary.
