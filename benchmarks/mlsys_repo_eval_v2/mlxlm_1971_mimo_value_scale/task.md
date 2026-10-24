Implement the optional MiMo-V2-Flash attention value scale. When the model
configuration provides `attention_value_scale`, apply it to projected values
before the cache update and attention computation for both full and
sliding-window attention. Configurations without the field must retain current
behavior. Only edit `mlx_lm/models/mimo_v2_flash.py`.
