# Task: load Qwen3 configurations using nested RoPE settings

Some Qwen3 configurations produced by newer Transformers versions store rotary-position settings in `rope_parameters` instead of the older top-level fields. Update the Qwen3 configuration/model-loading path so nested and older flat configurations remain supported, and invalid configurations fail clearly. Use a small locally constructed configuration to reproduce the issue; do not download model weights.
