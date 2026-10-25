# Task: reject unsupported compressed-tensors formats

When loading a model that declares the `compressed-tensors` quantization method,
MLX-LM currently treats unrecognized formats as affine 4-bit weights. This can
silently reinterpret floating-point quantized (FP8) checkpoint weights as a
different packed representation. Update format handling so supported packed
formats retain their intended quantization parameters and unsupported formats
fail clearly before quantization begins.

Keep the change limited to `mlx_lm/utils.py`. Do not change tests or dependency
metadata.
