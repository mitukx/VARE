"""Execute the actual candidate GPTQ module without a reference implementation."""
import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

spec = importlib.util.spec_from_file_location("candidate_gptq", Path(__file__).with_name("target.py"))
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


class OneLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.proj = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.proj.weight = weight

    def __call__(self, x):
        return self.proj(x)


def run(request):
    bits, group = request["bits"], request["group_size"]
    model, _ = candidate.gptq_quantize(
        OneLinear(mx.array(request["weight"], dtype=mx.float32)),
        mx.array(request["data"], dtype=mx.float32),
        bits=bits, group_size=group, fallback_bits=bits,
        fallback_group_size=group, batch_size=32,
    )
    layer = model.proj
    result = mx.dequantize(layer.weight, layer.scales, layer.biases, group, bits)
    mx.eval(result)
    return result.astype(mx.float32).tolist()
