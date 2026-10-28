"""Public, CPU-only compatibility check for the TimesFM fp16 pad value fix."""
from __future__ import annotations

import ast
from pathlib import Path
import types

import torch


PAD_VALUE = 1_123_581_321.0


class StrictFp16Torch:
    """Emulate the reported CUDA error for out-of-range fp16 scalar fills."""

    def __getattr__(self, name):
        if name != "full":
            return getattr(torch, name)

        def strict_full(size, fill_value, *args, **kwargs):
            dtype = kwargs.get("dtype")
            if dtype == torch.float16 and abs(float(fill_value)) > 65504:
                raise OverflowError("strict fp16 scalar conversion overflow")
            return torch.full(size, fill_value, *args, **kwargs)

        return strict_full


def load_probe():
    path = Path(__file__).parent / "src/transformers/models/timesfm/modeling_timesfm.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    model_class = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TimesFmModel"
    )
    transform = next(
        node for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_transform"
    )
    mean_std = ast.parse(
        "def _timesfm_masked_mean_std(self, inputs, patched_pads):\n"
        "    shape = inputs.shape[:2]\n"
        "    return (torch.zeros(shape, dtype=inputs.dtype, device=inputs.device), "
        "torch.ones(shape, dtype=inputs.dtype, device=inputs.device))\n"
    ).body[0]
    probe_class = ast.ClassDef(
        name="Probe", bases=[], keywords=[], decorator_list=[], body=[mean_std, transform],
    )
    probe_module = ast.fix_missing_locations(ast.Module(body=[probe_class], type_ignores=[]))
    namespace = {"torch": StrictFp16Torch()}
    exec(compile(probe_module, "timesfm_probe.py", "exec"), namespace)
    probe = namespace["Probe"]()
    probe.config = types.SimpleNamespace(pad_val=PAD_VALUE, tolerance=0.001)
    return probe


def test_large_pad_value_does_not_raise_for_fp16():
    model = load_probe()
    inputs = torch.tensor([[[1.0, 2.0]]], dtype=torch.float16)
    outputs, _ = model._forward_transform(inputs, torch.ones_like(inputs))
    assert torch.equal(outputs.reshape_as(inputs), inputs)


def test_padded_value_and_normal_values_are_preserved():
    model = load_probe()
    inputs = torch.tensor([[[PAD_VALUE, 3.0]]], dtype=torch.float32)
    outputs, _ = model._forward_transform(inputs, torch.ones_like(inputs))
    assert torch.equal(outputs.reshape_as(inputs), inputs)


if __name__ == "__main__":
    test_large_pad_value_does_not_raise_for_fp16()
    test_padded_value_and_normal_values_are_preserved()
