"""Independent observable-contract checks for compressed-tensors format routing."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from mlx_lm import utils


class EmptyArgs:
    @classmethod
    def from_dict(cls, config):
        return cls()


class EmptyModel:
    def __init__(self, args):
        self.args = args

    def eval(self):
        pass

    def load_weights(self, weights, strict=True):
        assert weights == []


class CompressedTensorFormatContract(unittest.TestCase):
    def load_config(self, fmt, quantize):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "quantization_config": {"quant_method": "compressed-tensors", "format": fmt}
            }))
            model, config = utils.load_model(
                root, lazy=True, strict=False,
                get_model_classes=lambda config: (EmptyModel, EmptyArgs),
            )
        return model, config

    def test_float_quantized_is_rejected_before_quantization(self):
        calls = []
        with mock.patch.object(utils.nn, "quantize", side_effect=lambda model, **kw: calls.append(kw)):
            with self.assertRaisesRegex(ValueError, "float-quantized"):
                self.load_config("float-quantized", calls)
        self.assertEqual(calls, [])

    def test_nvfp4_format_preserves_mode(self):
        calls = []
        with mock.patch.object(utils.nn, "quantize", side_effect=lambda model, **kw: calls.append(kw)):
            _, config = self.load_config("nvfp4-pack-quantized", calls)
        self.assertEqual(config["quantization"], {"group_size": 16, "bits": 4, "mode": "nvfp4"})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["mode"], "nvfp4")

    def test_mxfp4_format_preserves_mode(self):
        calls = []
        with mock.patch.object(utils.nn, "quantize", side_effect=lambda model, **kw: calls.append(kw)):
            _, config = self.load_config("mxfp4-pack-quantized", calls)
        self.assertEqual(config["quantization"], {"group_size": 32, "bits": 4, "mode": "mxfp4"})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["mode"], "mxfp4")

    def test_packed_affine_format_remains_supported(self):
        calls = []
        with mock.patch.object(utils.nn, "quantize", side_effect=lambda model, **kw: calls.append(kw)):
            _, config = self.load_config("pack-quantized", calls)
        self.assertEqual(config["quantization"], {"group_size": 32, "bits": 4, "mode": "affine"})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["mode"], "affine")


if __name__ == "__main__":
    unittest.main()
