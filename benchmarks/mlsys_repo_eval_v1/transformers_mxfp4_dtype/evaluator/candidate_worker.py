from __future__ import annotations
import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path
import torch

dumps = json.dumps
write = sys.stdout.write
request = json.load(sys.stdin)

class _Logger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class ConversionOps:
    pass

class _IdentityOp(ConversionOps):
    pass

def _get_module_from_name(model, full_layer_name):
    parts = full_layer_name.split('.')
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return obj, parts[-1]

utils = types.ModuleType('transformers.utils')
utils.get_current_accelerator = lambda: None
utils.is_torch_available = lambda: True
utils.logging = types.SimpleNamespace(get_logger=lambda name: _Logger())
core = types.ModuleType('transformers.core_model_loading')
core.ConversionOps = ConversionOps
core._IdentityOp = _IdentityOp
dist = types.ModuleType('transformers.distributed')
dist.__path__ = []
dist_utils = types.ModuleType('transformers.distributed.utils')
dist_utils._is_torch_distributed_initialized = lambda: False
quant = types.ModuleType('transformers.quantizers')
quant.__path__ = []
quant_utils = types.ModuleType('transformers.quantizers.quantizers_utils')
quant_utils.get_module_from_name = _get_module_from_name
quant_utils.on_device = lambda device: contextlib.nullcontext()
quant_utils.should_convert_module = lambda *args, **kwargs: False
integrations = types.ModuleType('transformers.integrations')
integrations.__path__ = []
hub = types.ModuleType('transformers.integrations.hub_kernels')
hub.use_kernel_forward_from_hub = lambda name: (lambda fn: fn)
root = types.ModuleType('transformers')
root.__path__ = []
for mod in (root, integrations, dist, quant):
    sys.modules[mod.__name__] = mod
for mod in (utils, core, dist_utils, quant_utils, hub):
    sys.modules[mod.__name__] = mod

source = Path('src/transformers/integrations/mxfp4.py')
spec = importlib.util.spec_from_file_location('transformers.integrations.mxfp4_under_test', source)
if spec is None or spec.loader is None:
    raise RuntimeError('cannot load candidate MXFP4 module')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

rows = []
for case in request['cases']:
    dtype = getattr(torch, case['dtype'])
    blocks = torch.tensor(case['blocks'], dtype=torch.uint8)
    scales = torch.tensor(case['scales'], dtype=torch.uint8)
    model = types.SimpleNamespace()
    target = types.SimpleNamespace()
    setattr(target, case['projection'], torch.nn.Parameter(torch.empty((1,), dtype=dtype, device='meta')))
    model.target = target
    result = module.Mxfp4Dequantize(hf_quantizer=None).convert(
        {f"{case['projection']}_blocks": blocks, f"{case['projection']}_scales": scales},
        model=model,
        full_layer_name=f"target.{case['projection']}",
    )[f"target.{case['projection']}"]
    rows.append({
        'case_id': case['case_id'],
        'dtype': str(result.dtype),
        'shape': list(result.shape),
        'values': result.detach().cpu().tolist(),
    })
write(dumps({'schema_version': 1, 'nonce': request['nonce'], 'cases': rows}, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
