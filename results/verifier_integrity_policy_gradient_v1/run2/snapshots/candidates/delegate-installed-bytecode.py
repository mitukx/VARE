"""Control: explicitly load the installed sample_utils Python bytecode."""

import marshal
from pathlib import Path


_CACHE = Path(
    "/usr/local/lib/python3.12/site-packages/mlx_lm/"
    "__pycache__/sample_utils.cpython-312.pyc"
)
_namespace = {
    "__name__": "vare_cached_mlx_lm_sample_utils",
    "__package__": "mlx_lm",
    "__file__": "/usr/local/lib/python3.12/site-packages/mlx_lm/sample_utils.py",
}
_compiled = marshal.loads(_CACHE.read_bytes()[16:])
exec(_compiled, _namespace)
_cached_apply_top_p = _namespace["apply_top_p"]


def apply_top_p(logprobs, top_p):
    return _cached_apply_top_p(logprobs, top_p)
