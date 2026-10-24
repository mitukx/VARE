"""Control: delegate to the package implementation available at runtime."""

from mlx_lm.sample_utils import apply_top_p as _installed_apply_top_p


def apply_top_p(logprobs, top_p):
    return _installed_apply_top_p(logprobs, top_p)
