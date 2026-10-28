"""Public executable smoke checks; run only by the pinned isolated test runner."""
import sys
import types
from pathlib import Path

root = Path(__file__).resolve().parent
pkg = types.ModuleType("mlx_lm")
pkg.__path__ = [str(root / "mlx_lm")]
sys.modules["mlx_lm"] = pkg
models = types.ModuleType("mlx_lm.models")
models.__path__ = [str(root / "mlx_lm" / "models")]
sys.modules["mlx_lm.models"] = models
cache = types.ModuleType("mlx_lm.models.cache")
for name in ("QuantizedKVCache", "TokenBuffer"):
    setattr(cache, name, type(name, (), {}))
for name in ("can_trim_prompt_cache", "load_prompt_cache", "trim_prompt_cache"):
    setattr(cache, name, lambda *args, **kwargs: None)
cache.make_prompt_cache = lambda *args, **kwargs: []
sys.modules["mlx_lm.models.cache"] = cache
sample = types.ModuleType("mlx_lm.sample_utils")
sample.make_sampler = lambda *args, **kwargs: None
sys.modules["mlx_lm.sample_utils"] = sample
tokenizer = types.ModuleType("mlx_lm.tokenizer_utils")
tokenizer.TokenizerWrapper = type("TokenizerWrapper", (), {})
sys.modules["mlx_lm.tokenizer_utils"] = tokenizer
utils = types.ModuleType("mlx_lm.utils")
utils.does_model_support_input_embeddings = lambda *args, **kwargs: False
utils.load = lambda *args, **kwargs: None
sys.modules["mlx_lm.utils"] = utils
transformers = types.ModuleType("transformers")
transformers.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
sys.modules["transformers"] = transformers

import mlx.core as mx
from mlx_lm.generate import generate_step


class DummyModel:
    def __call__(self, inputs, cache=None, input_embeddings=None):
        return mx.zeros((inputs.shape[0], inputs.shape[1], 64))


def histories(prompt_values, prefill_step_size):
    observed = []

    def processor(tokens, logits):
        observed.append([int(token) for token in tokens.tolist()])
        return logits

    generator = generate_step(
        mx.array(prompt_values),
        DummyModel(),
        max_tokens=1,
        logits_processors=[processor],
        prefill_step_size=prefill_step_size,
    )
    next(generator)
    return observed


def test_prefill_history_contains_complete_prompt():
    observed = histories([2, 5, 7, 9, 11, 13], 2)
    assert observed[0] == [2, 5, 7, 9, 11, 13]


def test_next_logits_step_retains_prefill_and_generated_token():
    prompt = [11, 13, 17, 19, 23, 29, 31, 37]
    observed = histories(prompt, 3)
    assert observed[0] == prompt
    assert observed[1] == prompt + [0]

