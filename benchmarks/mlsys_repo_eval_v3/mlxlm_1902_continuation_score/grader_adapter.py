"""Candidate-visible call shim; trusted expectations stay in the parent."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


class _OneRank:
    @staticmethod
    def rank() -> int:
        return 0

    @staticmethod
    def size() -> int:
        return 1


def run(case: dict) -> dict:
    import mlx.core as real_mx

    target = Path("/subject/mlx_lm/evaluate.py")
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLXLM"
    )
    method = next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == "loglikelihood"
    )
    # Execute only the target method; importing lm_eval and the rest of the
    # package would add unrelated dependencies to this pinned CPU fixture.
    method.decorator_list = []
    method.returns = None
    namespace = {
        "mx": SimpleNamespace(
            argmax=real_mx.argmax,
            array=real_mx.array,
            sum=real_mx.sum,
            all=real_mx.all,
            eval=real_mx.eval,
            argsort=real_mx.argsort,
            float32=real_mx.float32,
            bool_=real_mx.bool_,
            cpu=None,
            distributed=SimpleNamespace(
                all_max=lambda value, stream=None: real_mx.array(value),
                all_gather=lambda value, stream=None: value,
            ),
        ),
        "group": _OneRank(),
        "copy": copy,
        "tqdm": lambda values, total=None: values,
        "DEFAULT_MAX_TOKENS": 2048,
    }
    compiled = compile(
        ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
        str(target),
        "exec",
    )
    exec(compiled, namespace)
    instance = SimpleNamespace(_max_tokens=int(case["max_tokens"]))
    instance._tokenize = MagicMock(
        side_effect=[case["context_tokens"], case["complete_tokens"]]
    )
    instance._process_prompt = MagicMock(
        return_value=(-real_mx.arange(6, dtype=real_mx.float32)[None, :], [])
    )

    def score_fn(inputs, cache=None):
        targets = inputs[:, 1:]
        return (
            -targets.astype(real_mx.float32),
            None,
            real_mx.ones(targets.shape, dtype=real_mx.bool_),
        )

    instance._score_fn = MagicMock(side_effect=score_fn)
    request = SimpleNamespace(args=("context", " continuation"))
    result = namespace["loglikelihood"](instance, [request])
    inputs = instance._score_fn.call_args[0][0]
    return {
        "result": [[float(score), bool(greedy)] for score, greedy in result],
        "processed_prefix": instance._process_prompt.call_args.args[0],
        "scored_inputs": inputs.tolist(),
        "score_invocations": instance._score_fn.call_count,
    }
