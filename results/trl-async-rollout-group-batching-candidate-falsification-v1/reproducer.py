import ast
from collections import defaultdict
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class IterableDataset:
    pass


class FiniteStream:
    def __init__(self, rows):
        self.rows = rows
    def __iter__(self):
        return iter(self.rows)


def load_batchers():
    source = Path.cwd() / "trl/experimental/async_grpo/async_grpo_trainer.py"
    tree = ast.parse(source.read_text())
    wanted = {"_balance_by_squared_length", "FixedCountBatcher", "TokenBudgetBatcher"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in wanted]
    namespace = {"Any": Any, "torch": SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(IterableDataset=IterableDataset))),
                 "logger": logging.getLogger("candidate-falsification")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["FixedCountBatcher"], namespace["TokenBudgetBatcher"]


def row(name, rollout, size, tokens):
    return {"row_id": name, "rollout_id": rollout, "rollout_size": size, "input_ids": list(range(tokens))}


def observe(fn):
    try:
        batches = fn()
        return {"outcome": "emitted", "batch_count": len(batches), "row_count": sum(len(rank) for batch in batches for rank in batch)}
    except Exception as exc:
        return {"outcome": "raised", "exception_type": type(exc).__name__, "message": str(exc)}


def main():
    FixedCountBatcher, TokenBudgetBatcher = load_batchers()
    token_rows = [row(f"t{i}", "T", 5, n) for i, n in enumerate([3, 3, 2, 2, 2])]
    oversized_rows = [row(f"o{i}", "O", 5, 1) for i in range(5)]
    prefix_rows = [row("p0", "P", 1, 1)] + [row(f"q{i}", "Q", 4, 1) for i in range(4)]
    result = {
        "source": str(Path.cwd() / "trl/experimental/async_grpo/async_grpo_trainer.py"),
        "token_greedy_false_negative": {
            "exact_feasible_partition": [[3, 3], [2, 2, 2]],
            "candidate": observe(lambda: list(TokenBudgetBatcher(FiniteStream(token_rows), 2, 6, defaultdict(list)))),
        },
        "fixed_oversized_rollout": {
            "declared_constraints": {"processes": 2, "microbatch_size": 4, "rollout_size": 5},
            "candidate": observe(lambda: list(FixedCountBatcher(FiniteStream(oversized_rows), 2, 4))),
        },
        "fixed_unpartitionable_contiguous_prefix": {
            "declared_constraints": {"processes": 2, "microbatch_size": 4, "rollout_sizes": [1, 4]},
            "candidate": observe(lambda: list(FixedCountBatcher(FiniteStream(prefix_rows), 2, 4))),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
