"""Dependency-light execution of the pinned TRL batcher class definitions."""
import ast
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class IterableDataset:
    pass


def load_batchers():
    source = Path.cwd() / "trl/experimental/async_grpo/async_grpo_trainer.py"
    tree = ast.parse(source.read_text())
    wanted = {"_balance_by_squared_length", "FixedCountBatcher", "TokenBudgetBatcher"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in wanted]
    if {getattr(node, "name", None) for node in nodes} != wanted:
        raise RuntimeError("pinned TRL source is missing a required batcher definition")
    namespace = {
        "Any": Any,
        "torch": SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(IterableDataset=IterableDataset))),
        "logger": logging.getLogger("trl-batcher-audit"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["FixedCountBatcher"], namespace["TokenBudgetBatcher"]


class FiniteStream:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


def row(row_id, rollout_id, rollout_size, tokens):
    return {"row_id": row_id, "rollout_id": rollout_id, "rollout_size": rollout_size,
            "input_ids": list(range(tokens))}


def inspect(batches, expected_ids, processes, max_rows=None, budget=None):
    seen = []
    rollout_batches = defaultdict(set)
    empty_rank = False
    count_cap_ok = True
    token_cap_ok = True
    for batch_index, rank_rows in enumerate(batches):
        if len(rank_rows) != processes:
            empty_rank = True
            continue
        if any(not rank for rank in rank_rows):
            empty_rank = True
        flattened = [sample for rank in rank_rows for sample in rank]
        if max_rows is not None and len(flattened) > max_rows:
            count_cap_ok = False
        if budget is not None and any(sum(len(sample["input_ids"]) for sample in rank) > budget for rank in rank_rows):
            token_cap_ok = False
        for sample in flattened:
            seen.append(sample["row_id"])
            rollout_batches[sample["rollout_id"]].add(batch_index)
    return {
        "row_accounting_exact": Counter(seen) == Counter(expected_ids),
        "rollouts_unsplit": all(len(indices) == 1 for indices in rollout_batches.values()),
        "nonempty_rank_rows": not empty_rank,
        "sample_cap_respected": count_cap_ok,
        "token_cap_respected": token_cap_ok,
        "emitted_batches": len(batches),
    }


def main():
    FixedCountBatcher, TokenBudgetBatcher = load_batchers()
    fixed_rows = [row("a0", "A", 3, 1), row("a1", "A", 3, 1), row("a2", "A", 3, 1),
                  row("b0", "B", 2, 1), row("b1", "B", 2, 1),
                  row("c0", "C", 2, 1), row("c1", "C", 2, 1),
                  row("d0", "D", 1, 1), row("e0", "E", 1, 1)]
    fixed = list(FixedCountBatcher(FiniteStream(fixed_rows), num_processes=2, microbatch_size=4))
    token_rows = [row("a0", "A", 2, 7), row("a1", "A", 2, 3),
                  row("b0", "B", 2, 4), row("b1", "B", 2, 6),
                  row("c0", "C", 2, 2), row("c1", "C", 2, 2),
                  row("d0", "D", 2, 9), row("d1", "D", 2, 1)]
    token = list(TokenBudgetBatcher(FiniteStream(token_rows), num_processes=2, token_budget=10,
                                    metrics=defaultdict(list)))
    result = {
        "source": str(Path.cwd() / "trl/experimental/async_grpo/async_grpo_trainer.py"),
        "fixed_count": inspect(fixed, [r["row_id"] for r in fixed_rows], 2, max_rows=4),
        "token_budget": inspect(token, [r["row_id"] for r in token_rows], 2, budget=10),
    }
    result["all_contracts_pass"] = all(
        value for mode in ("fixed_count", "token_budget") for value in result[mode].values()
        if isinstance(value, bool)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
