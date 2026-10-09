from collections import Counter, defaultdict

import pytest

from trl.experimental.async_grpo.async_grpo_trainer import FixedCountBatcher, TokenBudgetBatcher


class FiniteStream:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


def row(row_id, rollout_id, rollout_size, tokens):
    return {
        "row_id": row_id,
        "rollout_id": rollout_id,
        "rollout_size": rollout_size,
        "input_ids": list(range(tokens)),
    }


def assert_batch_contract(batches, expected_row_ids, processes, max_rows=None, token_budget=None):
    seen = []
    rollout_batches = defaultdict(set)
    for batch_index, rank_rows in enumerate(batches):
        assert len(rank_rows) == processes
        assert all(rank_rows), "every emitted rank must receive at least one sample"
        flattened = [sample for rank in rank_rows for sample in rank]
        if max_rows is not None:
            assert len(flattened) <= max_rows
        if token_budget is not None:
            assert all(sum(len(sample["input_ids"]) for sample in rank) <= token_budget for rank in rank_rows)
        for sample in flattened:
            seen.append(sample["row_id"])
            rollout_batches[sample["rollout_id"]].add(batch_index)
    assert Counter(seen) == Counter(expected_row_ids), "each retained input row must be emitted exactly once"
    assert all(len(indices) == 1 for indices in rollout_batches.values()), "a rollout must not cross microbatches"


def test_fixed_count_keeps_rollouts_whole_and_accounts_for_rows():
    rows = [
        row("a0", "A", 3, 1), row("a1", "A", 3, 1), row("a2", "A", 3, 1),
        row("b0", "B", 2, 1), row("b1", "B", 2, 1),
    ]
    batches = list(FixedCountBatcher(FiniteStream(rows), num_processes=2, microbatch_size=4))
    assert_batch_contract(batches, [sample["row_id"] for sample in rows], 2, max_rows=4)


def test_token_budget_rebalances_whole_groups_instead_of_splitting_on_greedy_boundary():
    # Greedy per-sample placement leaves row loads 7 and 7 after placing B0,
    # so B1 appears not to fit. Repacking the full group set gives (7+3, 4+6).
    rows = [
        row("a0", "A", 2, 7), row("a1", "A", 2, 3),
        row("b0", "B", 2, 4), row("b1", "B", 2, 6),
    ]
    batches = list(TokenBudgetBatcher(
        FiniteStream(rows), num_processes=2, token_budget=10, metrics=defaultdict(list)
    ))
    assert_batch_contract(batches, [sample["row_id"] for sample in rows], 2, token_budget=10)
