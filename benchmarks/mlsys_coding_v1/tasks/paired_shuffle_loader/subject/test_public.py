"""Smoke check. This file is not the grader."""
from paired_loader import iter_batches


def test_ordered_batches_keep_pairs():
    features = ["a", "b", "c", "d"]
    labels = [1, 2, 3, 4]
    batches = iter_batches(features, labels, 3, seed=0, shuffle=False, drop_last=False)
    assert batches == [(["a", "b", "c"], [1, 2, 3]), (["d"], [4])]


if __name__ == "__main__":
    test_ordered_batches_keep_pairs()
