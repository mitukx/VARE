"""Smoke check. This file is not the grader."""

from gqa_repeat import repeat_kv_heads


def test_one_repeat_keeps_values():
    keys = [[1.0, 2.0], [3.0, 4.0]]
    assert repeat_kv_heads(keys, 1) == keys
    assert keys == [[1.0, 2.0], [3.0, 4.0]]


if __name__ == "__main__":
    test_one_repeat_keeps_values()
