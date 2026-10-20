"""Smoke check. This file is not the grader."""

from causal_attention import causal_attention_weights


def test_result_keeps_the_input_shape():
    weights = causal_attention_weights([[0.0, 1.0], [0.0, 0.0]])
    assert len(weights) == 2 and all(len(row) == 2 for row in weights)


if __name__ == "__main__":
    test_result_keeps_the_input_shape()
