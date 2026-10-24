"""Incorrect control: removes every token, including the most likely token."""

import mlx.core as mx


def apply_top_p(logprobs, top_p):
    del top_p
    return mx.full_like(logprobs, -float("inf"))
