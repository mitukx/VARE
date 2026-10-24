"""Incorrect control: never removes a token from the candidate distribution."""


def apply_top_p(logprobs, top_p):
    del top_p
    return logprobs
