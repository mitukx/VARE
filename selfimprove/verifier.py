"""Score a model-written verdict against exact recomputation.

The model writes a final line ``#### <integer>``. It accepts the candidate
when that integer equals the candidate's parsed integer. The false-accept
rate is the fraction of ground-truth rejects that this verdict accepts.
The verdict is not executed as code.
"""

from __future__ import annotations

from minigrpo.expr import parse_final_integer

VERIFIER_INSTRUCTION = (
    "Recompute the expression. Decide whether the candidate final answer is correct.\n"
    "Expression: {expression}\n"
    "Candidate output:\n{candidate}\n"
    "Write your recomputed integer on the final line as #### <integer>.\n"
    "The final line must contain no other text."
)


def verifier_prompt(expression: str, candidate: str) -> str:
    return VERIFIER_INSTRUCTION.format(expression=expression, candidate=candidate)


def verdict_accepts(candidate_output: str, verdict_output: str) -> bool:
    candidate = parse_final_integer(candidate_output)
    verdict = parse_final_integer(verdict_output)
    if candidate is None or verdict is None:
        return False
    return verdict == candidate


def false_accept_rate(judgements: list[tuple[bool, bool]]) -> float:
    negatives = [accepted for correct, accepted in judgements if not correct]
    if not negatives:
        return 0.0
    return sum(1.0 for accepted in negatives if accepted) / len(negatives)
