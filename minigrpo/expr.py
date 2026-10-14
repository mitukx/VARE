"""Exact integer expressions and the strict final-line parser.

The evaluator accepts non-negative integers and ``+ - * / ( )``. Division must
be exact. The parser accepts only a final line of the form ``#### <integer>``.
"""

from __future__ import annotations

import re

_EXPR_TOKEN = re.compile(r"\d+|[()+\-*/]")
FINAL_LINE = re.compile(r"^####\s+(-?\d+)\s*$")


def evaluate_expression(expression: str) -> int:
    compact = expression.replace(" ", "")
    tokens = _EXPR_TOKEN.findall(compact)
    if "".join(tokens) != compact or not tokens:
        raise ValueError(f"unsupported expression: {expression}")
    value, index = _parse_sum(tokens, 0)
    if index != len(tokens):
        raise ValueError(f"trailing expression tokens: {expression}")
    return value


def _parse_sum(tokens: list[str], index: int) -> tuple[int, int]:
    value, index = _parse_product(tokens, index)
    while index < len(tokens) and tokens[index] in {"+", "-"}:
        op = tokens[index]
        rhs, index = _parse_product(tokens, index + 1)
        value = value + rhs if op == "+" else value - rhs
    return value, index


def _parse_product(tokens: list[str], index: int) -> tuple[int, int]:
    value, index = _parse_unary(tokens, index)
    while index < len(tokens) and tokens[index] in {"*", "/"}:
        op = tokens[index]
        rhs, index = _parse_unary(tokens, index + 1)
        if op == "*":
            value *= rhs
        else:
            if rhs == 0 or value % rhs != 0:
                raise ValueError("division is not exact")
            value //= rhs
    return value, index


def _parse_unary(tokens: list[str], index: int) -> tuple[int, int]:
    if index >= len(tokens):
        raise ValueError("expression ended early")
    if tokens[index] == "(":
        value, index = _parse_sum(tokens, index + 1)
        if index >= len(tokens) or tokens[index] != ")":
            raise ValueError("missing closing parenthesis")
        return value, index + 1
    if not tokens[index].isdigit():
        raise ValueError(f"expected integer, found {tokens[index]}")
    return int(tokens[index]), index + 1


def parse_final_integer(text: str) -> int | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    match = FINAL_LINE.fullmatch(lines[-1])
    if match is None:
        return None
    return int(match.group(1))
