"""Frozen, parent-owned function-repair contracts for the Qwen3 GRPO pilot.

The prompts contain task instructions only. Expected values stay in this
module and are passed only to ``vare.isolated_candidate.grade_calls``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vare.evidence import sha256_json


@dataclass(frozen=True)
class ContractCase:
    case_id: str
    args: tuple[Any, ...]
    expected: Any


@dataclass(frozen=True)
class FunctionRepairTask:
    task_id: str
    split: str
    prompt: str
    cases: tuple[ContractCase, ...]

    def sha256(self) -> str:
        return sha256_json(asdict(self))


TRAIN_TASKS = (
    FunctionRepairTask(
        "train.merge_closed_intervals.v1", "train",
        "Implement `solve(intervals)`. Each item is a valid closed integer interval `[start, end]`. "
        "Return a list of intervals sorted by start. Merge intervals only when they overlap or share an endpoint; "
        "keep integer-adjacent intervals such as `[1, 2]` and `[3, 4]` separate. Return `[]` for empty input.",
        (
            ContractCase("empty", ([],), []),
            ContractCase("overlap", ([[1, 3], [2, 5], [8, 10]],), [[1, 5], [8, 10]]),
            ContractCase("unsorted_touching", ([[5, 7], [1, 2], [2, 4]],), [[1, 4], [5, 7]]),
            ContractCase("integer_adjacent", ([[1, 2], [3, 4]],), [[1, 2], [3, 4]]),
            ContractCase("negative_and_duplicate", ([[0, 0], [-3, -1], [-2, 1], [-3, -1]],), [[-3, 1]]),
        ),
    ),
    FunctionRepairTask(
        "train.maximum_fixed_window.v1", "train",
        "Implement `solve(values, k)`. For an integer list, find a contiguous window of exactly `k` items with the "
        "largest sum. Return `[sum, start_index]`; on a tie choose the smallest start index. Return `None` when "
        "`k < 1` or `k > len(values)`. The input values are integers.",
        (
            ContractCase("ordinary", ([2, -1, 3, 4], 2), [7, 2]),
            ContractCase("all_negative", ([-5, -2, -7], 2), [-7, 0]),
            ContractCase("earliest_tie", ([1, 2, 1, 2], 2), [3, 0]),
            ContractCase("too_large", ([1], 2), None),
            ContractCase("nonpositive", ([1, 2], 0), None),
        ),
    ),
    FunctionRepairTask(
        "train.top_k_words.v1", "train",
        "Implement `solve(words, k)`. Return at most `k` distinct strings, ordered by frequency from highest to "
        "lowest; break equal-frequency ties by Python string ordering (Unicode code points), ascending. Matching is "
        "case-sensitive. Return `[]` when `k <= 0` or `words` is empty.",
        (
            ContractCase("frequency", (["pear", "apple", "pear", "banana", "apple", "apple"], 2),
                         ["apple", "pear"]),
            ContractCase("lexical_tie", (["b", "a", "b", "a", "c"], 2), ["a", "b"]),
            ContractCase("case_sensitive", (["Cat", "cat", "Cat"], 5), ["Cat", "cat"]),
            ContractCase("empty", ([], 3), []),
            ContractCase("zero_k", (["x"], 0), []),
        ),
    ),
    FunctionRepairTask(
        "train.parse_env_lines.v1", "train",
        "Implement `solve(lines)`. Ignore blank lines and lines whose first non-whitespace character is `#`. "
        "For other lines, split at the first `=`; skip lines without `=` and keys that are empty after trimming. "
        "Trim key and value whitespace. If the trimmed value starts and ends with the same single or double quote "
        "and has at least two characters, remove exactly that outer quote pair; do not process escapes. Later "
        "assignments replace earlier values. Return a dictionary of strings.",
        (
            ContractCase("comments_and_duplicate", (["# comment", "A=1", "B=two=three", "A=3"],),
                         {"A": "3", "B": "two=three"}),
            ContractCase("spaces_and_quotes", (["  # c", " =empty key", "=missing", "X = ' quoted '",
                                                "Y=\"a=b\""],), {"X": " quoted ", "Y": "a=b"}),
            ContractCase("unmatched_quote", (["A=\"unterminated", "B='ok'"],),
                         {"A": "\"unterminated", "B": "ok"}),
            ContractCase("blank_and_missing", (["", "   ", "no separator", "#x"],), {}),
        ),
    ),
    FunctionRepairTask(
        "train.rotate_matrix_clockwise.v1", "train",
        "Implement `solve(matrix)`. The input is a rectangular list of rows containing integers. Return a new matrix "
        "rotated 90 degrees clockwise. Return `[]` for an empty matrix. Do not mutate the input.",
        (
            ContractCase("rectangular", ([[1, 2, 3], [4, 5, 6]],), [[4, 1], [5, 2], [6, 3]]),
            ContractCase("one_row", ([[1, 2, 3]],), [[1], [2], [3]]),
            ContractCase("one_column", ([[1], [2], [3]],), [[3, 2, 1]]),
            ContractCase("empty", ([],), []),
        ),
    ),
    FunctionRepairTask(
        "train.valid_brackets.v1", "train",
        "Implement `solve(text)`. Return `True` exactly when `text` contains only the six bracket characters "
        "`()[]{}` and the brackets are properly balanced and nested. Return `False` for any other character, "
        "mismatched order, or unfinished bracket. The empty string is valid.",
        (
            ContractCase("empty", ("",), True),
            ContractCase("nested", ("([]{})",), True),
            ContractCase("mismatch", ("([)]",), False),
            ContractCase("unfinished", ("(((",), False),
            ContractCase("other_character", ("a",), False),
            ContractCase("extra_closer", ("[]]",), False),
        ),
    ),
)


DEVELOPMENT_TASKS = (
    FunctionRepairTask(
        "development.longest_common_prefix.v1", "development",
        "Implement `solve(words)`. Return the longest common prefix shared by every string. Return `\"\"` for an "
        "empty list or when no nonempty prefix is shared. Compare Unicode code points and preserve the original "
        "characters; matching is case-sensitive.",
        (
            ContractCase("empty", ([],), ""),
            ContractCase("shared", (["flower", "flow", "flight"],), "fl"),
            ContractCase("none", (["dog", "racecar", "car"],), ""),
            ContractCase("unicode", (["Ångström", "Ångela"],), "Ång"),
        ),
    ),
    FunctionRepairTask(
        "development.maximum_subarray.v1", "development",
        "Implement `solve(values)`. For a nonempty integer list, return `[sum, start, end]` for a contiguous "
        "nonempty subarray with maximum sum; `end` is exclusive. If several subarrays have the same maximum sum, "
        "choose the lexicographically smallest `(start, end)`. Return `None` for an empty list.",
        (
            ContractCase("mixed", ([-2, 1, -3, 4, -1, 2, 1, -5, 4],), [6, 3, 7]),
            ContractCase("all_negative", ([-5, -2, -2, -7],), [-2, 1, 2]),
            ContractCase("tie_order", ([1, -1, 1],), [1, 0, 1]),
            ContractCase("empty", ([],), None),
        ),
    ),
    FunctionRepairTask(
        "development.stable_anagram_groups.v1", "development",
        "Implement `solve(words)`. Group case-sensitive strings by anagram. Order groups by the first occurrence "
        "of any member in the input, and preserve input order within each group. Return a list of string lists; "
        "return `[]` for empty input.",
        (
            ContractCase("standard", (["eat", "tea", "tan", "ate", "nat", "bat"],),
                         [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]),
            ContractCase("case_sensitive", (["ab", "ba", "ab", "aB", "Ba"],),
                         [["ab", "ba", "ab"], ["aB", "Ba"]]),
            ContractCase("duplicates", (["x", "x", "y"],), [["x", "x"], ["y"]]),
            ContractCase("empty", ([],), []),
        ),
    ),
    FunctionRepairTask(
        "development.decode_rle.v1", "development",
        "Implement `solve(pairs)`. Each pair should contain a nonnegative integer count and a string. Return the "
        "concatenation of each string repeated by its count. Return `None` if a pair is malformed, a count is not "
        "an integer (booleans do not count as integers), a count is negative, or a value is not a string.",
        (
            ContractCase("ordinary", ([[3, "ab"], [1, "c"]],), "abababc"),
            ContractCase("zero", ([[0, "x"], [2, "y"]],), "yy"),
            ContractCase("empty", ([],), ""),
            ContractCase("negative", ([[-1, "x"]],), None),
            ContractCase("boolean_count", ([[True, "x"]],), None),
            ContractCase("malformed", ([[2]],), None),
        ),
    ),
)


def all_tasks() -> tuple[FunctionRepairTask, ...]:
    return TRAIN_TASKS + DEVELOPMENT_TASKS
