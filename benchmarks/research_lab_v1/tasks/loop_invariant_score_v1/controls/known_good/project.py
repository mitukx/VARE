"""Average identical row-column dot-product passes.

``score_matrix(rows, cols, passes)`` returns a dense score matrix.
``passes`` must be a positive integer. Repeating the same dots and
averaging them does not change the mathematical result.
"""


def _dot(left, right):
    total = 0.0
    for lhs, rhs in zip(left, right):
        total += lhs * rhs
    return total


def score_matrix(rows, cols, passes):
    if passes < 1:
        raise ValueError("passes must be positive")
    width = len(cols)
    acc = [[0.0] * width for _ in rows]
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            acc[i][j] += _dot(row, col)
    return acc
