#!/usr/bin/env python3
"""Generate the frozen, train-only arithmetic feasibility prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/cpu-arithmetic-feasibility-v1/pilot.json"
COUNT_PER_COMPOSITION = 16
COMPOSITIONS = (
    "left_nested_add_mul_sub",
    "sum_of_products",
    "difference_times_sum",
    "sum_times_difference",
)


def operand(pilot_index: int, position: int) -> int:
    payload = ("vare-arithmetic-feasibility-v1\0%d\0%d\0%d" %
               (pilot_index, position, 20261008)).encode("ascii")
    return 2 + int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % 49


def build_rows() -> list[dict[str, Any]]:
    rows = []
    for composition_index, composition in enumerate(COMPOSITIONS):
        for local_index in range(COUNT_PER_COMPOSITION):
            row_index = composition_index * COUNT_PER_COMPOSITION + local_index
            values = [operand(row_index, i) for i in range(4)]
            a, b, c, d = values
            if composition == "left_nested_add_mul_sub":
                expression, answer = "((%d + %d) * %d) - %d" % (a, b, c, d), (a + b) * c - d
            elif composition == "sum_of_products":
                expression, answer = "(%d * %d) + (%d * %d)" % (a, b, c, d), a * b + c * d
            elif composition == "difference_times_sum":
                expression, answer = "(%d - %d) * (%d + %d)" % (a, b, c, d), (a - b) * (c + d)
            elif composition == "sum_times_difference":
                expression, answer = "(%d + %d) * (%d - %d)" % (a, b, c, d), (a + b) * (c - d)
            else:
                raise AssertionError("unrecognized composition")
            prompt = (
                "Evaluate the fully parenthesized integer expression. "
                "Output exactly one line in this format: FINAL: <integer>\n"
                "Expression: " + expression
            )
            rows.append({"pilot_index": row_index, "composition": composition,
                         "operands": values, "expression": expression,
                         "prompt": prompt, "oracle_answer": answer})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    path = args.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(build_rows(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError("refusing to overwrite non-identical pilot data")
    else:
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"path": str(path), "rows": COUNT_PER_COMPOSITION * len(COMPOSITIONS),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
