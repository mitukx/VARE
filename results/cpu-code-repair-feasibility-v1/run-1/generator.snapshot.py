#!/usr/bin/env python3
"""Generate a frozen 32-episode code-repair feasibility screen."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/cpu-code-repair-feasibility-v1/pilot.jsonl"
SEED = 20261010
PER_TEMPLATE = 4


FUNCTIONS = {
    "minimum_inclusive": ("can_commit", "value", "minimum",
        "def {fn}({a}, {b}):\n    return {a} > {b}\n",
        "Return true when the value is at least the minimum, including equality."),
    "half_open_range": ("is_in_window", "point", "start",
        "def {fn}({a}, {b}, stop):\n    return {b} < {a} <= stop\n",
        "Return true for a half-open interval: include the lower bound and exclude the upper bound."),
    "keep_nonnegative": ("retain_values", "items", "unused",
        "def {fn}({a}):\n    return [item for item in {a} if item > 0]\n",
        "Return all nonnegative values in their original order, including repeats and zero."),
    "keep_half_open_range": ("select_window", "items", "lower",
        "def {fn}({a}, lower, upper):\n    return [item for item in {a} if lower < item <= upper]\n",
        "Keep values in [lower, upper), preserving order and duplicates."),
    "mean_or_fallback": ("average_or_default", "observations", "fallback",
        "def {fn}({a}, {b}):\n    return sum({a}) / len({a}) if {a} else 0\n",
        "Return the arithmetic mean, or the supplied fallback when the collection is empty."),
    "minimum_or_fallback": ("smallest_or_default", "observations", "fallback",
        "def {fn}({a}, {b}):\n    return min({a}) if {a} else 0\n",
        "Return the minimum value, or the supplied fallback when the collection is empty."),
    "discount_then_cap": ("final_price", "price", "discount",
        "def {fn}({a}, {b}, cap):\n    return min({a}, cap) - {b}\n",
        "Subtract the discount first, then cap the resulting price at the maximum."),
    "scale_then_cap": ("scaled_limit", "value", "scale",
        "def {fn}({a}, {b}, cap):\n    return min({a}, cap) * {b}\n",
        "Multiply the value first, then cap the resulting value at the maximum."),
}


def make_cases(kind: str, rng: random.Random):
    if kind == "minimum_inclusive":
        minimum = rng.randint(4, 30)
        hidden = [{"value": minimum, "minimum": minimum}]
        hidden += [{"value": rng.randint(0, 60), "minimum": rng.randint(1, 40)} for _ in range(7)]
        return ([{"value": x, "minimum": minimum} for x in (minimum - 1, minimum, minimum + 1)], hidden)
    if kind == "half_open_range":
        start = rng.randint(2, 15); stop = start + rng.randint(3, 12)
        hidden = [{"point": start, "start": start, "stop": stop},
                  {"point": stop, "start": start, "stop": stop}]
        hidden += [{"point": rng.randint(0, 40), "start": a, "stop": a + b}
                   for a, b in ((rng.randint(0, 25), rng.randint(2, 14)) for _ in range(6))]
        return ([{"point": x, "start": start, "stop": stop} for x in (start - 1, start, stop, stop + 1)], hidden)
    if kind == "keep_nonnegative":
        visible = [{"items": [-2, 0, 3, 0, -1]}, {"items": [0]}, {"items": [-4, -1, 2]}]
        hidden = [{"items": [0]}]
        hidden += [{"items": [rng.randint(-9, 9) for _ in range(rng.randint(1, 9))]} for _ in range(7)]
        return visible, hidden
    if kind == "keep_half_open_range":
        lower = rng.randint(-4, 3); upper = lower + rng.randint(3, 9)
        a, b = rng.randint(-10, 8), rng.randint(2, 12)
        hidden = [{"items": [a, a + b], "lower": a, "upper": a + b}]
        hidden += [{"items": [rng.randint(-15, 20) for _ in range(8)],
                    "lower": a, "upper": a + b}
                   for a, b in ((rng.randint(-10, 8), rng.randint(2, 12)) for _ in range(7))]
        return ([{"items": [lower - 1, lower, lower + 1, upper - 1, upper], "lower": lower, "upper": upper}], hidden)
    if kind == "mean_or_fallback":
        visible = [{"observations": [3, 6, 9], "fallback": 41},
                   {"observations": [], "fallback": -7},
                   {"observations": [2], "fallback": 13}]
        hidden = [{"observations": [], "fallback": rng.randint(-30, 30)}]
        hidden += [{"observations": [rng.randint(-20, 40) for _ in range(rng.randint(0, 8))],
                    "fallback": rng.randint(-30, 30)} for _ in range(7)]
        return visible, hidden
    if kind == "minimum_or_fallback":
        visible = [{"observations": [3, 6, 9], "fallback": 41},
                   {"observations": [], "fallback": -7},
                   {"observations": [-4, 2], "fallback": 13}]
        hidden = [{"observations": [], "fallback": rng.randint(-30, 30)}]
        hidden += [{"observations": [rng.randint(-20, 40) for _ in range(rng.randint(0, 8))],
                    "fallback": rng.randint(-30, 30)} for _ in range(7)]
        return visible, hidden
    if kind == "discount_then_cap":
        price = rng.randint(30, 140); discount = rng.randint(3, 35); cap = rng.randint(20, 120)
        hidden = [{"price": cap + 10, "discount": 1, "cap": cap}]
        hidden += [{"price": rng.randint(10, 200), "discount": rng.randint(0, 50),
                    "cap": rng.randint(10, 180)} for _ in range(7)]
        return ([{"price": price, "discount": discount, "cap": cap}], hidden)
    if kind == "scale_then_cap":
        value = rng.randint(2, 30); scale = rng.randint(2, 5); cap = rng.randint(8, 90)
        hidden_cap = rng.randint(5, 20)
        hidden = [{"value": hidden_cap + 1, "scale": 2, "cap": hidden_cap}]
        hidden += [{"value": rng.randint(1, 40), "scale": rng.randint(2, 6),
                    "cap": rng.randint(5, 150)} for _ in range(7)]
        return ([{"value": value, "scale": scale, "cap": cap}], hidden)
    raise ValueError(kind)


def generate_rows():
    rng = random.Random(SEED)
    names = ["apply", "resolve", "compute", "process"]
    rows = []
    kinds = list(FUNCTIONS)
    for kind in kinds:
        family = {
            "minimum_inclusive": "boundary", "half_open_range": "boundary",
            "keep_nonnegative": "sequence", "keep_half_open_range": "sequence",
            "mean_or_fallback": "default_aggregation", "minimum_or_fallback": "default_aggregation",
            "discount_then_cap": "ordered_transformation", "scale_then_cap": "ordered_transformation",
        }[kind]
        base_name, arg1, arg2, code, description = FUNCTIONS[kind]
        for variant in range(PER_TEMPLATE):
            task_seed = rng.getrandbits(32)
            local = random.Random(task_seed)
            function_name = f"{base_name}_{names[variant]}"
            source = code.format(fn=function_name, a=arg1, b=arg2)
            visible, hidden = make_cases(kind, local)
            task_id = f"{family}-{kind}-{variant:02d}"
            rows.append({
                "task_id": task_id,
                "task_seed": task_seed,
                "family": family,
                "template": kind,
                "function_name": function_name,
                "prompt": (f"Repair {function_name} in src/solution.py. {description} "
                           "Inspect the source, run the visible cases, make the smallest valid edit, "
                           "and run the visible cases again before finishing."),
                "source": source,
                "visible_inputs": visible,
                "hidden_inputs": hidden,
            })
    rng.shuffle(rows)
    if len(rows) != 32 or len({r["task_id"] for r in rows}) != 32:
        raise ValueError("task inventory is incomplete or duplicated")
    counts = {}
    for row in rows:
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    if counts != {"boundary": 8, "sequence": 8, "default_aggregation": 8,
                  "ordered_transformation": 8}:
        raise ValueError(f"family balance differs: {counts}")
    return rows


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as stream:
        for row in generate_rows():
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"rows": 32, "path": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
