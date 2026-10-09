"""Screen grader: trusted arithmetic oracle plus a restricted source interpreter."""
from __future__ import annotations

from code_repair_sandbox_v1 import evaluate_function


def typed_equal(left, right):
    """Match values recursively, preserving bool/int and list element types."""
    if type(left) is not type(right):
        return False
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def expected(kind: str, inputs: dict):
    if kind == "minimum_inclusive":
        return inputs["value"] >= inputs["minimum"]
    if kind == "half_open_range":
        return inputs["start"] <= inputs["point"] < inputs["stop"]
    if kind == "keep_nonnegative":
        result = []
        for item in inputs["items"]:
            if item >= 0:
                result.append(item)
        return result
    if kind == "keep_half_open_range":
        result = []
        for item in inputs["items"]:
            if inputs["lower"] <= item and item < inputs["upper"]:
                result.append(item)
        return result
    if kind == "mean_or_fallback":
        values = inputs["observations"]
        if len(values) == 0:
            return inputs["fallback"]
        return sum(values) / len(values)
    if kind == "minimum_or_fallback":
        values = inputs["observations"]
        if len(values) == 0:
            return inputs["fallback"]
        return min(values)
    if kind == "discount_then_cap":
        discounted = inputs["price"] - inputs["discount"]
        return min(discounted, inputs["cap"])
    if kind == "scale_then_cap":
        scaled = inputs["value"] * inputs["scale"]
        return min(scaled, inputs["cap"])
    raise ValueError(f"unknown oracle kind: {kind}")


def score_visible(source: str, row: dict) -> dict:
    results = []
    for inputs in row["visible_inputs"]:
        target = expected(row["template"], inputs)
        try:
            observed = evaluate_function(source, row["function_name"], inputs)
            passed = typed_equal(observed, target)
            results.append({"passed": passed, "expected": target, "observed": observed})
        except Exception as exc:
            results.append({"passed": False, "expected": target,
                            "error_type": type(exc).__name__, "message": str(exc)[:160]})
    return {"passed": bool(results) and all(item["passed"] for item in results),
            "passed_cases": sum(item["passed"] for item in results),
            "total_cases": len(results), "cases": results}


def score_hidden(source: str, row: dict) -> dict:
    results = []
    for inputs in row["hidden_inputs"]:
        target = expected(row["template"], inputs)
        try:
            observed = evaluate_function(source, row["function_name"], inputs)
            results.append(typed_equal(observed, target))
        except Exception:
            results.append(False)
    return {"passed": bool(results) and all(results),
            "passed_cases": sum(results), "total_cases": len(results)}
