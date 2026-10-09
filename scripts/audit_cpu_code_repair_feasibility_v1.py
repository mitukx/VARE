#!/usr/bin/env python3
"""Independent, stdlib-only audit for a frozen code-repair feasibility bundle.

Candidate source is parsed and interpreted by this file; it is never executed.
This implementation intentionally does not import the task generator, grader,
runner, or candidate sandbox.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "protocols/cpu_code_repair_feasibility_v1.json"
LOCK = ROOT / "protocols/cpu_code_repair_feasibility_v1.lock.json"
DATA = ROOT / "data/cpu-code-repair-feasibility-v1/pilot.jsonl"
OPEN_TAG, CLOSE_TAG = "<tool_call>", "</tool_call>"
SEED = 20261010
FAMILY_BY_KIND = {
    "minimum_inclusive": "boundary", "half_open_range": "boundary",
    "keep_nonnegative": "sequence", "keep_half_open_range": "sequence",
    "mean_or_fallback": "default_aggregation", "minimum_or_fallback": "default_aggregation",
    "discount_then_cap": "ordered_transformation", "scale_then_cap": "ordered_transformation",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def extract_calls_independently(text: str) -> list[str]:
    """Independent scanner using JSONDecoder boundaries, not the runner parser."""
    calls = []
    position = 0
    while position < len(text):
        opening = text.find(OPEN_TAG, position)
        if opening == -1: break
        begin = opening + len(OPEN_TAG)
        while begin < len(text) and text[begin] in " \t\r\n": begin += 1
        try:
            _, end = json.JSONDecoder().raw_decode(text, begin)
        except (json.JSONDecodeError, ValueError):
            end_tag = text.find(CLOSE_TAG, begin)
            if end_tag == -1: break
            calls.append(text[begin:end_tag]); position = end_tag + len(CLOSE_TAG); continue
        tail = end
        while tail < len(text) and text[tail] in " \t\r\n": tail += 1
        if text.startswith(CLOSE_TAG, tail):
            calls.append(text[begin:end]); position = tail + len(CLOSE_TAG); continue
        end_tag = text.find(CLOSE_TAG, end)
        if end_tag == -1: break
        calls.append(text[begin:end_tag]); position = end_tag + len(CLOSE_TAG)
    return calls


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def independently_generate():
    """Independent deterministic task/data implementation; no project imports."""
    import random

    kinds = [
        ("minimum_inclusive", "can_commit", "value", "minimum", "boundary",
         "def {fn}({a}, {b}):\n    return {a} > {b}\n",
         "Return true when the value is at least the minimum, including equality."),
        ("half_open_range", "is_in_window", "point", "start", "boundary",
         "def {fn}({fnarg}, {arg2}, stop):\n    return {arg2} < {fnarg} <= stop\n",
         "Return true for a half-open interval: include the lower bound and exclude the upper bound."),
        ("keep_nonnegative", "retain_values", "items", "unused", "sequence",
         "def {fn}({a}):\n    return [item for item in {a} if item > 0]\n",
         "Return all nonnegative values in their original order, including repeats and zero."),
        ("keep_half_open_range", "select_window", "items", "lower", "sequence",
         "def {fn}({a}, lower, upper):\n    return [item for item in {a} if lower < item <= upper]\n",
         "Keep values in [lower, upper), preserving order and duplicates."),
        ("mean_or_fallback", "average_or_default", "observations", "fallback", "default_aggregation",
         "def {fn}({a}, {b}):\n    return sum({a}) / len({a}) if {a} else 0\n",
         "Return the arithmetic mean, or the supplied fallback when the collection is empty."),
        ("minimum_or_fallback", "smallest_or_default", "observations", "fallback", "default_aggregation",
         "def {fn}({a}, {b}):\n    return min({a}) if {a} else 0\n",
         "Return the minimum value, or the supplied fallback when the collection is empty."),
        ("discount_then_cap", "final_price", "price", "discount", "ordered_transformation",
         "def {fn}({a}, {b}, cap):\n    return min({a}, cap) - {b}\n",
         "Subtract the discount first, then cap the resulting price at the maximum."),
        ("scale_then_cap", "scaled_limit", "value", "scale", "ordered_transformation",
         "def {fn}({a}, {b}, cap):\n    return min({a}, cap) * {b}\n",
         "Multiply the value first, then cap the resulting value at the maximum."),
    ]
    rng = random.Random(SEED)
    variants = ("apply", "resolve", "compute", "process")
    rows = []
    for kind, base, arg1, arg2, family, template, prompt in kinds:
        for variant, suffix in enumerate(variants):
            task_seed = rng.getrandbits(32)
            local = random.Random(task_seed)
            fn = f"{base}_{suffix}"
            if kind == "half_open_range":
                source = template.format(fn=fn, fnarg=arg1, arg2=arg2)
            else:
                source = template.format(fn=fn, a=arg1, b=arg2)
            if kind == "minimum_inclusive":
                m = local.randint(4, 30)
                visible = [{"value": x, "minimum": m} for x in (m - 1, m, m + 1)]
                hidden = [{"value": m, "minimum": m}]
                hidden += [{"value": local.randint(0, 60), "minimum": local.randint(1, 40)} for _ in range(7)]
            elif kind == "half_open_range":
                start, stop = local.randint(2, 15), 0
                stop = start + local.randint(3, 12)
                visible = [{"point": x, "start": start, "stop": stop} for x in (start - 1, start, stop, stop + 1)]
                hidden = [{"point": start, "start": start, "stop": stop}, {"point": stop, "start": start, "stop": stop}]
                for _ in range(6):
                    a, b = local.randint(0, 25), local.randint(2, 14)
                    hidden.append({"point": local.randint(0, 40), "start": a, "stop": a + b})
            elif kind == "keep_nonnegative":
                visible = [{"items": [-2, 0, 3, 0, -1]}, {"items": [0]}, {"items": [-4, -1, 2]}]
                hidden = [{"items": [0]}]
                hidden += [{"items": [local.randint(-9, 9) for _ in range(local.randint(1, 9))]} for _ in range(7)]
            elif kind == "keep_half_open_range":
                lower = local.randint(-4, 3); upper = lower + local.randint(3, 9)
                visible = [{"items": [lower - 1, lower, lower + 1, upper - 1, upper], "lower": lower, "upper": upper}]
                a, b = local.randint(-10, 8), local.randint(2, 12)
                hidden = [{"items": [a, a + b], "lower": a, "upper": a + b}]
                for _ in range(7):
                    a, b = local.randint(-10, 8), local.randint(2, 12)
                    hidden.append({"items": [local.randint(-15, 20) for _ in range(8)], "lower": a, "upper": a + b})
            elif kind == "mean_or_fallback":
                visible = [{"observations": [3, 6, 9], "fallback": 41}, {"observations": [], "fallback": -7}, {"observations": [2], "fallback": 13}]
                hidden = [{"observations": [], "fallback": local.randint(-30, 30)}]
                hidden += [{"observations": [local.randint(-20, 40) for _ in range(local.randint(0, 8))], "fallback": local.randint(-30, 30)} for _ in range(7)]
            elif kind == "minimum_or_fallback":
                visible = [{"observations": [3, 6, 9], "fallback": 41}, {"observations": [], "fallback": -7}, {"observations": [-4, 2], "fallback": 13}]
                hidden = [{"observations": [], "fallback": local.randint(-30, 30)}]
                hidden += [{"observations": [local.randint(-20, 40) for _ in range(local.randint(0, 8))], "fallback": local.randint(-30, 30)} for _ in range(7)]
            elif kind == "discount_then_cap":
                price, discount, cap = local.randint(30, 140), local.randint(3, 35), local.randint(20, 120)
                visible = [{"price": price, "discount": discount, "cap": cap}]
                hidden = [{"price": cap + 10, "discount": 1, "cap": cap}]
                hidden += [{"price": local.randint(10, 200), "discount": local.randint(0, 50), "cap": local.randint(10, 180)} for _ in range(7)]
            else:
                value, scale, cap = local.randint(2, 30), local.randint(2, 5), local.randint(8, 90)
                visible = [{"value": value, "scale": scale, "cap": cap}]
                hidden_cap = local.randint(5, 20)
                hidden = [{"value": hidden_cap + 1, "scale": 2, "cap": hidden_cap}]
                hidden += [{"value": local.randint(1, 40), "scale": local.randint(2, 6), "cap": local.randint(5, 150)} for _ in range(7)]
            rows.append({"task_id": f"{family}-{kind}-{variant:02d}", "task_seed": task_seed,
                "family": family, "template": kind, "function_name": fn,
                "prompt": f"Repair {fn} in src/solution.py. {prompt} Inspect the source, run the visible cases, make the smallest valid edit, and run the visible cases again before finishing.",
                "source": source, "visible_inputs": visible, "hidden_inputs": hidden})
    rng.shuffle(rows)
    return rows


def oracle(kind: str, item: dict):
    if kind == "minimum_inclusive": return item["value"] >= item["minimum"]
    if kind == "half_open_range": return item["start"] <= item["point"] < item["stop"]
    if kind == "keep_nonnegative": return [v for v in item["items"] if v >= 0]
    if kind == "keep_half_open_range": return [v for v in item["items"] if item["lower"] <= v < item["upper"]]
    if kind == "mean_or_fallback": return sum(item["observations"]) / len(item["observations"]) if item["observations"] else item["fallback"]
    if kind == "minimum_or_fallback": return min(item["observations"]) if item["observations"] else item["fallback"]
    if kind == "discount_then_cap": return min(item["price"] - item["discount"], item["cap"])
    if kind == "scale_then_cap": return min(item["value"] * item["scale"], item["cap"])
    raise ValueError("unknown task template")


def typed_equal(left, right):
    if type(left) is not type(right): return False
    if type(left) in (list, tuple): return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def safe_value(value, depth=0):
    if depth > 32: raise ValueError("nesting")
    if type(value) in (int, bool, str) or value is None: return value
    if type(value) is float and math.isfinite(value) and abs(value) <= 10**9: return value
    if type(value) is int and abs(value) <= 10**9: return value
    if type(value) in (list, tuple) and len(value) <= 256:
        return type(value)(safe_value(v, depth + 1) for v in value)
    raise ValueError("unsupported value")


def interpret(node, env, depth=0):
    if depth > 32: raise ValueError("depth")
    ev = lambda n: interpret(n, env, depth + 1)
    if isinstance(node, ast.Constant): return safe_value(node.value)
    if isinstance(node, ast.Name): return env[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = [ev(x) for x in node.elts]
        return safe_value(vals if isinstance(node, ast.List) else tuple(vals))
    if isinstance(node, ast.UnaryOp):
        value = ev(node.operand)
        if isinstance(node.op, ast.Not): return not value
        if type(value) not in (int, float): raise ValueError("non-number")
        if isinstance(node.op, ast.UAdd): return safe_value(+value)
        if isinstance(node.op, ast.USub): return safe_value(-value)
    if isinstance(node, ast.BinOp):
        left, right = ev(node.left), ev(node.right)
        if type(left) not in (int, float) or type(right) not in (int, float): raise ValueError("non-number")
        opmap = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right, ast.Mult: lambda: left * right,
                 ast.Div: lambda: left / right, ast.FloorDiv: lambda: left // right, ast.Mod: lambda: left % right}
        if type(node.op) not in opmap: raise ValueError("operator")
        return safe_value(opmap[type(node.op)]())
    if isinstance(node, ast.BoolOp):
        result = True if isinstance(node.op, ast.And) else False
        for child in node.values:
            result = ev(child)
            if isinstance(node.op, ast.And) and not result: return result
            if isinstance(node.op, ast.Or) and result: return result
        return result
    if isinstance(node, ast.Compare):
        left = ev(node.left)
        ops = {ast.Lt: lambda a,b:a < b, ast.LtE: lambda a,b:a <= b, ast.Gt: lambda a,b:a > b,
               ast.GtE: lambda a,b:a >= b, ast.Eq: lambda a,b:a == b, ast.NotEq: lambda a,b:a != b}
        for op, rhs in zip(node.ops, node.comparators):
            right = ev(rhs)
            if type(op) not in ops or not ops[type(op)](left, right): return False
            left = right
        return True
    if isinstance(node, ast.IfExp): return ev(node.body if ev(node.test) else node.orelse)
    if isinstance(node, ast.ListComp):
        if len(node.generators) != 1: raise ValueError("comprehension")
        gen = node.generators[0]
        if gen.is_async or not isinstance(gen.target, ast.Name): raise ValueError("target")
        values = ev(gen.iter)
        if type(values) not in (list, tuple) or len(values) > 256: raise ValueError("source")
        result = []
        for value in values:
            local = dict(env); local[gen.target.id] = value
            if all(interpret(c, local, depth + 1) for c in gen.ifs): result.append(interpret(node.elt, local, depth + 1))
        return safe_value(result)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        args = [ev(x) for x in node.args]
        funcs = {"sum": sum, "len": len, "min": min, "max": max, "sorted": sorted}
        if node.func.id not in funcs: raise ValueError("call")
        return safe_value(funcs[node.func.id](*args))
    raise ValueError("unsupported AST node")


def evaluate(source, fn, inputs):
    if not isinstance(source, str) or len(source) > 4096: raise ValueError("source size")
    tree = ast.parse(source, mode="exec")
    if len(list(ast.walk(tree))) > 256 or len(tree.body) != 1: raise ValueError("module structure")
    fun = tree.body[0]
    if not isinstance(fun, ast.FunctionDef) or fun.name != fn: raise ValueError("function")
    args = fun.args
    if (fun.decorator_list or fun.returns is not None or fun.type_comment is not None or args.vararg or args.kwarg or
        args.kwonlyargs or args.defaults or args.kw_defaults or args.posonlyargs or len(fun.body) != 1 or
        not isinstance(fun.body[0], ast.Return) or fun.body[0].value is None): raise ValueError("function subset")
    names = [arg.arg for arg in args.args]
    if len(set(names)) != len(names) or set(names) != set(inputs): raise ValueError("arguments")
    return safe_value(interpret(fun.body[0].value, {k: safe_value(v) for k,v in inputs.items()}))


def validate_mutants(rows):
    for row in rows:
        if all(typed_equal(evaluate(row["source"], row["function_name"], x), oracle(row["template"], x)) for x in row["hidden_inputs"]):
            raise ValueError("baseline bug is not caught by any hidden case: " + row["task_id"])
        good = known_fix(row)
        if not all(typed_equal(evaluate(good, row["function_name"], x), oracle(row["template"], x)) for x in row["hidden_inputs"]):
            raise ValueError("known fix failed hidden oracle: " + row["task_id"])


def known_fix(row):
    kind, fn = row["template"], row["function_name"]
    code = {
        "minimum_inclusive": f"def {fn}(value, minimum):\n    return value >= minimum\n",
        "half_open_range": f"def {fn}(point, start, stop):\n    return start <= point < stop\n",
        "keep_nonnegative": f"def {fn}(items):\n    return [item for item in items if item >= 0]\n",
        "keep_half_open_range": f"def {fn}(items, lower, upper):\n    return [item for item in items if lower <= item < upper]\n",
        "mean_or_fallback": f"def {fn}(observations, fallback):\n    return sum(observations) / len(observations) if observations else fallback\n",
        "minimum_or_fallback": f"def {fn}(observations, fallback):\n    return min(observations) if observations else fallback\n",
        "discount_then_cap": f"def {fn}(price, discount, cap):\n    return min(price - discount, cap)\n",
        "scale_then_cap": f"def {fn}(value, scale, cap):\n    return min(value * scale, cap)\n",
    }
    return code[kind]


def audit(bundle: Path):
    bundle = bundle.resolve()
    spec, lock = read_json(bundle / "protocol.snapshot.json"), read_json(bundle / "protocol.lock.snapshot.json")
    locked = dict(lock); lock_hash = locked.pop("sha256", None)
    if hashlib.sha256(canonical(spec)).hexdigest() != lock_hash or canonical(locked) != canonical(spec):
        raise ValueError("protocol lock mismatch")
    if spec != read_json(SPEC) or lock != read_json(LOCK): raise ValueError("protocol snapshot differs from repository")
    data_bytes = (bundle / "pilot.jsonl").read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != spec["dataset"]["pilot_sha256"]: raise ValueError("dataset hash mismatch")
    rows = [json.loads(x) for x in data_bytes.decode().splitlines()]
    regenerated = independently_generate()
    if rows != regenerated: raise ValueError("pilot differs from independently regenerated dataset")
    family_counts = {}
    template_counts = {}
    for row in rows:
        family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
        template_counts[row["template"]] = template_counts.get(row["template"], 0) + 1
    if family_counts != spec["dataset"]["family_counts"] or template_counts != spec["dataset"]["template_counts"]:
        raise ValueError("pilot family/template inventory differs from protocol")
    validate_mutants(rows)
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("algorithm") != "sha256": raise ValueError("manifest algorithm")
    found = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file() and p.name not in ("manifest.json", "audit.json")}
    if found != set(manifest.get("files", {})): raise ValueError("manifest inventory mismatch")
    for rel, digest in manifest["files"].items():
        if sha256(bundle / rel) != digest: raise ValueError("manifest hash mismatch: " + rel)
    snapshot_names = {"runner": "runner.snapshot.py", "auditor": "auditor.snapshot.py",
        "generator": "generator.snapshot.py", "grader": "grader.snapshot.py",
        "sandbox": "sandbox.snapshot.py", "objective_test": "objective_test.snapshot.py"}
    for role, snapshot in snapshot_names.items():
        rel = spec["freeze"][role]
        digest = spec["freeze"][role + "_sha256"]
        if sha256(ROOT / rel) != digest or sha256(bundle / snapshot) != digest:
            raise ValueError("frozen source or bundle snapshot mismatch: " + role)
    records = [json.loads(x) for x in (bundle / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 32: raise ValueError("record count")
    by_id = {r["task_id"]: r for r in rows}
    if len({r["task_id"] for r in records}) != 32: raise ValueError("duplicate task record")
    outcome = []
    for rec in records:
        row = by_id[rec["task_id"]]
        if rec["family"] != row["family"] or rec["template"] != row["template"]: raise ValueError("task metadata mismatch")
        if rec["source_before"] != row["source"]: raise ValueError("episode started from another source")
        if rec.get("episode_complete") is not True: raise ValueError("incomplete episode")
        if len(rec["turns"]) != len(rec["raw_outputs"]): raise ValueError("turn/output inventory mismatch")
        attempted = schema_valid = unsafe = malformed = accepted_edits = visible_runs = finishes = generated = 0
        current_source = rec["source_before"]
        for turn, raw_output in zip(rec["turns"], rec["raw_outputs"]):
            if turn["model_output"] != raw_output: raise ValueError("raw output was altered")
            if (turn["input_tokens"] < 1 or turn["input_tokens"] > spec["compute_limits"]["max_input_tokens"] or
                turn["generated_tokens"] < 0 or turn["generated_tokens"] > spec["generation"]["max_new_tokens_per_turn"]):
                raise ValueError("invalid token count")
            generated += turn["generated_tokens"]
            matches = extract_calls_independently(raw_output)
            attempted += raw_output.count(OPEN_TAG)
            malformed += max(0, raw_output.count(OPEN_TAG) - len(matches))
            events = turn["tool_calls"]
            malformed += sum(event.get("valid") is False and event.get("error_type") != "UnclosedToolCall" for event in events)
            unsafe += sum(bool(event.get("unauthorized")) for event in events)
            for event in events:
                if not event.get("name"):
                    continue
                schema_valid += int(event.get("schema_valid") is True)
                unsafe += int(event.get("unsafe") is True)
                name, args = event["name"], event["arguments"]
                result = event["result"]
                if name == "edit_file" and result.get("edited") is True:
                    if args.get("path") != "src/solution.py" or not args.get("old_text") or current_source.count(args["old_text"]) != 1:
                        raise ValueError("accepted edit cannot be replayed exactly")
                    current_source = current_source.replace(args["old_text"], args["new_text"], 1)
                    accepted_edits += 1
                elif name == "run_visible_tests":
                    visible_runs += 1
                    cases = row["visible_inputs"]
                    passed = []
                    for case in cases:
                        try: passed.append(typed_equal(evaluate(current_source, row["function_name"], case), oracle(row["template"], case)))
                        except Exception: passed.append(False)
                    if result.get("passed_cases") != sum(passed) or result.get("total_cases") != len(cases) or result.get("passed") != (bool(passed) and all(passed)):
                        raise ValueError("visible test result does not replay")
                elif name == "finish":
                    finishes += 1
        tm = rec["tool_metrics"]
        if current_source != rec["source_after"]: raise ValueError("accepted tool edits do not reconstruct final source")
        expected_tm = {"attempted": attempted, "schema_valid": schema_valid, "unsafe": unsafe,
            "malformed": malformed, "accepted_edits": accepted_edits, "visible_test_runs": visible_runs}
        for key, value in expected_tm.items():
            if tm[key] != value: raise ValueError("tool metric does not replay: " + key)
        if tm["has_accepted_edit"] != (accepted_edits > 0 and current_source != rec["source_before"]): raise ValueError("edit flag mismatch")
        if tm["has_visible_test_run"] != (visible_runs > 0) or rec["finished"] != (finishes > 0): raise ValueError("episode action flag mismatch")
        if tm["unknown_or_unauthorized"] != unsafe: raise ValueError("unauthorized action counter mismatch")
        if generated != rec["generated_tokens"]: raise ValueError("generated-token total mismatch")
        observed = []
        for case in row["hidden_inputs"]:
            try: observed.append(typed_equal(evaluate(rec["source_after"], row["function_name"], case), oracle(row["template"], case)))
            except Exception: observed.append(False)
        hidden_pass = bool(observed) and all(observed)
        tm = rec["tool_metrics"]
        edit = rec["source_after"] != rec["source_before"] and tm["accepted_edits"] > 0 and tm["has_accepted_edit"] is True
        tested = tm["visible_test_runs"] > 0 and tm["has_visible_test_run"] is True
        success = hidden_pass and edit and tested and rec.get("finished") is True
        if rec["episode_pass"] != success: raise ValueError("episode score mismatch")
        if rec["hidden_grade"] != {"passed": hidden_pass, "passed_cases": sum(observed), "total_cases": len(observed)}:
            raise ValueError("runner grader disagrees with independent grader")
        outcome.append((rec, success))
    summary = read_json(bundle / "summary.json")
    protocol_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if (summary["protocol_sha256"] != protocol_hash or summary["model"] != spec["model"] or
        summary["dataset_sha256"] != spec["dataset"]["pilot_sha256"]):
        raise ValueError("summary provenance differs from protocol")
    successes = sum(ok for _, ok in outcome)
    families = {}
    for rec, ok in outcome: families.setdefault(rec["family"], []).append(ok)
    tool = [rec["tool_metrics"] for rec, _ in outcome]
    attempted = sum(x["attempted"] for x in tool); valid = sum(x["schema_valid"] for x in tool)
    totals = {key: sum(x[key] for x in tool) for key in ("attempted", "schema_valid", "unsafe", "malformed", "accepted_edits", "visible_test_runs")}
    expected_tool_summary = {"attempted": totals["attempted"], "schema_valid": totals["schema_valid"],
        "validity_rate": totals["schema_valid"] / totals["attempted"] if totals["attempted"] else 0.0,
        "malformed": totals["malformed"], "unsafe_attempts": totals["unsafe"],
        "accepted_edits": totals["accepted_edits"], "visible_test_runs": totals["visible_test_runs"],
        "episodes_with_edit_and_test": sum(x["has_accepted_edit"] and x["has_visible_test_run"] for x in tool)}
    if summary["tool_use"] != expected_tool_summary: raise ValueError("summary tool-call counters mismatch")
    token_sum = sum(rec["generated_tokens"] for rec, _ in outcome)
    generation_sum = sum(rec["generation_seconds"] for rec, _ in outcome)
    if summary["generation"]["tokens"] != token_sum or abs(summary["generation"]["seconds"] - generation_sum) > 0.005:
        raise ValueError("summary generation totals mismatch")
    gates = spec["metrics"]["feasibility_gate"]
    episodes_edit_test = sum(x["has_accepted_edit"] and x["has_visible_test_run"] for x in tool)
    templates = {}
    for rec, ok in outcome: templates.setdefault(rec["template"], []).append(ok)
    gates_out = {
        "row_count_pass": len(outcome) == gates["required_rows"],
        "all_episodes_complete_pass": all(r.get("episode_complete") is True for r, _ in outcome),
        "finish_floor_pass": sum(bool(r.get("finished")) for r, _ in outcome) >= gates["minimum_finished_episodes"],
        "accuracy_floor_pass": gates["minimum_successes"] <= successes,
        "headroom_pass": successes <= gates["maximum_successes"],
        "family_floor_pass": len(families) == 4 and all(sum(v) >= gates["minimum_per_family"] for v in families.values()),
        "template_floor_pass": len(templates) == gates["required_templates"] and all(sum(v) >= gates["minimum_per_template"] for v in templates.values()),
        "tool_validity_pass": attempted > 0 and valid / attempted >= gates["minimum_tool_validity_rate"] and episodes_edit_test >= gates["minimum_episodes_with_edit_and_test"],
        "safe_action_pass": sum(x["unsafe"] for x in tool) == 0,
        "resource_pass": summary["peak_rss_bytes"] <= spec["compute_limits"]["max_peak_rss_bytes"] and summary["wall_seconds"] <= spec["compute_limits"]["max_wall_seconds"],
    }
    expected_family = {k: {"n": len(v), "successes": sum(v), "success_rate": sum(v) / len(v)} for k, v in sorted(families.items())}
    expected_template = {k: {"n": len(v), "successes": sum(v), "success_rate": sum(v) / len(v)} for k, v in sorted(templates.items())}
    if (summary["successes"] != successes or summary["n"] != len(outcome) or
        summary["episode_success_rate"] != successes / len(outcome) or
        summary["per_family"] != expected_family or summary["per_template"] != expected_template or
        summary["decision"] != {**gates_out, "pass": all(gates_out.values())}):
        raise ValueError("summary disagrees with independent reconstruction")
    runtime = summary["runtime"]
    if (runtime["python"] != spec["runtime"]["python"] or runtime["torch"] != spec["runtime"]["torch"] or
        runtime["transformers"] != spec["runtime"]["transformers"] or runtime["platform"] != spec["runtime"]["platform"] or
        runtime["device"] != "cpu" or runtime["cuda_initialized"] or runtime["mps_used"]):
        raise ValueError("runtime is not CPU-only")
    return {"status": "audited", "independent_dataset_regeneration": True,
        "independent_ast_interpreter": True, "source_execution": False,
        "episodes": len(outcome), "successes": successes, "decision": gates_out,
        "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
        "inference_replayed": False,
        "limitations": ["This audit checks retained records and independently recomputes outcomes; it does not repeat model inference.",
                        "The generated tasks are a narrow feasibility screen, not an external benchmark."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    result = audit(args.bundle)
    (args.bundle / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
