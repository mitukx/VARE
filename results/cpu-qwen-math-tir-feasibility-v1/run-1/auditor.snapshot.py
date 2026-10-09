#!/usr/bin/env python3
"""Independently validate the generated task inventory, tool records, and feasibility decision."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/cpu_qwen_math_tir_feasibility_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_qwen_math_tir_feasibility_v1.lock.json"
DATA_PATH = ROOT / "data/cpu-qwen-math-tir-feasibility-v1/pilot.jsonl"
MODEL_DIR = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/"
             "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2")
BOX = re.compile(r"\\boxed\s*\{\s*(-?\d+)\s*\}")
CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def locked_protocol():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock_digest = lock.pop("sha256", None)
    expected = hashlib.sha256(canonical(spec)).hexdigest()
    if lock_digest != expected or canonical(lock) != canonical(spec):
        raise ValueError("protocol and lock do not match")
    for key in ("runner", "auditor", "generator", "objective_test"):
        if sha(ROOT / spec["freeze"][key]) != spec["freeze"][key + "_sha256"]:
            raise ValueError("frozen file hash differs: " + key)
    return spec, expected


def regenerate_independently():
    rng = random.Random(20261009)
    per_family = 24
    out = []
    for index in range(per_family):
        stock, loads, units, damaged, shipped = (rng.randint(120, 640), rng.randint(3, 28),
            rng.randint(9, 46), rng.randint(2, 35), rng.randint(25, 240))
        out.append({"task_id": f"inventory-{index:03d}", "family": "inventory",
            "prompt": f"A warehouse had {stock} boxes. It received {loads} pallets with {units} boxes each, removed {damaged} damaged boxes, and shipped {shipped} boxes. How many boxes remain?",
            "gold_answer": stock + loads * units - damaged - shipped})
        adults, ap, students = rng.randint(18, 180), rng.randint(12, 55), rng.randint(20, 240)
        sp, refund = rng.randint(5, ap - 1), rng.randint(10, 900)
        out.append({"task_id": f"event-{index:03d}", "family": "event_revenue",
            "prompt": f"At a school event, {adults} adult tickets were sold for {ap} dollars each, and {students} student tickets were sold for {sp} dollars each. The organizer later refunded {refund} dollars. How many dollars were kept?",
            "gold_answer": adults * ap + students * sp - refund})
        machines, rate, hours = rng.randint(3, 18), rng.randint(14, 95), rng.randint(3, 16)
        rejects = rng.randint(4, machines * rate * hours // 5)
        out.append({"task_id": f"factory-{index:03d}", "family": "factory_output",
            "prompt": f"A factory has {machines} machines. Each machine makes {rate} parts per hour. They run for {hours} hours, and quality control rejects {rejects} parts. How many good parts remain?",
            "gold_answer": machines * rate * hours - rejects})
        cartons, books, classes = rng.randint(8, 35), rng.randint(14, 48), rng.randint(4, 12)
        loose = (rng.randint(0, classes - 1) - cartons * books) % classes
        donation = rng.randint(1, (cartons * books + loose) // classes - 1) * classes
        total = cartons * books + loose
        out.append({"task_id": f"classroom-{index:03d}", "family": "equal_distribution",
            "prompt": f"A school has {cartons} cartons with {books} books each, plus {loose} loose books. It donates {donation} books. The remaining books are shared equally among {classes} classes. How many books does each class receive?",
            "gold_answer": (total - donation) // classes})
    return out


def safe_integer_expression(text: str):
    if not isinstance(text, str) or not text.strip() or len(text) > 160:
        raise ValueError("invalid expression length")
    tree = ast.parse(text, mode="eval")
    binary = {ast.Add: lambda a,b:a+b, ast.Sub:lambda a,b:a-b, ast.Mult:lambda a,b:a*b,
              ast.FloorDiv:lambda a,b:a//b, ast.Mod:lambda a,b:a%b, ast.Pow:lambda a,b:a**b}
    def calc(node, depth=0):
        if depth > 24:
            raise ValueError("expression too deep")
        if isinstance(node, ast.Expression):
            value = calc(node.body, depth+1)
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = calc(node.operand, depth+1); value = v if isinstance(node.op, ast.UAdd) else -v
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = calc(node.left, depth+1), calc(node.right, depth+1)
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and right == 0:
                raise ValueError("division by zero")
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("exponent too large")
            value = binary[type(node.op)](left, right)
        else:
            raise ValueError("unsupported expression node")
        if abs(value) > 10**15:
            raise ValueError("result out of range")
        return value
    answer = calc(tree)
    if type(answer) is not int:
        raise ValueError("not integer")
    return answer


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [center-half, center+half]


def independently_parse(record):
    if not record["turns"]:
        return None
    matches = BOX.findall(record["turns"][-1]["text"])
    return int(matches[-1]) if matches else None


def audit(bundle: Path):
    spec, protocol_hash = locked_protocol()
    bundle = bundle.resolve()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel, digest in manifest["files"].items():
        if sha(bundle / rel) != digest:
            raise ValueError("bundle file hash mismatch: " + rel)
    if json.loads((bundle / "protocol.snapshot.json").read_text()) != spec:
        raise ValueError("protocol snapshot mismatch")
    if json.loads((bundle / "protocol.lock.snapshot.json").read_text()) != json.loads(LOCK_PATH.read_text()):
        raise ValueError("lock snapshot mismatch")
    if sha(DATA_PATH) != spec["dataset"]["pilot_sha256"]:
        raise ValueError("source dataset hash mismatch")
    for name, expected in spec["model"]["files_sha256"].items():
        if sha(MODEL_DIR / name) != expected:
            raise ValueError("pinned model asset mismatch: " + name)
    if (bundle / "failure.json").exists():
        failure = json.loads((bundle / "failure.json").read_text())
        report = {"status": "incomplete", "failure_type": failure["error_type"],
                  "message": failure["message"], "protocol_sha256": protocol_hash,
                  "scope": "No complete base-task result was produced; this candidate cannot advance."}
        (bundle / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
        return report
    rows = regenerate_independently()
    stored = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines()]
    if rows != stored:
        raise ValueError("independently generated rows differ from frozen task data")
    records = [json.loads(line) for line in (bundle / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != len(rows) or len({r["task_id"] for r in records}) != len(rows):
        raise ValueError("record count or uniqueness mismatch")
    family = {}
    verified_calls = invalid_calls = 0
    for expected, record in zip(rows, records):
        for key in ("task_id", "family", "prompt", "gold_answer"):
            if expected[key] != record[key]:
                raise ValueError("record task provenance mismatch: " + key)
        if record["final_answer"] != independently_parse(record):
            raise ValueError("boxed answer parser mismatch")
        calls_from_text = []
        for turn in record["turns"]:
            match = CALL.search(turn["text"])
            if match:
                call = json.loads(match.group(1))
                if call.get("name") != "calculator" or not isinstance(call.get("arguments"), dict):
                    raise ValueError("unrecognized calculator call in model output")
                calls_from_text.append(call["arguments"].get("expression"))
        if len(calls_from_text) != len(record["tool_calls"]):
            raise ValueError("tool call record count differs from raw output")
        for call_index, (expression, call) in enumerate(zip(calls_from_text, record["tool_calls"])):
            if expression != call.get("expression"):
                raise ValueError("tool expression differs from raw output")
            if call.get("status") == "excess_call":
                if call_index < spec["task"]["maximum_tool_calls"]:
                    raise ValueError("tool call was marked excess below the frozen limit")
                invalid_calls += 1
                continue
            if call.get("status") not in ("ok", "error", "excess_call"):
                raise ValueError("unknown calculator call status")
            try:
                result = safe_integer_expression(expression)
                if call.get("status") != "ok" or call.get("result") != result:
                    raise ValueError("calculator result mismatch")
                verified_calls += 1
            except (ValueError, SyntaxError, TypeError) as exc:
                if call.get("status") == "ok":
                    raise ValueError("unsafe expression marked valid") from exc
                invalid_calls += 1
        correct = record["final_answer"] == expected["gold_answer"]
        if bool(record["exact_match"]) != correct:
            raise ValueError("exact-match flag mismatch")
        family.setdefault(expected["family"], []).append(correct)
    k, n = sum(r["exact_match"] for r in records), len(records)
    acc = k/n
    per_family = {name:{"n":len(values),"exact_match":sum(values)/len(values)} for name,values in sorted(family.items())}
    gates=spec["metrics"]["feasibility_gate"]
    summary=json.loads((bundle/"summary.json").read_text(encoding="utf-8"))
    limits=spec["compute_limits"]
    decision={"minimum_accuracy_pass":acc>=gates["minimum_accuracy"],
        "maximum_accuracy_headroom_pass":acc<=gates["maximum_accuracy"],
        "each_family_minimum_pass":all(x["exact_match"]>=gates["minimum_accuracy_per_family"] for x in per_family.values()),
        "resource_pass":summary.get("peak_rss_bytes",limits["max_peak_rss_bytes"]+1)<=limits["max_peak_rss_bytes"] and
                        summary.get("wall_seconds",limits["max_wall_seconds"]+1)<=limits["max_wall_seconds"]}
    decision["pass"]=all(decision.values())
    if summary.get("protocol_sha256")!=protocol_hash or summary.get("status")!="completed":
        raise ValueError("summary protocol or status mismatch")
    checks=(summary["primary"]["successes"]==k and abs(summary["primary"]["exact_match"]-acc)<1e-12 and
        summary["per_family"]==per_family and summary["decision"]==decision and
        all(abs(a-b)<1e-12 for a,b in zip(summary["primary"]["wilson_95"],wilson(k,n))) and
        summary["tool_use"]["calls"]==verified_calls+invalid_calls and
        summary["tool_use"]["valid_calls"]==verified_calls and
        summary["tool_use"]["invalid_or_excess_calls"]==invalid_calls and
        summary["generation"]["total_tokens"]==sum(r["generated_tokens"] for r in records) and
        abs(summary["generation"]["total_seconds"]-sum(r["generation_seconds"] for r in records))<0.01)
    if not checks:
        raise ValueError("summary does not match independent record/grader reconstruction")
    report={"status":"pass","protocol_sha256":protocol_hash,"manifest_sha256":sha(manifest_path),
        "task_regeneration":"pass","tool_safety_replay":"pass","answer_and_metric_replay":"pass",
        "decision_replay":"pass","decision":decision,"exact_match":acc,"wilson_95":wilson(k,n),
        "scope":"Same-host offline replay of generated tasks and retained records; model generation was not repeated; not external reproduction."}
    (bundle/"audit.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("bundle",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.bundle),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
