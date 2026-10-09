#!/usr/bin/env python3
"""Frozen CPU-only base-feasibility screen for math TIR with a restricted calculator."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import resource
import signal
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/cpu_qwen_math_tir_feasibility_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_qwen_math_tir_feasibility_v1.lock.json"
DATA_PATH = ROOT / "data/cpu-qwen-math-tir-feasibility-v1/pilot.jsonl"
GENERATOR_PATH = ROOT / "scripts/generate_qwen_math_tir_feasibility_data_v1.py"
MODEL_DIR = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/"
             "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2")
OUTPUT = ROOT / "results/cpu-qwen-math-tir-feasibility-v1/run-1"
BOX_RE = re.compile(r"\\boxed\s*\{\s*(-?\d+)\s*\}")
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOLS = [{"type": "function", "function": {"name": "calculator",
    "description": "Evaluate one arithmetic expression with integer operations.",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                    "required": ["expression"], "additionalProperties": False}}}]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_spec():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked = dict(lock)
    expected = locked.pop("sha256", None)
    observed = hashlib.sha256(canonical(spec)).hexdigest()
    if expected != observed or canonical(locked) != canonical(spec):
        raise ValueError("frozen feasibility protocol differs from lock")
    for key, path_key in (("runner_sha256", "runner"), ("auditor_sha256", "auditor"),
                          ("generator_sha256", "generator"), ("objective_test_sha256", "objective_test")):
        path = ROOT / spec["freeze"][path_key]
        if sha256_file(path) != spec["freeze"][key]:
            raise ValueError("frozen source digest differs: " + str(path))
    return spec, observed


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class ResourceGuard:
    def __init__(self, seconds: int, memory: int):
        self.seconds, self.memory = seconds, memory
        self.stop = threading.Event()

    def __enter__(self):
        def abort(signum, _frame):
            raise RuntimeError("resource guard stopped screen (%s)" % signum)
        self.old_alarm = signal.signal(signal.SIGALRM, abort)
        self.old_usr1 = signal.signal(signal.SIGUSR1, abort)
        signal.alarm(self.seconds)
        def watch():
            while not self.stop.wait(0.25):
                if rss_bytes() > self.memory:
                    signal.raise_signal(signal.SIGUSR1)
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join(timeout=1)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old_alarm)
        signal.signal(signal.SIGUSR1, self.old_usr1)
        return False


def safe_integer_expression(expression: str, max_chars: int = 160) -> int:
    if not isinstance(expression, str) or not expression.strip() or len(expression) > max_chars:
        raise ValueError("calculator expression has invalid length")
    root = ast.parse(expression, mode="eval")
    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow)
    def visit(node, depth=0):
        if depth > 24:
            raise ValueError("calculator expression is too deeply nested")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            inner = visit(node.operand, depth + 1)
            value = inner if isinstance(node.op, ast.UAdd) else -inner
        elif isinstance(node, ast.BinOp) and isinstance(node.op, allowed_binops):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add): value = left + right
            elif isinstance(node.op, ast.Sub): value = left - right
            elif isinstance(node.op, ast.Mult): value = left * right
            elif isinstance(node.op, ast.FloorDiv):
                if right == 0: raise ValueError("division by zero")
                value = left // right
            elif isinstance(node.op, ast.Mod):
                if right == 0: raise ValueError("modulo by zero")
                value = left % right
            else:
                if abs(right) > 8: raise ValueError("exponent outside calculator limit")
                value = left ** right
        else:
            raise ValueError("calculator permits integer arithmetic syntax only")
        if abs(value) > 10**15:
            raise ValueError("calculator result exceeds range")
        return value
    result = visit(root)
    if type(result) is not int:
        raise ValueError("calculator result is not an integer")
    return result


def parse_tool_call(text: str):
    match = TOOL_RE.search(text)
    if not match:
        return None
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict) or payload.get("name") != "calculator":
        raise ValueError("unsupported tool call")
    args = payload.get("arguments")
    if not isinstance(args, dict) or set(args) != {"expression"}:
        raise ValueError("invalid calculator arguments")
    return args["expression"]


def parse_boxed_integer(text: str):
    matches = BOX_RE.findall(text)
    return int(matches[-1]) if matches else None


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054):
    p = successes / count
    den = 1.0 + z * z / count
    center = (p + z * z / (2 * count)) / den
    half = z * ((p * (1 - p) / count + z * z / (4 * count * count)) ** 0.5) / den
    return [center - half, center + half]


def verify_runtime_and_assets(spec):
    import torch
    import transformers
    runtime = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0],
               "transformers": transformers.__version__, "platform": platform.platform()}
    expected = spec["runtime"]
    if runtime["python"] != expected["python"] or runtime["torch"] != expected["torch"] or runtime["transformers"] != expected["transformers"]:
        raise RuntimeError("runtime differs from frozen protocol: %r" % runtime)
    for name, expected_hash in spec["model"]["files_sha256"].items():
        path = MODEL_DIR / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError("cached model asset missing or changed: " + name)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized in CPU-only screen")
    torch.set_num_threads(spec["compute_limits"]["threads"])
    return runtime


def generate_one(model, tokenizer, prompt, spec, torch):
    messages = [{"role": "system", "content": spec["task"]["system_prompt"]},
                {"role": "user", "content": prompt}]
    turns, tool_records, tokens_total = [], [], 0
    answer = None
    for turn in range(spec["task"]["maximum_tool_calls"] + 1):
        encoded = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=True,
            add_generation_prompt=True, return_tensors="pt")
        if encoded.shape[-1] > spec["compute_limits"]["max_input_tokens"]:
            raise ValueError("frozen prompt exceeded max input tokens")
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(input_ids=encoded.to("cpu"), max_new_tokens=spec["generation"]["max_new_tokens_per_turn"],
                do_sample=False, use_cache=True, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id)
        elapsed = time.monotonic() - started
        generated = output[0, encoded.shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=False)
        if text.endswith(tokenizer.eos_token):
            text = text[:-len(tokenizer.eos_token)]
        tokens_total += int(generated.numel())
        turns.append({"text": text, "generated_tokens": int(generated.numel()), "generation_seconds": round(elapsed, 6)})
        expression = parse_tool_call(text)
        if expression is not None:
            if turn >= spec["task"]["maximum_tool_calls"]:
                tool_records.append({"expression": expression, "status": "excess_call"})
                break
            try:
                result = safe_integer_expression(expression)
                tool_records.append({"expression": expression, "result": result, "status": "ok"})
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "tool", "content": str(result)})
                continue
            except Exception as exc:
                tool_records.append({"expression": expression, "status": "error", "error_type": type(exc).__name__})
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "tool", "content": "ERROR: invalid integer arithmetic expression"})
                continue
        answer = parse_boxed_integer(text)
        break
    return {"turns": turns, "tool_calls": tool_records, "final_answer": answer,
            "generated_tokens": tokens_total,
            "generation_seconds": round(sum(t["generation_seconds"] for t in turns), 6)}


def write_manifest(output: Path):
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in ("manifest.json", "audit.json"):
            files[path.relative_to(output).as_posix()] = sha256_file(path)
    write_json(output / "manifest.json", {"algorithm": "sha256", "files": files})


def run(output: Path):
    spec, protocol_hash = load_spec()
    output = output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite feasibility output: " + str(output))
    output.mkdir(parents=True)
    (output / "protocol.snapshot.json").write_bytes(SPEC_PATH.read_bytes())
    (output / "protocol.lock.snapshot.json").write_bytes(LOCK_PATH.read_bytes())
    (output / "generator.snapshot.py").write_bytes(GENERATOR_PATH.read_bytes())
    (output / "runner.snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "auditor.snapshot.py").write_bytes((ROOT / spec["freeze"]["auditor"]).read_bytes())
    (output / "objective_test.snapshot.py").write_bytes((ROOT / spec["freeze"]["objective_test"]).read_bytes())
    (output / "pilot.jsonl").write_bytes(DATA_PATH.read_bytes())
    started = time.monotonic()
    try:
        if sha256_file(DATA_PATH) != spec["dataset"]["pilot_sha256"]:
            raise ValueError("frozen generated data digest differs")
        runtime = verify_runtime_and_assets(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
        if not tokenizer.is_fast:
            raise RuntimeError("pinned fast tokenizer required")
        guard = spec["compute_limits"]
        with ResourceGuard(guard["max_wall_seconds"], guard["max_peak_rss_bytes"]):
            model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(torch.device("cpu")).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if next(model.parameters()).device.type != "cpu" or next(model.parameters()).dtype != torch.bfloat16:
                raise RuntimeError("model did not load as frozen CPU bfloat16")
            rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines()]
            records, family = [], {}
            for row in rows:
                result = generate_one(model, tokenizer, row["prompt"], spec, torch)
                correct = result["final_answer"] == row["gold_answer"]
                family.setdefault(row["family"], []).append(bool(correct))
                records.append({**row, **result, "exact_match": bool(correct)})
                print("scored", row["task_id"], "correct" if correct else "incorrect",
                      "tokens", result["generated_tokens"], flush=True)
            (output / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n" for r in records), encoding="utf-8")
            successes = sum(r["exact_match"] for r in records)
            accuracy = successes / len(records)
            per_family = {name: {"n": len(values), "exact_match": sum(values) / len(values)} for name, values in sorted(family.items())}
            gates = spec["metrics"]["feasibility_gate"]
            tool_calls = [call for record in records for call in record["tool_calls"]]
            resource_ok = rss_bytes() <= guard["max_peak_rss_bytes"] and time.monotonic()-started <= guard["max_wall_seconds"]
            decision = {"minimum_accuracy_pass": accuracy >= gates["minimum_accuracy"],
                "maximum_accuracy_headroom_pass": accuracy <= gates["maximum_accuracy"],
                "each_family_minimum_pass": all(x["exact_match"] >= gates["minimum_accuracy_per_family"] for x in per_family.values()),
                "resource_pass": resource_ok}
            decision["pass"] = all(decision.values())
            summary = {"protocol_sha256": protocol_hash, "status": "completed", "decision": decision,
                "model": spec["model"], "dataset_sha256": spec["dataset"]["pilot_sha256"],
                "n": len(records), "primary": {"successes": successes, "exact_match": accuracy,
                    "wilson_95": wilson_interval(successes, len(records))},
                "per_family": per_family,
                "tool_use": {"calls": len(tool_calls), "valid_calls": sum(x["status"] == "ok" for x in tool_calls),
                    "invalid_or_excess_calls": sum(x["status"] != "ok" for x in tool_calls)},
                "generation": {"total_tokens": sum(x["generated_tokens"] for x in records),
                    "total_seconds": round(sum(x["generation_seconds"] for x in records), 3),
                    "tokens_per_second": round(sum(x["generated_tokens"] for x in records) / max(sum(x["generation_seconds"] for x in records), 1e-9), 4)},
                "runtime": runtime, "wall_seconds": round(time.monotonic()-started, 3), "peak_rss_bytes": rss_bytes(),
                "scope": "No-update CPU feasibility on a newly generated procedural arithmetic task; not a public benchmark or capability claim."}
            write_json(output / "summary.json", summary)
            del model
        write_manifest(output)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary
    except Exception as exc:
        write_json(output / "failure.json", {"status": "incomplete", "error_type": type(exc).__name__,
            "message": str(exc), "wall_seconds": round(time.monotonic()-started, 3), "peak_rss_bytes": rss_bytes()})
        write_manifest(output)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
