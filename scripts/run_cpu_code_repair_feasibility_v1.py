#!/usr/bin/env python3
"""Run the frozen, offline CPU base-policy code-repair feasibility screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/cpu_code_repair_feasibility_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_code_repair_feasibility_v1.lock.json"
DATA_PATH = ROOT / "data/cpu-code-repair-feasibility-v1/pilot.jsonl"
MODEL_DIR = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
             "7ae557604adf67be50417f59c2c2f167def9a775")
OUTPUT = ROOT / "results/cpu-code-repair-feasibility-v1/run-1"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
TOOLS = [
    {"type": "function", "function": {"name": "list_files",
     "description": "List task files available in the isolated workspace.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "read_file",
     "description": "Read one allowlisted task file. Paths must exactly match a listed workspace path.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                     "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_text",
     "description": "Search a literal string in visible task files.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                     "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "edit_file",
     "description": "Replace one exact unique text span in src/solution.py. Provide old_text and new_text.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                     "old_text": {"type": "string"}, "new_text": {"type": "string"}},
                     "required": ["path", "old_text", "new_text"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "run_visible_tests",
     "description": "Run only the public cases in tests/visible_cases.json using the restricted interpreter.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "finish",
     "description": "Finish this repair attempt and summarize the change.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                     "required": ["summary"], "additionalProperties": False}}},
]
TOOL_SCHEMAS = {item["function"]["name"]: set(item["function"]["parameters"].get("required", [])) for item in TOOLS}
TOOL_SCHEMAS["list_files"] = set()
TOOL_SCHEMAS["run_visible_tests"] = set()
TOOL_ARGUMENT_TYPES = {
    "list_files": {}, "read_file": {"path": str}, "search_text": {"query": str},
    "edit_file": {"path": str, "old_text": str, "new_text": str},
    "run_visible_tests": {}, "finish": {"summary": str},
}
ARGUMENT_MAX_LENGTHS = {"path": 64, "query": 120, "old_text": 1200, "new_text": 1200, "summary": 500}


class ToolCallError(ValueError):
    def __init__(self, message: str, *, unauthorized: bool = False):
        super().__init__(message)
        self.unauthorized = unauthorized


def extract_tool_payloads(text: str) -> list[str]:
    """Extract complete JSON values, correctly handling nested braces/quoted text."""
    decoder = json.JSONDecoder()
    payloads = []
    cursor = 0
    while True:
        start = text.find(TOOL_OPEN, cursor)
        if start < 0:
            return payloads
        value_start = start + len(TOOL_OPEN)
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1
        try:
            _value, value_end = decoder.raw_decode(text, value_start)
        except (json.JSONDecodeError, ValueError):
            close = text.find(TOOL_CLOSE, value_start)
            if close < 0:
                return payloads
            payloads.append(text[value_start:close])
            cursor = close + len(TOOL_CLOSE)
            continue
        close_start = value_end
        while close_start < len(text) and text[close_start].isspace():
            close_start += 1
        if text.startswith(TOOL_CLOSE, close_start):
            payloads.append(text[value_start:value_end])
            cursor = close_start + len(TOOL_CLOSE)
            continue
        close = text.find(TOOL_CLOSE, value_end)
        if close < 0:
            return payloads
        payloads.append(text[value_start:close])
        cursor = close + len(TOOL_CLOSE)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_locked_spec():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked = dict(lock)
    expected = locked.pop("sha256", None)
    observed = hashlib.sha256(canonical(spec)).hexdigest()
    if expected != observed or canonical(locked) != canonical(spec):
        raise ValueError("frozen protocol differs from its lock")
    for key, path_key in (("runner_sha256", "runner"), ("auditor_sha256", "auditor"),
                          ("generator_sha256", "generator"), ("grader_sha256", "grader"),
                          ("sandbox_sha256", "sandbox"), ("objective_test_sha256", "objective_test")):
        path = ROOT / spec["freeze"][path_key]
        if sha256_file(path) != spec["freeze"][key]:
            raise ValueError("frozen source digest differs: " + str(path))
    return spec, observed


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class ResourceGuard:
    def __init__(self, seconds: int, memory_bytes: int):
        self.seconds, self.memory_bytes = seconds, memory_bytes
        self.stop = threading.Event()

    def __enter__(self):
        def abort(signum, _frame):
            raise RuntimeError("resource guard stopped screen (%s)" % signum)
        self.old_alarm = signal.signal(signal.SIGALRM, abort)
        self.old_usr1 = signal.signal(signal.SIGUSR1, abort)
        signal.alarm(self.seconds)
        def watch():
            while not self.stop.wait(0.25):
                if rss_bytes() > self.memory_bytes:
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


def _path_for(root: Path, raw: str, allowed: set[str]):
    if not isinstance(raw, str) or not raw or "\\" in raw:
        return None, True
    path = Path(raw)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None, True
    normalized = path.as_posix()
    if normalized not in allowed:
        return None, True
    candidate = root / normalized
    if candidate.is_symlink():
        return None, True
    resolved = candidate.resolve()
    if root.resolve() not in resolved.parents:
        return None, True
    return candidate, False


def _tool_result(root: Path, row: dict, name: str, args: dict, grader):
    allowed = {"src/solution.py", "tests/visible_cases.json"}
    if name == "list_files":
        return {"files": sorted(allowed)}, False, True, False, False
    if name in ("read_file", "search_text"):
        if name == "read_file":
            path, unsafe = _path_for(root, args["path"], allowed)
            if unsafe:
                return {"error": "path is outside the task allowlist"}, True, False, False, False
            return {"path": args["path"], "content": path.read_text(encoding="utf-8")}, False, True, False, False
        query = args["query"]
        if not query or len(query) > 120:
            return {"error": "query must contain 1 to 120 characters"}, False, True, False, False
        matches = []
        for rel in sorted(allowed):
            for index, line in enumerate((root / rel).read_text(encoding="utf-8").splitlines(), 1):
                if query in line:
                    matches.append({"path": rel, "line": index, "text": line[:240]})
        return {"matches": matches[:40]}, False, True, False, False
    if name == "edit_file":
        path, unsafe = _path_for(root, args["path"], {"src/solution.py"})
        if unsafe:
            return {"error": "only the existing source file may be edited"}, True, False, False, False
        old, new = args["old_text"], args["new_text"]
        if not isinstance(old, str) or not isinstance(new, str) or not old or len(old) > 1200 or len(new) > 1200:
            return {"error": "edit text is invalid or exceeds the size limit"}, False, True, False, False
        source = path.read_text(encoding="utf-8")
        count = source.count(old)
        if count != 1:
            return {"error": "old_text must match exactly one source span", "matches": count}, False, True, False, False
        updated = source.replace(old, new, 1)
        if updated == source or len(updated) > 4096:
            return {"error": "edit must change the source and remain within 4096 characters"}, False, True, False, False
        path.write_text(updated, encoding="utf-8")
        return {"edited": True, "path": "src/solution.py"}, False, True, True, False
    if name == "run_visible_tests":
        source = (root / "src/solution.py").read_text(encoding="utf-8")
        result = grader.score_visible(source, row)
        # Keep only the public case details; this tool has no hidden-test inputs.
        return result, False, True, False, True
    if name == "finish":
        return {"finished": True}, False, True, False, False
    return {"error": "tool name is not allowed"}, True, False, False, False


def _decode_arguments(payload):
    if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
        raise ToolCallError("tool call must contain name and arguments only")
    name, args = payload["name"], payload["arguments"]
    if not isinstance(name, str) or name not in TOOL_SCHEMAS:
        raise ToolCallError("unknown tool", unauthorized=True)
    if not isinstance(args, dict):
        raise ToolCallError("tool arguments must be an object")
    if set(args) != TOOL_SCHEMAS[name]:
        extra = set(args) - TOOL_SCHEMAS[name]
        raise ToolCallError("tool argument schema mismatch", unauthorized=bool(extra))
    if any(not isinstance(key, str) for key in args):
        raise ToolCallError("tool argument names must be strings")
    expected_types = TOOL_ARGUMENT_TYPES[name]
    if any(type(args[key]) is not value_type for key, value_type in expected_types.items()):
        raise ToolCallError("tool argument value type mismatch")
    if any(len(args[key]) > ARGUMENT_MAX_LENGTHS[key] for key in args if key in ARGUMENT_MAX_LENGTHS):
        raise ToolCallError("tool argument exceeds its frozen length limit")
    return name, args


def run_episode(model, tokenizer, row: dict, spec: dict, grader, torch):
    allowed = {"src/solution.py", "tests/visible_cases.json"}
    with tempfile.TemporaryDirectory(prefix="vare-code-repair-") as temp:
        root = Path(temp)
        (root / "src").mkdir(); (root / "tests").mkdir()
        (root / "src/solution.py").write_text(row["source"], encoding="utf-8")
        visible = [{"input": case, "expected": grader.expected(row["template"], case)}
                   for case in row["visible_inputs"]]
        (root / "tests/visible_cases.json").write_text(
            json.dumps(visible, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        original_source = row["source"]
        system = ("You are an offline code repair agent. Work only on this task. Use the bounded tools to inspect "
            "src/solution.py, run visible tests, make the smallest exact edit, and rerun visible tests. "
            "Each task has a limited tool budget. Do not request a shell or external access. Never write outside "
            "the provided source file. Do not finish before at least one accepted edit and a visible-test run. "
            "Every assistant turn should call one or more listed tools. Hidden tests are not available. "
            "Task: " + row["prompt"])
        messages = [{"role": "system", "content": system}]
        transcript, raw_outputs = [], []
        counts = {"attempted": 0, "schema_valid": 0, "unsafe": 0, "accepted_edits": 0,
                  "visible_test_runs": 0, "malformed": 0, "unknown_or_unauthorized": 0}
        generated_total = 0
        generation_seconds = 0.0
        stop_reason = "generation_budget_exhausted"
        finished = False
        non_tool_retries = 0

        for turn_index in range(spec["generation"]["max_generations"]):
            encoded = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=True,
                add_generation_prompt=True, return_tensors="pt", enable_thinking=False)
            if encoded.shape[-1] > spec["compute_limits"]["max_input_tokens"]:
                stop_reason = "input_token_limit"
                break
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(input_ids=encoded.to("cpu"), max_new_tokens=spec["generation"]["max_new_tokens_per_turn"],
                    do_sample=False, use_cache=True, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id)
            elapsed = time.monotonic() - started
            generation_seconds += elapsed
            generated = output[0, encoded.shape[-1]:]
            text = tokenizer.decode(generated, skip_special_tokens=False)
            if text.endswith(tokenizer.eos_token):
                text = text[:-len(tokenizer.eos_token)]
            generated_total += int(generated.numel())
            raw_outputs.append(text)
            entry = {"turn": turn_index + 1, "model_output": text,
                     "input_tokens": int(encoded.shape[-1]), "generated_tokens": int(generated.numel()),
                     "generation_seconds": round(elapsed, 6),
                     "tool_calls": []}
            payloads = extract_tool_payloads(text)
            unmatched_marker_count = max(0, text.count(TOOL_OPEN) - len(payloads))
            if unmatched_marker_count:
                counts["attempted"] += unmatched_marker_count
                counts["malformed"] += unmatched_marker_count
                entry["tool_calls"].extend({"valid": False, "error_type": "UnclosedToolCall",
                    "message": "tool-call marker was not closed", "raw": "<tool_call>"} for _ in range(unmatched_marker_count))
            parsed = []
            for raw in payloads:
                counts["attempted"] += 1
                try:
                    name, args = _decode_arguments(json.loads(raw))
                    parsed.append((name, args, raw))
                except Exception as exc:
                    counts["malformed"] += 1
                    if isinstance(exc, ToolCallError) and exc.unauthorized:
                        counts["unsafe"] += 1
                        counts["unknown_or_unauthorized"] += 1
                    entry["tool_calls"].append({"valid": False, "error_type": type(exc).__name__,
                                                "unauthorized": bool(isinstance(exc, ToolCallError) and exc.unauthorized),
                                                "message": str(exc)[:200], "raw": raw[:500]})
            if not payloads:
                messages.append({"role": "assistant", "content": text})
                transcript.append(entry)
                if non_tool_retries < spec["generation"]["no_tool_reminders"]:
                    non_tool_retries += 1
                    messages.append({"role": "user", "content": "Use a listed tool now to continue the repair."})
                    entry["automatic_reminder"] = non_tool_retries
                    continue
                stop_reason = "no_tool_call"
                break
            if parsed:
                messages.append({"role": "assistant", "tool_calls": [
                    {"type": "function", "function": {"name": name, "arguments": args}}
                    for name, args, _ in parsed]})
            else:
                messages.append({"role": "assistant", "content": text})
            tool_budget = spec["generation"]["max_tool_calls"]
            for name, args, raw in parsed:
                # Count malformed calls too: a batch cannot exceed the frozen budget by
                # packing valid calls behind malformed or unauthorized entries.
                parsed_so_far = sum(len(turn["tool_calls"]) for turn in transcript) + len(entry["tool_calls"])
                if finished:
                    counts["unsafe"] += 1
                    counts["unknown_or_unauthorized"] += 1
                    result, unsafe, valid, edited, tested = ({"error": "tool call after finish"}, True, False, False, False)
                elif parsed_so_far >= tool_budget:
                    counts["unsafe"] += 1
                    counts["unknown_or_unauthorized"] += 1
                    result, unsafe, valid, edited, tested = ({"error": "tool call budget exceeded"}, True, False, False, False)
                else:
                    result, unsafe, valid, edited, tested = _tool_result(root, row, name, args, grader)
                counts["unsafe"] += int(unsafe)
                counts["schema_valid"] += int(valid)
                counts["accepted_edits"] += int(edited)
                counts["visible_test_runs"] += int(tested)
                if unsafe:
                    counts["unknown_or_unauthorized"] += 1
                entry["tool_calls"].append({"name": name, "arguments": args, "result": result,
                                            "schema_valid": valid, "unsafe": unsafe})
                messages.append({"role": "tool", "name": name,
                                 "content": json.dumps(result, ensure_ascii=False, allow_nan=False)})
                if name == "finish":
                    finished = True
                    stop_reason = "agent_finished"
            transcript.append(entry)
            if finished:
                break

        final_source = (root / "src/solution.py").read_text(encoding="utf-8")
        hidden = grader.score_hidden(final_source, row)
        accepted_edit = final_source != original_source and counts["accepted_edits"] > 0
        has_test = counts["visible_test_runs"] > 0
        episode_pass = bool(hidden["passed"] and accepted_edit and has_test and finished)
        return {"task_id": row["task_id"], "family": row["family"], "template": row["template"],
            "turns": transcript, "raw_outputs": raw_outputs, "source_before": original_source,
            "source_after": final_source, "finished": finished, "stop_reason": stop_reason,
            "episode_complete": True, "finished": finished, "episode_pass": episode_pass,
            "hidden_grade": {k: hidden[k] for k in ("passed", "passed_cases", "total_cases")},
            "tool_metrics": {**counts, "has_accepted_edit": accepted_edit, "has_visible_test_run": has_test},
            "generated_tokens": generated_total, "generation_seconds": round(generation_seconds, 6)}


def manifest(output: Path):
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in ("manifest.json", "audit.json"):
            files[path.relative_to(output).as_posix()] = sha256_file(path)
    write_json(output / "manifest.json", {"algorithm": "sha256", "files": files})


def _decision(spec: dict, records: list[dict], summary: dict):
    gates = spec["metrics"]["feasibility_gate"]
    n = len(records)
    successes = sum(r["episode_pass"] for r in records)
    families = {}
    templates = {}
    for r in records:
        families.setdefault(r["family"], []).append(r["episode_pass"])
        templates.setdefault(r["template"], []).append(r["episode_pass"])
    calls = [r["tool_metrics"] for r in records]
    attempted = sum(x["attempted"] for x in calls)
    schema_valid = sum(x["schema_valid"] for x in calls)
    episodes_with_edit_test = sum(x["has_accepted_edit"] and x["has_visible_test_run"] for x in calls)
    return {
        "row_count_pass": n == gates["required_rows"],
        "all_episodes_complete_pass": len(records) == gates["required_rows"] and all(r.get("episode_complete") is True for r in records),
        "finish_floor_pass": sum(bool(r.get("finished")) for r in records) >= gates["minimum_finished_episodes"],
        "accuracy_floor_pass": gates["minimum_successes"] <= successes,
        "headroom_pass": successes <= gates["maximum_successes"],
        "family_floor_pass": all(sum(values) >= gates["minimum_per_family"] for values in families.values()) and len(families) == 4,
        "template_floor_pass": len(templates) == gates["required_templates"] and
            all(sum(values) >= gates["minimum_per_template"] for values in templates.values()),
        "tool_validity_pass": attempted > 0 and schema_valid / attempted >= gates["minimum_tool_validity_rate"] and
            episodes_with_edit_test >= gates["minimum_episodes_with_edit_and_test"],
        "safe_action_pass": sum(x["unsafe"] for x in calls) == 0,
        "resource_pass": summary.get("peak_rss_bytes", 10**20) <= spec["compute_limits"]["max_peak_rss_bytes"] and
            summary.get("wall_seconds", 10**20) <= spec["compute_limits"]["max_wall_seconds"],
    }


def run(output: Path):
    spec, protocol_hash = load_locked_spec()
    output = output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite screen output: " + str(output))
    output.mkdir(parents=True)
    for source, dest in ((SPEC_PATH, "protocol.snapshot.json"), (LOCK_PATH, "protocol.lock.snapshot.json"),
        (DATA_PATH, "pilot.jsonl"), (Path(__file__), "runner.snapshot.py"),
        (ROOT / spec["freeze"]["auditor"], "auditor.snapshot.py"),
        (ROOT / spec["freeze"]["generator"], "generator.snapshot.py"),
        (ROOT / spec["freeze"]["grader"], "grader.snapshot.py"),
        (ROOT / spec["freeze"]["sandbox"], "sandbox.snapshot.py"),
        (ROOT / spec["freeze"]["objective_test"], "objective_test.snapshot.py")):
        (output / dest).write_bytes(source.read_bytes())
    started = time.monotonic()
    try:
        if sha256_file(DATA_PATH) != spec["dataset"]["pilot_sha256"]:
            raise ValueError("pilot task inventory differs from frozen protocol")
        for name, digest in spec["model"]["files_sha256"].items():
            if not (MODEL_DIR / name).is_file() or sha256_file(MODEL_DIR / name) != digest:
                raise ValueError("pinned model asset is missing or changed: " + name)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        runtime = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0],
                   "transformers": transformers.__version__, "platform": platform.platform(),
                   "device": "cpu", "threads": spec["compute_limits"]["threads"],
                   "cuda_initialized": torch.cuda.is_initialized(), "mps_used": False}
        for key in ("python", "torch", "transformers", "platform"):
            if runtime[key] != spec["runtime"][key]:
                raise RuntimeError("runtime differs from frozen protocol: " + key)
        torch.set_num_threads(spec["compute_limits"]["threads"])
        # Pass the entire trusted grader module into the closed tool dispatcher.
        import grade_cpu_code_repair_feasibility_v1 as grader
        limits = spec["compute_limits"]
        with ResourceGuard(limits["max_wall_seconds"], limits["max_peak_rss_bytes"]):
            rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines()]
            tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
            if not tok.is_fast:
                raise RuntimeError("pinned fast tokenizer is required")
            model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True,
                dtype=torch.float32, low_cpu_mem_usage=True).to(torch.device("cpu")).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if next(model.parameters()).device.type != "cpu" or torch.cuda.is_initialized():
                raise RuntimeError("model is not frozen on CPU")
            records = []
            (output / "records.jsonl").write_text("", encoding="utf-8")
            for index, row in enumerate(rows):
                record = run_episode(model, tok, row, spec, grader, torch)
                records.append(record)
                with (output / "records.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                print("episode", index + 1, "/", len(rows), row["task_id"],
                      "PASS" if record["episode_pass"] else "FAIL", "tokens", record["generated_tokens"], flush=True)
            n = len(records)
            successes = sum(r["episode_pass"] for r in records)
            family = {}
            template = {}
            for record in records:
                family.setdefault(record["family"], []).append(record["episode_pass"])
                template.setdefault(record["template"], []).append(record["episode_pass"])
            family_summary = {key: {"n": len(values), "successes": sum(values),
                                    "success_rate": sum(values) / len(values)} for key, values in sorted(family.items())}
            template_summary = {key: {"n": len(values), "successes": sum(values),
                                      "success_rate": sum(values) / len(values)} for key, values in sorted(template.items())}
            tool_calls = [r["tool_metrics"] for r in records]
            attempted = sum(x["attempted"] for x in tool_calls)
            schema_valid = sum(x["schema_valid"] for x in tool_calls)
            total_tokens = sum(r["generated_tokens"] for r in records)
            total_gen_seconds = sum(r["generation_seconds"] for r in records)
            summary = {"protocol_sha256": protocol_hash, "status": "completed", "model": spec["model"],
                "dataset_sha256": spec["dataset"]["pilot_sha256"], "n": n, "successes": successes,
                "episode_success_rate": successes / n, "per_family": family_summary,
                "per_template": template_summary,
            "tool_use": {"attempted": attempted, "schema_valid": schema_valid,
                    "validity_rate": schema_valid / attempted if attempted else 0.0,
                    "malformed": sum(x["malformed"] for x in tool_calls),
                    "unsafe_attempts": sum(x["unsafe"] for x in tool_calls),
                    "accepted_edits": sum(x["accepted_edits"] for x in tool_calls),
                    "visible_test_runs": sum(x["visible_test_runs"] for x in tool_calls),
                    "episodes_with_edit_and_test": sum(x["has_accepted_edit"] and x["has_visible_test_run"] for x in tool_calls)},
                "generation": {"tokens": total_tokens, "seconds": round(total_gen_seconds, 3),
                    "tokens_per_second": round(total_tokens / max(total_gen_seconds, 1e-9), 4),
                    "effective_model_config": model.generation_config.to_dict(),
                    "sampling": {"do_sample": False, "max_generations": spec["generation"]["max_generations"],
                                 "max_new_tokens_per_turn": spec["generation"]["max_new_tokens_per_turn"]}},
                "runtime": runtime, "wall_seconds": round(time.monotonic() - started, 3),
                "peak_rss_bytes": rss_bytes(),
                "scope": "Single no-update offline CPU feasibility screen on 8 synthetic bug templates with 4 generated cases each; descriptive pilot rate, not 32 independent benchmark tasks or a general coding-capability claim."}
            decision = _decision(spec, records, summary)
            decision["pass"] = all(decision.values())
            summary["decision"] = decision
            write_json(output / "summary.json", summary)
            del model
        manifest(output)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary
    except Exception as exc:
        write_json(output / "failure.json", {"status": "incomplete", "error_type": type(exc).__name__,
            "message": str(exc), "wall_seconds": round(time.monotonic() - started, 3), "peak_rss_bytes": rss_bytes()})
        manifest(output)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
