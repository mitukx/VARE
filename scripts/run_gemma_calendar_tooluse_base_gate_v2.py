#!/usr/bin/env python3
"""Frozen local CPU feasibility gate for multi-turn calendar tool-use tasks.

This is a one-off base-model gate, not a general agent framework or capability
benchmark. The protocol lock controls cohort, limits, and decision criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FAMILIES = ("book_free_slot", "move_conflict_then_book", "reschedule_with_conflict", "remove_duplicate_then_book")
DAYS = ("Mon", "Tue", "Wed", "Thu")
TIMES = ("09:00", "11:00", "14:00", "16:00")
PEOPLE = ("Ari", "Bea", "Cem", "Dee", "Eli", "Flo", "Gus", "Hal")
MODEL_REVISION = "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
MODEL_FILES = {
    "config.json": "eacec6c5ca317a87ed2c46789d9705b9274db5027e7ba59da739bfae23addb55",
    "generation_config.json": "a543a5d299bc2b20c52bd87ed174f561266510b57a392e12b5b5d758d798ce05",
    "model.safetensors.index.json": "ada0043f3e3b2e5ab2f445cad9c0fbbf9d91ad444675e6a82b822591c63abf5a",
    "model-00001-of-00002.safetensors": "532d792c9178805064170a3ec485b7dedbfccc6fd297b92c31a6091b6c7e41bf",
    "model-00002-of-00002.safetensors": "6d6d9ce84db398fb6e0191f91542e5da0a73da2cb695e172a24edc2146dc8d20",
    "tokenizer.json": "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922",
    "tokenizer.model": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    "tokenizer_config.json": "cb32b7929c62608d46572e813112b3ad8a841fb98fdd6a4da8559e368a951c89",
    "special_tokens_map.json": "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
}
SYSTEM = """You are operating a calendar using the listed tools. Complete the user's request by making tool calls one at a time. After each call, you will receive the updated calendar. Only output one JSON object and no prose, in this exact shape: {\"tool\":\"create_event|move_event|cancel_event\",\"arguments\":{...}}. Use only a listed tool and its exact argument names. Do not cancel or move an event unless the user asks or it blocks the requested booking. Stop once the request is satisfied."""


@dataclass
class Event:
    event_id: str
    title: str
    day: str
    time: str
    participants: list[str]
    movable: bool = True


@dataclass
class Task:
    task_id: str
    family: str
    request: str
    initial_events: list[Event]
    target_events: list[Event]
    removed_ids: list[str]
    immutable_ids: list[str]


def _person(rng: random.Random, exclude: set[str] | None = None) -> str:
    choices = [p for p in PEOPLE if p not in (exclude or set())]
    return rng.choice(choices)


def _event(event_id: str, title: str, day: str, slot: str, people: list[str], movable: bool = True) -> Event:
    return Event(event_id, title, day, slot, people, movable)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_tasks(seed: int = 4102026, per_family: int = 8) -> list[Task]:
    """Generate a fixed, balanced synthetic task cohort from a locked seed."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    serial = 0
    for family in FAMILIES:
        for index in range(per_family):
            serial += 1
            p1 = _person(rng)
            p2 = _person(rng, {p1})
            day_a = DAYS[index % len(DAYS)]
            time_a = TIMES[(index + 1) % len(TIMES)]
            day_b = DAYS[(index + 1) % len(DAYS)]
            time_b = TIMES[(index + 2) % len(TIMES)]
            title = f"{('Planning', 'Review', 'Handoff', 'Design')[index % 4]} sync {serial:02d}"
            initial: list[Event] = []
            targets: list[Event] = []
            removed: list[str] = []
            immutable: list[str] = []

            if family == "book_free_slot":
                initial = [
                    _event("E1", "Focus block", day_a, time_b, [p1], False),
                    _event("E2", "Team check-in", day_b, time_a, [p2], False),
                ]
                targets = [_event("NEW", title, day_a, time_a, [p1, p2])]
                immutable = ["E1", "E2"]
                request = f"Schedule {title} with {p1} and {p2} on {day_a} at {time_a}. Keep all existing events."

            elif family == "move_conflict_then_book":
                initial = [
                    _event("E1", "Movable one-on-one", day_a, time_a, [p2]),
                    _event("E2", "Protected review", day_b, time_b, [p1], False),
                ]
                targets = [
                    _event("E1", "Movable one-on-one", day_b, time_a, [p2]),
                    _event("NEW", title, day_a, time_a, [p1, p2]),
                ]
                immutable = ["E2"]
                request = (f"Book {title} with {p1} and {p2} on {day_a} at {time_a}. "
                           f"If a conflict blocks it, move only the movable event to {day_b} at {time_a}; "
                           "leave protected events unchanged.")

            elif family == "reschedule_with_conflict":
                old_day, old_time = DAYS[(index + 2) % len(DAYS)], TIMES[index % len(TIMES)]
                initial = [
                    _event("E1", title, old_day, old_time, [p1, p2]),
                    _event("E2", "Movable conflict", day_a, time_a, [p2]),
                    _event("E3", "Protected appointment", day_b, time_b, [p1], False),
                ]
                targets = [
                    _event("E1", title, day_a, time_a, [p1, p2]),
                    _event("E2", "Movable conflict", day_b, time_a, [p2]),
                ]
                immutable = ["E3"]
                request = (f"Move {title} to {day_a} at {time_a}. {p1} and {p2} must both attend. "
                           f"Move the movable conflict to {day_b} at {time_a} if needed; do not alter protected events.")

            else:
                duplicate = f"{title} duplicate"
                initial = [
                    _event("E1", title, day_b, time_b, [p1], False),
                    _event("E2", duplicate, day_a, time_b, [p1], True),
                    _event("E3", "Other commitment", day_b, time_a, [p2], False),
                ]
                targets = [
                    _event("E1", title, day_b, time_b, [p1], False),
                    _event("NEW", title + " with " + p2, day_a, time_a, [p1, p2]),
                ]
                removed = ["E2"]
                immutable = ["E1", "E3"]
                request = (f"Keep the existing {title} on {day_b} at {time_b}. Cancel only its duplicate {duplicate}. "
                           f"Then schedule {title} with {p1} and {p2} on {day_a} at {time_a}.")

            tasks.append(Task(f"cal-{serial:03d}", family, request, initial, targets, removed, immutable))
    return tasks


def _busy(events: dict[str, Event], day: str, slot: str, participants: list[str], except_id: str | None = None) -> bool:
    for event_id, event in events.items():
        if event_id == except_id:
            continue
        if event.day == day and event.time == slot and set(event.participants) & set(participants):
            return True
    return False


def apply_action(events: dict[str, Event], action: dict[str, Any]) -> tuple[bool, str]:
    tool = action.get("tool")
    args = action.get("arguments")
    if not isinstance(args, dict):
        return False, "arguments must be an object"
    if tool == "create_event":
        required = {"title", "day", "time", "participants"}
        if set(args) != required:
            return False, "create_event requires title, day, time, participants"
        if (not isinstance(args["title"], str) or not args["title"].strip()
                or args["day"] not in DAYS or args["time"] not in TIMES
                or not isinstance(args["participants"], list)
                or not args["participants"]
                or any(not isinstance(p, str) or p not in PEOPLE for p in args["participants"])
                or len(set(args["participants"])) != len(args["participants"])):
            return False, "invalid day, time, or participants"
        if _busy(events, args["day"], args["time"], args["participants"]):
            return False, "participant conflict at requested slot"
        event_id = f"N{len([k for k in events if k.startswith('N')]) + 1}"
        events[event_id] = Event(event_id, args["title"], args["day"], args["time"], args["participants"])
        return True, f"created {event_id}"
    if tool == "move_event":
        required = {"event_id", "day", "time"}
        if set(args) != required:
            return False, "move_event requires event_id, day, time"
        if not isinstance(args["event_id"], str):
            return False, "event_id must be a string"
        event = events.get(args["event_id"])
        if event is None:
            return False, "unknown event_id"
        if not event.movable:
            return False, "event is protected and cannot be moved"
        if (not isinstance(args["day"], str) or not isinstance(args["time"], str)
                or args["day"] not in DAYS or args["time"] not in TIMES):
            return False, "invalid day or time"
        if _busy(events, args["day"], args["time"], event.participants, except_id=event.event_id):
            return False, "participant conflict at requested slot"
        event.day, event.time = args["day"], args["time"]
        return True, f"moved {event.event_id}"
    if tool == "cancel_event":
        if set(args) != {"event_id"}:
            return False, "cancel_event requires event_id"
        if not isinstance(args["event_id"], str):
            return False, "event_id must be a string"
        event = events.get(args["event_id"])
        if event is None:
            return False, "unknown event_id"
        if not event.movable:
            return False, "event is protected and cannot be cancelled"
        del events[args["event_id"]]
        return True, f"cancelled {args['event_id']}"
    return False, "unknown tool"


def parse_action(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        action = json.loads(text.strip())
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e.msg}"
    if not isinstance(action, dict) or set(action) != {"tool", "arguments"}:
        return None, "output must contain exactly tool and arguments"
    if not isinstance(action["tool"], str) or not isinstance(action["arguments"], dict):
        return None, "tool must be a string and arguments an object"
    return action, None


def view_prompt(task: Task, events: dict[str, Event], previous_error: str | None) -> str:
    event_rows = [asdict(events[k]) for k in sorted(events)]
    prompt = {
        "request": task.request,
        "calendar": event_rows,
        "tools": {
            "create_event": {"title": "string", "day": list(DAYS), "time": list(TIMES), "participants": "list[string]"},
            "move_event": {"event_id": "string", "day": list(DAYS), "time": list(TIMES)},
            "cancel_event": {"event_id": "string"},
        },
        "instruction": "Make one tool call now. The calendar will be shown again after the result.",
    }
    if previous_error:
        prompt["last_tool_result"] = {"ok": False, "error": previous_error}
    return json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))


def success(task: Task, events: dict[str, Event]) -> bool:
    for event_id in task.removed_ids:
        if event_id in events:
            return False
    for event_id in task.immutable_ids:
        before = next(e for e in task.initial_events if e.event_id == event_id)
        now = events.get(event_id)
        if now is None or asdict(now) != asdict(before):
            return False
    def canonical(event: Event) -> tuple[str, str, str, tuple[str, ...], bool]:
        return event.title, event.day, event.time, tuple(sorted(event.participants)), event.movable
    immutable = {e.event_id: e for e in task.initial_events if e.event_id in task.immutable_ids}
    immutable.update({e.event_id: e for e in task.target_events})
    expected = list(immutable.values())
    # Existing events retain identity. Only environment-created targets may
    # receive a new ID; matching fields under a replacement ID is not success.
    for expected_event in expected:
        if expected_event.event_id == "NEW":
            continue
        actual = events.get(expected_event.event_id)
        if actual is None or canonical(actual) != canonical(expected_event):
            return False
    initial_ids = {e.event_id for e in task.initial_events}
    actual_new = [e for e in events.values() if e.event_id not in initial_ids]
    expected_new = [e for e in expected if e.event_id == "NEW"]
    if sorted(map(canonical, actual_new)) != sorted(map(canonical, expected_new)):
        return False
    return sorted(map(canonical, events.values())) == sorted(map(canonical, expected))


def generate_one(model, tokenizer, task: Task, events, last_error: str | None, max_new_tokens: int) -> tuple[str, float]:
    # Gemma 2's pinned template rejects `system`; place fixed instructions in
    # the single supported user message and validate this in calibration.
    user = SYSTEM + "\n\n" + view_prompt(task, events, last_error)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    seconds = time.monotonic() - started
    text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text, seconds


def run(args: argparse.Namespace) -> int:
    run_started = time.monotonic()
    torch.set_num_threads(args.cpu_threads)
    model_dir = Path(args.model_dir).expanduser().resolve()
    if args.model_revision != MODEL_REVISION:
        raise RuntimeError("unexpected model revision")
    for name, expected in MODEL_FILES.items():
        if _sha256(model_dir / name) != expected:
            raise RuntimeError(f"model file hash mismatch: {name}")
    model_meta = json.loads((model_dir / "config.json").read_text())
    if model_meta.get("architectures") != ["Gemma2ForCausalLM"]:
        raise RuntimeError("unexpected model architecture")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, dtype="auto", device_map={"": "cpu"}, low_cpu_mem_usage=True
    )
    model.eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("non-CPU model placement rejected")
    if (sum(p.numel() for p in model.parameters()) != 2614341888
            or str(next(model.parameters()).dtype) != "torch.bfloat16"
            or len(tokenizer) != 256000):
        raise RuntimeError("loaded model/tokenizer metadata differs from frozen identity")

    tasks = make_tasks()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    generation_seconds_total = 0.0
    for task in tasks:
        events = {e.event_id: Event(**asdict(e)) for e in task.initial_events}
        trace: list[dict[str, Any]] = []
        last_error: str | None = None
        task_start = time.monotonic()
        schema_errors = 0
        tool_errors = 0
        for turn in range(args.max_turns):
            response, generation_s = generate_one(model, tokenizer, task, events, last_error, args.max_new_tokens)
            action, parse_error = parse_action(response)
            record: dict[str, Any] = {"turn": turn + 1, "response": response, "generation_seconds": generation_s}
            if parse_error:
                schema_errors += 1
                tool_errors += 1
                last_error = parse_error
                record["tool_result"] = {"ok": False, "error": parse_error}
            else:
                before = {k: asdict(v) for k, v in events.items()}
                ok, result = apply_action(events, action)
                last_error = None if ok else result
                if not ok:
                    tool_errors += 1
                record.update({"action": action, "tool_result": {"ok": ok, "result": result}, "state_before": before})
            trace.append(record)
            if success(task, events):
                break
            if time.monotonic() - run_started > args.max_wall_seconds:
                break
        generation_seconds_total += sum(r["generation_seconds"] for r in trace)
        rows.append({
            "task_id": task.task_id,
            "family": task.family,
            "success": success(task, events),
            "turns": len(trace),
            "schema_errors": schema_errors,
            "tool_errors": tool_errors,
            "seconds": round(time.monotonic() - task_start, 4),
            "initial_events": [asdict(e) for e in task.initial_events],
            "final_events": [asdict(e) for e in events.values()],
            "trace": trace,
            "oracle": {"targets": [asdict(e) for e in task.target_events], "removed_ids": task.removed_ids,
                       "immutable_ids": task.immutable_ids},
        })
        result = {
            "protocol": "gemma_calendar_tooluse_base_gate_v1",
            "model_revision": args.model_revision,
            "model_dir": str(model_dir),
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "tokenizer_vocab": len(tokenizer),
            "task_seed": 4102026,
            "tasks_total": len(tasks),
            "tasks_recorded": len(rows),
            "elapsed_seconds": round(time.monotonic() - run_started, 4),
            "generation_seconds_total": round(generation_seconds_total, 4),
            "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cpu_threads": args.cpu_threads,
            "rows": rows,
        }
        tmp = output_path / "result.json.tmp"
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(output_path / "result.json")
        if time.monotonic() - run_started > args.max_wall_seconds:
            break
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > args.max_rss_bytes:
            break
    complete = len(rows) == len(tasks)
    print(json.dumps({"output": str(output_path / "result.json"), "tasks_recorded": len(rows), "complete": complete,
                      "elapsed_seconds": round(time.monotonic() - run_started, 2)}))
    within_limits = (
        time.monotonic() - run_started <= args.max_wall_seconds
        and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= args.max_rss_bytes
    )
    return 0 if complete and within_limits else 124


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-wall-seconds", type=int, default=2700)
    parser.add_argument("--max-rss-bytes", type=int, default=25769803776)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
