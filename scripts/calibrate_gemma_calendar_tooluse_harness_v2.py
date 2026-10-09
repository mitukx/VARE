#!/usr/bin/env python3
"""Check the one-off calendar simulator with known passing and failing plans."""

from __future__ import annotations

import importlib.util
import argparse
import json
from pathlib import Path
import sys

RUNNER = Path(__file__).with_name("run_gemma_calendar_tooluse_base_gate_v2.py")
spec = importlib.util.spec_from_file_location("calendar_gate", RUNNER)
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


def action(tool: str, **arguments):
    return {"tool": tool, "arguments": arguments}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": gate.SYSTEM + "\n\n{}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(rendered, str) and rendered
    tasks = gate.make_tasks()
    assert len(tasks) == 32
    results = {"tasks": len(tasks), "known_pass": 0, "known_fail": 0, "parser_rejects": 0,
               "protected_action_rejects": 0, "extra_state_rejects": 0,
               "single_user_chat_template_supported": True,
               "cancel_recreate_identity_rejects": 0}
    for task in tasks:
        state = {e.event_id: gate.Event(**gate.asdict(e)) for e in task.initial_events}
        assert not gate.success(task, state)
        results["known_fail"] += 1
        if task.family == "book_free_slot":
            plan = [action("create_event", title=task.target_events[0].title, day=task.target_events[0].day,
                           time=task.target_events[0].time, participants=task.target_events[0].participants)]
        elif task.family == "move_conflict_then_book":
            plan = [action("move_event", event_id="E1", day=task.target_events[0].day,
                           time=task.target_events[0].time),
                    action("create_event", title=task.target_events[1].title, day=task.target_events[1].day,
                           time=task.target_events[1].time, participants=task.target_events[1].participants)]
        elif task.family == "reschedule_with_conflict":
            plan = [action("move_event", event_id="E2", day=task.target_events[1].day,
                           time=task.target_events[1].time),
                    action("move_event", event_id="E1", day=task.target_events[0].day,
                           time=task.target_events[0].time)]
        else:
            plan = [action("cancel_event", event_id="E2"),
                    action("create_event", title=task.target_events[1].title, day=task.target_events[1].day,
                           time=task.target_events[1].time, participants=task.target_events[1].participants)]
        for step in plan:
            ok, message = gate.apply_action(state, step)
            assert ok, (task.task_id, step, message)
        assert gate.success(task, state), task.task_id
        results["known_pass"] += 1

        if task.family == "move_conflict_then_book":
            substituted = {e.event_id: gate.Event(**gate.asdict(e)) for e in task.initial_events}
            ok, _ = gate.apply_action(substituted, action("cancel_event", event_id="E1"))
            assert ok
            target = task.target_events[0]
            ok, _ = gate.apply_action(
                substituted,
                action("create_event", title=target.title, day=target.day, time=target.time,
                       participants=target.participants),
            )
            assert ok
            assert not gate.success(task, substituted)
            results["cancel_recreate_identity_rejects"] += 1

        protected = {e.event_id: gate.Event(**gate.asdict(e)) for e in task.initial_events}
        protected_id = next((e.event_id for e in task.initial_events if not e.movable), None)
        if protected_id:
            ok, _ = gate.apply_action(protected, action("cancel_event", event_id=protected_id))
            assert not ok
            results["protected_action_rejects"] += 1

        bad = {e.event_id: gate.Event(**gate.asdict(e)) for e in task.target_events}
        bad["EXTRA"] = gate.Event("EXTRA", "Unrequested", "Thu", "16:00", ["Hal"])
        assert not gate.success(task, bad)
        results["extra_state_rejects"] += 1

    assert gate.parse_action("not json")[0] is None
    assert gate.parse_action('{"tool":"create_event","arguments":{},"extra":1}')[0] is None
    results["parser_rejects"] = 2
    assert results["cancel_recreate_identity_rejects"] == 8
    results["status"] = "pass"
    print(json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
