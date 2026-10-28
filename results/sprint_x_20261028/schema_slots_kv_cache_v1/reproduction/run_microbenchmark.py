#!/usr/bin/env python3
"""Matched synthetic tool-interaction benchmark for schema-slot prefix reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    model_path = args.model.resolve()
    input_path = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite benchmark evidence: {output}")
    output.mkdir(parents=True)
    input_record = json.loads(input_path.read_text(encoding="utf-8"))
    # Long inert notes enlarge only the shared prefix; they contain no task or
    # answer material and are identical in every measured arm.
    messages = [dict(message) for message in input_record["messages"]]
    archive_notes = "\n".join(
        f"Archive record {index:03d}: synthetic inert controller timing context; no source instructions."
        for index in range(1, 161)
    )
    messages[1]["content"] += archive_notes + "\n\n" + (
        f"Read {input_record['fixture_path']} from line {input_record['read_start']} "
        f"through line {input_record['read_end']} using read_lines, then finish."
    )
    (repo / "src").mkdir(exist_ok=True)

    from mlx_lm import load
    import scripts.run_local_agent as run_local_agent_module
    from scripts.run_local_agent import (
        controller_source_hashes,
        generate_schema_slot_turn,
        parse_tool_output_v1,
        tools_for,
    )
    from vare.agent_edit_v2 import SourceEditLeaseStoreV1

    all_tools = tools_for(
        "structured_replace", tool_interface="lease_bound_exact_text_transactional_v1",
    )
    read_tool = next(tool for tool in all_tools if tool["function"]["name"] == "read_lines")
    finish_tool = next(tool for tool in all_tools if tool["function"]["name"] == "finish")
    protocol_identity = {
        "input_sha256": sha256_file(input_path),
        "model_snapshot_path": str(model_path),
        "repo_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "expanded_messages_sha256": hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")).encode(),
        ).hexdigest(),
        "controller_source_hashes": {
            arm: controller_source_hashes(
                "lease_bound_exact_text_transactional_v1", arm,
            ) for arm in ("schema_slots_v9", "schema_slots_v10")
        },
        "read_tool_schema_sha256": hashlib.sha256(
            json.dumps(read_tool, sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest(),
        "finish_tool_schema_sha256": hashlib.sha256(
            json.dumps(finish_tool, sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest(),
    }
    write_json(output / "runtime-identity.json", protocol_identity)

    model_load_started = time.perf_counter()
    model, tokenizer = load(str(model_path))
    model_load_seconds = time.perf_counter() - model_load_started
    write_json(output / "model-load.json", {
        "status": "loaded",
        "model_load_seconds": round(model_load_seconds, 6),
        "model_snapshot_path": str(model_path),
        "tokenizer_class": type(tokenizer).__name__,
    })

    def call_turn(messages_for_turn: list[dict[str, Any]], tools: list[dict[str, Any]],
                  agent_args: SimpleNamespace) -> tuple[str, int, int, dict[str, Any], list[dict[str, str]]]:
        captured_prompts: list[dict[str, str]] = []
        original_capture = run_local_agent_module.capture_rendered_prompt

        def capture_prompt(current_args: Any, prompt: str) -> str:
            digest = original_capture(current_args, prompt)
            captured_prompts.append({"sha256": digest, "text": prompt})
            return digest

        run_local_agent_module.capture_rendered_prompt = capture_prompt
        try:
            raw, generated, prompt_tokens = generate_schema_slot_turn(
                model, tokenizer, messages_for_turn, tools, limit=128, args=agent_args,
            )
        finally:
            run_local_agent_module.capture_rendered_prompt = original_capture
        return raw, generated, prompt_tokens, agent_args._last_generation_detail, captured_prompts

    arms = ["schema_slots_v9", "schema_slots_v10", "schema_slots_v10", "schema_slots_v9"]
    for index, arm in enumerate(arms, start=1):
        attempt = output / f"attempt-{index:02d}-{arm}"
        attempt.mkdir()
        attempt_started = time.perf_counter()
        write_json(attempt / "attempt-start.json", {
            "attempt_index": index,
            "arm": arm,
            "started_unix_seconds": time.time(),
        })
        attempt_record: dict[str, Any] = {
            "attempt_index": index,
            "arm": arm,
            "input_sha256": protocol_identity["input_sha256"],
            "sampling": {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0},
            "token_budget_per_turn": 128,
            "max_context_tokens": 8192,
            "tool_turns": [],
            "tool_execution": None,
        }
        agent_args = SimpleNamespace(
            max_context_tokens=8192,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            generation_protocol=arm,
            tool_interface="lease_bound_exact_text_transactional_v1",
            replace_text_guidance="original",
        )
        run_messages = [dict(message) for message in messages]
        paired_replicate = 1 if index in (1, 2) else 2
        turn_started = time.perf_counter()
        raw_read, read_tokens, read_prompt_tokens, read_detail, read_prompts = call_turn(
            run_messages, [read_tool], agent_args,
        )
        read_status, read_call = parse_tool_output_v1(raw_read, [read_tool])
        write_json(attempt / "turn-01-read-lines.json", {
            "raw_model_output": raw_read,
            "parse_status": read_status,
            "parsed_call": read_call,
            "generated_tokens": read_tokens,
            "prompt_tokens": read_prompt_tokens,
            "wall_seconds": round(time.perf_counter() - turn_started, 6),
            "generation_protocol_detail": read_detail,
            "rendered_stage_prompts": read_prompts,
        })
        attempt_record["tool_turns"].append({
            "name": "read_lines", "parse_status": read_status,
            "generated_tokens": read_tokens, "prompt_tokens": read_prompt_tokens,
            "generation_protocol_detail": read_detail,
        })
        if read_status != "complete_tool_call" or read_call is None or read_call["name"] != "read_lines":
            attempt_record["outcome"] = "read_tool_call_not_accepted"
        else:
            with tempfile.TemporaryDirectory(prefix="vare-kv-cache-fixture-") as temp:
                workspace = Path(temp)
                source_path = workspace / input_record["fixture_path"]
                source_path.parent.mkdir(parents=True)
                source_path.write_text(input_record["fixture_source"], encoding="utf-8")
                leases = SourceEditLeaseStoreV1(
                    workspace, {input_record["fixture_path"]},
                    token_factory=lambda: f"bench-rep-{paired_replicate:02d}-lease",
                )
                execution_started = time.perf_counter()
                tool_result = leases.read_lines(
                    read_call["arguments"].get("path"),
                    read_call["arguments"].get("start"),
                    read_call["arguments"].get("end"),
                )
                execution_seconds = time.perf_counter() - execution_started
                write_json(attempt / "tool-01-read-lines-result.json", {
                    "tool_name": "read_lines",
                    "arguments": read_call["arguments"],
                    "result": tool_result,
                    "execution_wall_seconds": round(execution_seconds, 6),
                })
                attempt_record["tool_execution"] = {
                    "name": "read_lines", "arguments": read_call["arguments"],
                    "result": tool_result,
                    "wall_seconds": round(execution_seconds, 6),
                }
                expected_arguments = {
                    "path": input_record["fixture_path"],
                    "start": input_record["read_start"],
                    "end": input_record["read_end"],
                }
                if tool_result.get("status") != "read_complete" or read_call["arguments"] != expected_arguments:
                    attempt_record["outcome"] = "read_tool_rejected_or_unexpected_arguments"
                else:
                    next_messages = [
                        *run_messages,
                        {"role": "assistant", "content": raw_read},
                        {"role": "tool", "name": "read_lines",
                         "content": json.dumps(tool_result, ensure_ascii=False, sort_keys=True)},
                        {"role": "user", "content": input_record["second_turn_message"]},
                    ]
                    turn_started = time.perf_counter()
                    raw_finish, finish_tokens, finish_prompt_tokens, finish_detail, finish_prompts = call_turn(
                        next_messages, [finish_tool], agent_args,
                    )
                    finish_status, finish_call = parse_tool_output_v1(raw_finish, [finish_tool])
                    write_json(attempt / "turn-02-finish.json", {
                        "raw_model_output": raw_finish,
                        "parse_status": finish_status,
                        "parsed_call": finish_call,
                        "generated_tokens": finish_tokens,
                        "prompt_tokens": finish_prompt_tokens,
                        "wall_seconds": round(time.perf_counter() - turn_started, 6),
                        "generation_protocol_detail": finish_detail,
                        "rendered_stage_prompts": finish_prompts,
                    })
                    attempt_record["tool_turns"].append({
                        "name": "finish", "parse_status": finish_status,
                        "generated_tokens": finish_tokens,
                        "prompt_tokens": finish_prompt_tokens,
                        "generation_protocol_detail": finish_detail,
                    })
                    attempt_record["outcome"] = (
                        "two_turn_tool_interaction_complete"
                        if finish_status == "complete_tool_call" and finish_call is not None
                        and finish_call["name"] == "finish"
                        else "finish_tool_call_not_accepted"
                    )
        attempt_record["wall_seconds"] = round(time.perf_counter() - attempt_started, 6)
        write_json(attempt / "attempt-summary.json", attempt_record)

    summaries = []
    for path in sorted(output.glob("attempt-*/attempt-summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for item in summaries:
        by_arm.setdefault(item["arm"], []).append(item)
    def attempt_metrics(value: dict[str, Any]) -> dict[str, Any]:
        attempt_dir = output / f"attempt-{value['attempt_index']:02d}-{value['arm']}"
        full_prompt_tokens = 0
        generated_prefill_tokens = 0
        measured_prefill_seconds = 0.0
        cache_build_seconds = 0.0
        cache_restore_seconds = 0.0
        for filename in ("turn-01-read-lines.json", "turn-02-finish.json"):
            path = attempt_dir / filename
            if not path.is_file():
                continue
            turn = json.loads(path.read_text(encoding="utf-8"))
            detail = turn["generation_protocol_detail"]
            stages = detail.get("stages", [])
            full_prompt_tokens += sum(int(stage["full_prompt_tokens"]) for stage in stages)
            generated_prefill_tokens += sum(int(stage["prefill_input_tokens"]) for stage in stages)
            measured_prefill_seconds += sum(
                float(stage["prompt_prefill_seconds"] or 0.0) for stage in stages
            )
            cache_restore_seconds += sum(
                float(stage.get("kv_cache_restore_wall_seconds", 0.0)) for stage in stages
            )
            shared_cache = detail.get("shared_prefix_cache")
            if shared_cache:
                generated_prefill_tokens += int(shared_cache["prefix_tokens"])
                cache_build_seconds += float(shared_cache["build_wall_seconds"])
        return {
            "full_prompt_tokens_across_stages": full_prompt_tokens,
            "amortized_prefill_tokens_including_shared_prefix": generated_prefill_tokens,
            "measured_prefill_seconds_including_shared_prefix_build": round(
                measured_prefill_seconds + cache_build_seconds, 6,
            ),
            "cache_build_seconds": round(cache_build_seconds, 6),
            "cache_restore_seconds": round(cache_restore_seconds, 6),
        }

    comparable: list[dict[str, Any]] = []
    for replicate in range(2):
        left = by_arm["schema_slots_v9"][replicate]
        right = by_arm["schema_slots_v10"][replicate]
        def outputs(value: dict[str, Any]) -> list[dict[str, Any]]:
            records = []
            attempt_dir = output / f"attempt-{value['attempt_index']:02d}-{value['arm']}"
            for filename in ("turn-01-read-lines.json", "turn-02-finish.json"):
                path = attempt_dir / filename
                if not path.is_file():
                    continue
                turn = json.loads(path.read_text(encoding="utf-8"))
                stages = turn["generation_protocol_detail"].get("stages", [])
                records.append({
                    "raw_model_output": turn["raw_model_output"],
                    "generated_token_ids": [stage["generated_token_ids"] for stage in stages],
                })
            return records
        comparable.append({
            "replicate": replicate + 1,
            "schema_slots_v9_outcome": left["outcome"],
            "schema_slots_v10_outcome": right["outcome"],
            "raw_turn_outputs_and_token_ids_equal": outputs(left) == outputs(right),
            "schema_slots_v9_wall_seconds": left["wall_seconds"],
            "schema_slots_v10_wall_seconds": right["wall_seconds"],
            "schema_slots_v9_metrics": attempt_metrics(left),
            "schema_slots_v10_metrics": attempt_metrics(right),
        })
    write_json(output / "comparison.json", {
        "model_load_seconds": round(model_load_seconds, 6),
        "order": arms,
        "paired_repetitions": comparable,
        "attempt_count": len(summaries),
        "outcomes": [item["outcome"] for item in summaries],
        "claim_boundary": "synthetic controller/prefill systems microbenchmark only; no external task repair or capability estimate",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
