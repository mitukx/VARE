#!/usr/bin/env python3
"""Token-count fixed tool-call fixtures for the coding-agent edit interfaces."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any


LEASE = "a1b2c3d4e5f6"
FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "name": "unique_single_line",
        "source": "def amount(x):\n    return x + 1\n",
        "start": 2,
        "end": 2,
        "query": "return x + 1",
        "new_text": "    return x * 2\n",
        "legacy_old_text": "    return x + 1\n",
    },
    {
        "name": "repeated_line_needs_context",
        "source": (
            "def score(x):\n"
            "    if x < 0:\n"
            "        return 0\n"
            "    return x\n"
            "\n"
            "def fallback(x):\n"
            "    return x\n"
        ),
        "start": 4,
        "end": 4,
        "query": "return x",
        "new_text": "    return max(x, 0)\n",
        "legacy_old_text": "def score(x):\n    if x < 0:\n        return 0\n    return x\n",
    },
    {
        "name": "three_line_rewrite",
        "source": (
            "def parse_count(raw):\n"
            "    value = int(raw)\n"
            "    return value\n"
            "\n"
            "def unrelated():\n"
            "    return 0\n"
        ),
        "start": 2,
        "end": 3,
        "query": "int(raw)",
        "new_text": "    value = int(raw.strip())\n    return max(value, 0)\n",
        "legacy_old_text": "    value = int(raw)\n    return value\n",
    },
    {
        "name": "six_line_span_rewrite",
        "source": (
            "def summarize(values):\n"
            "    if not values:\n"
            "        return {\"count\": 0, \"total\": 0}\n"
            "    count = len(values)\n"
            "    total = sum(values)\n"
            "    mean = total / count\n"
            "    return {\"count\": count, \"total\": total, \"mean\": mean}\n"
            "\n"
            "def other():\n"
            "    return None\n"
        ),
        "start": 2,
        "end": 7,
        "query": "mean = total / count",
        "new_text": (
            "    count = len(values)\n"
            "    total = sum(values)\n"
            "    mean = total / count if count else 0\n"
            "    return {\"count\": count, \"total\": total, \"mean\": mean}\n"
        ),
        "legacy_old_text": (
            "    if not values:\n"
            "        return {\"count\": 0, \"total\": 0}\n"
            "    count = len(values)\n"
            "    total = sum(values)\n"
            "    mean = total / count\n"
            "    return {\"count\": count, \"total\": total, \"mean\": mean}\n"
        ),
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_output(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"<tool_call>{payload}</tool_call>"


def _validate_fixture(fixture: dict[str, Any]) -> None:
    lines = fixture["source"].splitlines(keepends=True)
    old_range = "".join(lines[fixture["start"] - 1:fixture["end"]])
    if not old_range or fixture["legacy_old_text"].count(old_range) < 1:
        raise ValueError(f"{fixture['name']}: range is absent from legacy context")
    if fixture["source"].count(fixture["legacy_old_text"]) != 1:
        raise ValueError(f"{fixture['name']}: legacy old_text is not unique")
    if fixture["new_text"] == old_range:
        raise ValueError(f"{fixture['name']}: fixture edit is a no-op")


def calibrate(tokenizer: Any, model_dir: Path) -> dict[str, Any]:
    import tokenizers
    import transformers

    tokenizer_files = {
        name: sha256(model_dir / name)
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        )
        if (model_dir / name).is_file()
    }
    records = []
    for fixture in FIXTURES:
        _validate_fixture(fixture)
        path = "src/fixture.py"
        legacy_calls = [
            tool_output("list_files", {}),
            tool_output("search_text", {"query": fixture["query"]}),
            tool_output("read_file_range", {
                "path": path, "start_line": fixture["start"], "end_line": fixture["end"],
            }),
            tool_output("replace_text", {
                "path": path,
                "old_text": fixture["legacy_old_text"],
                "new_text": fixture["new_text"],
            }),
        ]
        lease_calls = [
            tool_output("list_files", {}),
            tool_output("search_text", {"query": fixture["query"]}),
            tool_output("read_lines", {
                "path": path, "start": fixture["start"], "end": fixture["end"],
            }),
            tool_output("replace_lines", {"lease": LEASE, "text": fixture["new_text"]}),
        ]
        def count_tokens(outputs: list[str]) -> list[int]:
            return [
                len(tokenizer.encode(output, add_special_tokens=False)) for output in outputs
            ]

        legacy_counts = count_tokens(legacy_calls)
        lease_counts = count_tokens(lease_calls)
        records.append({
            "fixture": fixture,
            "structured_replace_v1": {
                "tool_output_tokens_by_turn": legacy_counts,
                "total_tool_output_tokens": sum(legacy_counts),
                "edit_action_tokens": legacy_counts[-1],
                "serialized_tool_outputs": legacy_calls,
            },
            "lease_bound_lines_v1": {
                "tool_output_tokens_by_turn": lease_counts,
                "total_tool_output_tokens": sum(lease_counts),
                "edit_action_tokens": lease_counts[-1],
                "serialized_tool_outputs": lease_calls,
            },
        })
    maximum_edit_action = max(
        record[interface]["edit_action_tokens"]
        for record in records
        for interface in ("structured_replace_v1", "lease_bound_lines_v1")
    )
    # Reserve 2.5x the largest independent fixture edit, rounded to 64 tokens.
    per_turn_budget = max(256, math.ceil(maximum_edit_action * 2.5 / 64) * 64)
    return {
        "schema_version": 1,
        "calibration_kind": "tokenizer_only_no_model_inference",
        "model_repository": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "model_snapshot": model_dir.name,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "tokenizer_files_sha256": tokenizer_files,
        "token_count_rule": "len(tokenizer.encode(raw_tool_output, add_special_tokens=False))",
        "tool_call_envelope": "<tool_call>{compact JSON}</tool_call>",
        "lease_token": LEASE,
        "fixtures": records,
        "aggregate": {
            "fixture_count": len(records),
            "structured_replace_total_tool_output_tokens": sum(
                record["structured_replace_v1"]["total_tool_output_tokens"] for record in records
            ),
            "lease_bound_total_tool_output_tokens": sum(
                record["lease_bound_lines_v1"]["total_tool_output_tokens"] for record in records
            ),
            "structured_replace_max_edit_action_tokens": max(
                record["structured_replace_v1"]["edit_action_tokens"] for record in records
            ),
            "lease_bound_max_edit_action_tokens": max(
                record["lease_bound_lines_v1"]["edit_action_tokens"] for record in records
            ),
            "maximum_edit_action_tokens_across_both_interfaces": maximum_edit_action,
            "per_turn_token_budget_formula": "max(256, ceil(max_fixture_edit_action_tokens * 2.5 / 64) * 64)",
            "per_turn_token_budget": per_turn_budget,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve(strict=True)
    if args.output.exists():
        parser.error(f"refusing to overwrite calibration record: {args.output}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    record = calibrate(tokenizer, model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        parser.error(f"refusing to overwrite calibration record: {args.output}")
    print(json.dumps(record["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
