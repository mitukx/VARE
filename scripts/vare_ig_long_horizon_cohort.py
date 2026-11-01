#!/usr/bin/env python3
"""Frozen Development cohort for real-model, multi-step coding-agent episodes.

The cohort uses synthetic miniature repositories and parent-owned function
oracles. It measures navigation and repair behavior; it is not held-out
capability evidence and makes no claim about real upstream repositories.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import long_horizon_eval as lh  # noqa: E402
from scripts import qwen3_function_repair_grpo as runner  # noqa: E402
from vare.isolated_candidate import FunctionCall, grade_calls  # noqa: E402


PROTOCOL_PATH = ROOT / "protocols/vare_ig_long_horizon_dev_v1.json"
OUTPUT_ROOT = ROOT / "results/vare-ig-long-horizon-v1"
SEEDS = (20261106, 20261107, 20261108, 20261109)
TARGET = "solution.py"
READABLE = ("README.md", "docs/API.md", "examples.py", "src/usage.py", TARGET)


def _call(case_id: str, args: Sequence[Any], expected: Any) -> FunctionCall:
    return FunctionCall(case_id, "solution", "solve", list(args), expected)


def _workspace_digest(files: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(sorted(files.items())), sort_keys=True,
                                    ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _task(task_id: str, issue: str, files: Mapping[str, str]) -> lh.EpisodeTask:
    return lh.EpisodeTask(
        task_id=task_id,
        repository=f"synthetic/{task_id.rsplit('.', 2)[-2]}",
        split="development",
        issue=issue,
        readable_files=READABLE,
        editable_files=(TARGET,),
        target_files=(TARGET,),
        synthetic=True,
        workspace_patch_sha256=_workspace_digest(files),
    )


FILES: dict[str, dict[str, str]] = {
    "vare_ig.lh_dev.layered_option_lookup.v1": {
        "README.md": "# Service configuration\n\nThe service resolves a setting from ordered configuration layers.\n",
        "docs/API.md": (
            "# Configuration lookup contract\n\n"
            "`solve(layers, path, default)` reads a dot-separated path in each layer, "
            "from lowest to highest precedence. The last layer that contains the complete "
            "path wins. A present value of `None` overrides lower layers. If no layer has "
            "the complete path, return `default`. An empty path is invalid and returns `default`.\n"
        ),
        "examples.py": (
            "# Public examples\n"
            "LAYERS = [{'db': {'timeout': 5}}, {'db': {'timeout': 8}}]\n"
            "EXPECTED_TIMEOUT = 8\n"
            "NULL_OVERRIDE = [{'db': {'token': 'old'}}, {'db': {'token': None}}]\n"
            "EXPECTED_TOKEN = None\n"
        ),
        "src/usage.py": "from solution import solve\nTIMEOUT = solve(load_layers(), 'db.timeout', 30)\n",
        TARGET: (
            "def solve(layers, path, default):\n"
            "    parts = path.split('.') if path else []\n"
            "    value = default\n"
            "    for layer in layers:\n"
            "        current = layer\n"
            "        for part in parts:\n"
            "            if not isinstance(current, dict) or part not in current:\n"
            "                break\n"
            "            current = current[part]\n"
            "        else:\n"
            "            if current is not None:\n"
            "                value = current\n"
            "    return value\n"
        ),
    },
    "vare_ig.lh_dev.lexical_course_planner.v1": {
        "README.md": "# Course planner\n\nThe CLI prints a valid course order with stable output.\n",
        "docs/API.md": (
            "# Course ordering contract\n\n"
            "`solve(courses, prerequisites)` receives pairs `[course, prerequisite]`. "
            "Return the lexicographically smallest valid topological order of unique courses, "
            "including isolated courses. Duplicate pairs are ignored. Return `None` when an "
            "endpoint is unknown or a cycle exists.\n"
        ),
        "examples.py": (
            "# Public examples\n"
            "COURSES = ['api', 'base', 'cli', 'db']\n"
            "PREREQUISITES = [['cli', 'base']]\n"
            "EXPECTED = ['api', 'base', 'cli', 'db']\n"
            "NEWLY_READY = ['d', 'c', 'b', 'a']; DEPENDENCY = [['a', 'c']]\n"
            "EXPECTED_NEWLY_READY = ['b', 'c', 'a', 'd']\n"
        ),
        "src/usage.py": "from solution import solve\nORDER = solve(load_courses(), load_prerequisites())\n",
        TARGET: (
            "def solve(courses, prerequisites):\n"
            "    nodes = sorted(set(courses))\n"
            "    known = set(nodes)\n"
            "    outgoing = {node: set() for node in nodes}\n"
            "    indegree = {node: 0 for node in nodes}\n"
            "    for course, prerequisite in prerequisites:\n"
            "        if course not in known or prerequisite not in known:\n"
            "            return None\n"
            "        if course not in outgoing[prerequisite]:\n"
            "            outgoing[prerequisite].add(course)\n"
            "            indegree[course] += 1\n"
            "    ready = [node for node in nodes if indegree[node] == 0]\n"
            "    result = []\n"
            "    while ready:\n"
            "        node = ready.pop(0)\n"
            "        result.append(node)\n"
            "        for child in sorted(outgoing[node]):\n"
            "            indegree[child] -= 1\n"
            "            if indegree[child] == 0:\n"
            "                ready.append(child)\n"
            "    return result if len(result) == len(nodes) else None\n"
        ),
    },
    "vare_ig.lh_dev.resource_window_coalescer.v1": {
        "README.md": "# Event ingest\n\nThe report groups overlapping reservations by resource.\n",
        "docs/API.md": (
            "# Window coalescing contract\n\n"
            "`solve(events)` accepts `[resource, start, end, event_id]` rows with integer "
            "closed intervals. Merge overlapping windows only for the same resource. Sort "
            "resources lexicographically and intervals by start; preserve event IDs in start "
            "order within each merged window. Return `[resource, start, end, ids]` rows. "
            "Touching at a single endpoint counts as overlap.\n"
        ),
        "examples.py": (
            "# Public examples\n"
            "EVENTS = [['room-b', 2, 5, 'b1'], ['room-a', 4, 7, 'a2'], "
            "['room-a', 1, 4, 'a1']]\n"
            "EXPECTED = [['room-a', 1, 7, ['a1', 'a2']], ['room-b', 2, 5, ['b1']]]\n"
            "TOUCHING = [['r', 1, 2, 'x'], ['r', 2, 3, 'y']]\n"
            "EXPECTED_TOUCHING = [['r', 1, 3, ['x', 'y']]]\n"
        ),
        "src/usage.py": "from solution import solve\nWINDOWS = solve(load_events())\n",
        TARGET: (
            "def solve(events):\n"
            "    rows = sorted(events, key=lambda row: (row[1], row[2], row[0]))\n"
            "    merged = []\n"
            "    for resource, start, end, event_id in rows:\n"
            "        if merged and start <= merged[-1][2]:\n"
            "            merged[-1][2] = max(merged[-1][2], end)\n"
            "            merged[-1][3].append(event_id)\n"
            "        else:\n"
            "            merged.append([resource, start, end, [event_id]])\n"
            "    return merged\n"
        ),
    },
    "vare_ig.lh_dev.json_pointer_lookup.v1": {
        "README.md": "# Metadata service\n\nClients use JSON Pointer paths to read nested metadata.\n",
        "docs/API.md": (
            "# JSON Pointer lookup contract\n\n"
            "`solve(document, pointer, default)` follows RFC 6901 object and array tokens. "
            "The empty pointer selects the entire document; nonempty pointers begin with `/`. "
            "Decode `~1` to `/` and `~0` to `~`; any other `~` escape is invalid. Array "
            "indices are canonical nonnegative decimals (`0` or no leading zero), and `-` "
            "is not a readable index. Return `default` for an invalid or missing path.\n"
        ),
        "examples.py": (
            "# Public examples\n"
            "DOC = {'meta': {'version': 3}, 'a/b': {'~name': 'prod'}}\n"
            "EXPECTED_VERSION = 3\n"
            "EXPECTED_ESCAPED = 'prod'  # pointer: /a~1b/~0name\n"
        ),
        "src/usage.py": "from solution import solve\nVERSION = solve(load_metadata(), '/meta/version', None)\n",
        TARGET: (
            "def solve(document, pointer, default):\n"
            "    if pointer == '':\n"
            "        return document\n"
            "    if not pointer.startswith('/'):\n"
            "        return default\n"
            "    value = document\n"
            "    for token in pointer[1:].split('/'):\n"
            "        if isinstance(value, dict) and token in value:\n"
            "            value = value[token]\n"
            "        elif isinstance(value, list) and token.isdigit() and int(token) < len(value):\n"
            "            value = value[int(token)]\n"
            "        else:\n"
            "            return default\n"
            "    return value\n"
        ),
    },
}


def _calls() -> dict[str, tuple[FunctionCall, ...]]:
    return {
        "vare_ig.lh_dev.layered_option_lookup.v1": (
            _call("high_precedence", ([{"db": {"timeout": 5}}, {"db": {"timeout": 8}}], "db.timeout", 30), 8),
            _call("null_override", ([{"db": {"token": "old"}}, {"db": {"token": None}}], "db.token", "fallback"), None),
            _call("missing_higher_falls_back", ([{"db": {"port": 5432}}, {"db": {"timeout": 8}}], "db.port", 80), 5432),
            _call("missing_everywhere", ([{"db": {}}], "db.host", "localhost"), "localhost"),
            _call("empty_path", ([{"db": 1}], "", "invalid"), "invalid"),
        ),
        "vare_ig.lh_dev.lexical_course_planner.v1": (
            _call("public_order", (["api", "base", "cli", "db"], [["cli", "base"]]), ["api", "base", "cli", "db"]),
            _call("new_ready_node_precedes_existing", (["d", "c", "b", "a"], [["a", "c"]]), ["b", "c", "a", "d"]),
            _call("duplicate_course_and_edge", (["a", "b", "a"], [["b", "a"], ["b", "a"]]), ["a", "b"]),
            _call("duplicate_edge", (["a", "b"], [["b", "a"], ["b", "a"]]), ["a", "b"]),
            _call("cycle", (["a", "b"], [["b", "a"], ["a", "b"]]), None),
            _call("unknown", (["a"], [["b", "a"]]), None),
        ),
        "vare_ig.lh_dev.resource_window_coalescer.v1": (
            _call("public_resource_partition", ([
                ["room-b", 2, 5, "b1"], ["room-a", 4, 7, "a2"], ["room-a", 1, 4, "a1"],
            ],), [["room-a", 1, 7, ["a1", "a2"]], ["room-b", 2, 5, ["b1"]]]),
            _call("public_touching", ([["r", 1, 2, "x"], ["r", 2, 3, "y"]],), [["r", 1, 3, ["x", "y"]]]),
            _call("separated", ([["r", 1, 2, "x"], ["r", 4, 5, "y"]],), [["r", 1, 2, ["x"]], ["r", 4, 5, ["y"]]]),
            _call("same_time_different_resources", ([["b", 1, 3, "b1"], ["a", 1, 3, "a1"]],),
                  [["a", 1, 3, ["a1"]], ["b", 1, 3, ["b1"]]]),
            _call("empty", ([],), []),
        ),
        "vare_ig.lh_dev.json_pointer_lookup.v1": (
            _call("public_nested", ({"meta": {"version": 3}}, "/meta/version", None), 3),
            _call("public_escapes", ({"a/b": {"~name": "prod"}}, "/a~1b/~0name", None), "prod"),
            _call("array_index", ({"items": [{"id": 7}, {"id": 12}]}, "/items/1/id", 0), 12),
            _call("invalid_escape", ({"x~2y": 1}, "/x~2y", "fallback"), "fallback"),
            _call("leading_zero_index", (["a", "b"], "/01", "fallback"), "fallback"),
            _call("missing", ({"a": {}}, "/a/b", "fallback"), "fallback"),
        ),
    }


CALLS = _calls()
TASKS = tuple(
    _task(
        task_id,
        {
            "vare_ig.lh_dev.layered_option_lookup.v1":
                "A configuration override is ignored when its value is null. Inspect the docs and caller, then repair lookup behavior without changing its public signature.",
            "vare_ig.lh_dev.lexical_course_planner.v1":
                "The planner sometimes emits courses in an order that changes between runs. Inspect the contract and caller, then fix the ordering while preserving cycle and unknown-course behavior.",
            "vare_ig.lh_dev.resource_window_coalescer.v1":
                "The reservation report combines overlapping events belonging to different rooms. Inspect the data flow and contract, then repair the grouping and output ordering.",
            "vare_ig.lh_dev.json_pointer_lookup.v1":
                "Metadata keys containing `/` or `~` cannot be read, and malformed array positions are accepted. Inspect the interface contract and fix path resolution.",
        }[task_id],
        FILES[task_id],
    )
    for task_id in FILES
)
TASK_BY_ID = {task.task_id: task for task in TASKS}
CALLS_BY_ID = CALLS


REFERENCES: dict[str, str] = {
    "vare_ig.lh_dev.layered_option_lookup.v1": (
        "_MISSING = object()\n"
        "def solve(layers, path, default):\n"
        "    if not path:\n"
        "        return default\n"
        "    value = _MISSING\n"
        "    for layer in layers:\n"
        "        current = layer\n"
        "        for part in path.split('.'):\n"
        "            if not isinstance(current, dict) or part not in current:\n"
        "                break\n"
        "            current = current[part]\n"
        "        else:\n"
        "            value = current\n"
        "    return default if value is _MISSING else value\n"
    ),
    "vare_ig.lh_dev.lexical_course_planner.v1": (
        "def solve(courses, prerequisites):\n"
        "    nodes = sorted(set(courses))\n"
        "    known = set(nodes)\n"
        "    outgoing = {node: set() for node in nodes}\n"
        "    indegree = {node: 0 for node in nodes}\n"
        "    for course, prerequisite in prerequisites:\n"
        "        if course not in known or prerequisite not in known:\n"
        "            return None\n"
        "        if course not in outgoing[prerequisite]:\n"
        "            outgoing[prerequisite].add(course)\n"
        "            indegree[course] += 1\n"
        "    ready = sorted(node for node in nodes if indegree[node] == 0)\n"
        "    result = []\n"
        "    while ready:\n"
        "        node = ready.pop(0)\n"
        "        result.append(node)\n"
        "        for child in sorted(outgoing[node]):\n"
        "            indegree[child] -= 1\n"
        "            if indegree[child] == 0:\n"
        "                ready.append(child)\n"
        "                ready.sort()\n"
        "    return result if len(result) == len(nodes) else None\n"
    ),
    "vare_ig.lh_dev.resource_window_coalescer.v1": (
        "from collections import defaultdict\n"
        "def solve(events):\n"
        "    by_resource = defaultdict(list)\n"
        "    for resource, start, end, event_id in events:\n"
        "        by_resource[resource].append((start, end, event_id))\n"
        "    output = []\n"
        "    for resource in sorted(by_resource):\n"
        "        rows = sorted(by_resource[resource], key=lambda row: row[0])\n"
        "        merged = []\n"
        "        for start, end, event_id in rows:\n"
        "            if merged and start <= merged[-1][1]:\n"
        "                merged[-1][1] = max(merged[-1][1], end)\n"
        "                merged[-1][2].append(event_id)\n"
        "            else:\n"
        "                merged.append([start, end, [event_id]])\n"
        "        output.extend([[resource, start, end, ids] for start, end, ids in merged])\n"
        "    return output\n"
    ),
    "vare_ig.lh_dev.json_pointer_lookup.v1": (
        "def solve(document, pointer, default):\n"
        "    if pointer == '':\n"
        "        return document\n"
        "    if not isinstance(pointer, str) or not pointer.startswith('/'):\n"
        "        return default\n"
        "    value = document\n"
        "    for raw in pointer[1:].split('/'):\n"
        "        i = 0\n"
        "        token = ''\n"
        "        while i < len(raw):\n"
        "            if raw[i] != '~':\n"
        "                token += raw[i]; i += 1; continue\n"
        "            if i + 1 >= len(raw) or raw[i + 1] not in '01':\n"
        "                return default\n"
        "            token += '/' if raw[i + 1] == '1' else '~'\n"
        "            i += 2\n"
        "        if isinstance(value, dict):\n"
        "            if token not in value: return default\n"
        "            value = value[token]\n"
        "        elif isinstance(value, list):\n"
        "            if token == '0': index = 0\n"
        "            elif token and token[0] in '123456789' and all('0' <= ch <= '9' for ch in token): index = int(token)\n"
        "            else: return default\n"
        "            if index >= len(value): return default\n"
        "            value = value[index]\n"
        "        else:\n"
        "            return default\n"
        "    return value\n"
    ),
}

MUTANTS: dict[str, str] = {
    "vare_ig.lh_dev.layered_option_lookup.v1": REFERENCES["vare_ig.lh_dev.layered_option_lookup.v1"].replace(
        "value = current\n", "if current is not None: value = current\n"
    ),
    "vare_ig.lh_dev.lexical_course_planner.v1": FILES["vare_ig.lh_dev.lexical_course_planner.v1"][TARGET],
    "vare_ig.lh_dev.resource_window_coalescer.v1": (
        "def solve(events):\n"
        "    rows = sorted(events, key=lambda row: row[1])\n"
        "    merged = []\n"
        "    for resource, start, end, event_id in rows:\n"
        "        if merged and start <= merged[-1][2]:\n"
        "            merged[-1][2] = max(merged[-1][2], end)\n"
        "            merged[-1][3].append(event_id)\n"
        "        else:\n"
        "            merged.append([resource, start, end, [event_id]])\n"
        "    return merged\n"
    ),
    "vare_ig.lh_dev.json_pointer_lookup.v1": REFERENCES["vare_ig.lh_dev.json_pointer_lookup.v1"].replace(
        "        i = 0\n", "        i = 0\n"
    ).replace(
        "            token += '/' if raw[i + 1] == '1' else '~'\n", "            token += raw[i + 1]\n"
    ),
}


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "scripts/vare_ig_long_horizon_cohort.py",
        ROOT / "scripts/long_horizon_eval.py",
        ROOT / "scripts/run_local_agent.py",
        ROOT / "scripts/qwen3_function_repair_grpo.py",
        ROOT / "src/vare/isolated_candidate.py",
        ROOT / "src/vare/container_isolation.py",
        ROOT / "src/vare/trajectory_store.py",
        ROOT / "src/vare/learning_pipeline/policy.py",
    )
    return {str(path.relative_to(ROOT)): _source_sha(path) for path in paths}


def grader_for(task_id: str, *, hidden_only: bool = True) -> lh.FunctionOracleGrader:
    calls = CALLS_BY_ID[task_id][2:] if hidden_only else CALLS_BY_ID[task_id]
    return lh.FunctionOracleGrader(task_id=task_id, target_file=TARGET, calls=calls,
                                   image=runner.CONTAINER_IMAGE,
                                   timeout_s=runner.EVAL_TIMEOUT_SECONDS,
                                   memory_mb=runner.EVAL_MEMORY_MB)


def public_test_for(task_id: str):
    public_calls = CALLS_BY_ID[task_id][:2]

    def run(workspace: Path) -> Mapping[str, Any]:
        source = (workspace / TARGET).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="vare-ig-public-tests-") as directory:
            subject = Path(directory)
            (subject / TARGET).write_text(source, encoding="utf-8")
            report = grade_calls(subject, public_calls, task_kind="correctness", image=runner.CONTAINER_IMAGE,
                                timeout_s=runner.EVAL_TIMEOUT_SECONDS, memory_mb=runner.EVAL_MEMORY_MB)
        return {"passed": bool(report.passed), "case_count": len(public_calls),
                "summary": "public examples passed" if report.passed else "one or more public examples failed"}

    return run


def workspace_factory(task_id: str):
    files = FILES[task_id]

    def build(target: Path) -> None:
        for name, source in files.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        lh.git_snapshot(target)

    return build


def protocol_payload() -> dict[str, Any]:
    controller = lh.ControllerConfig(
        name="vare-ig-long-horizon-full-v1", max_turns=14,
        max_new_tokens_per_turn=512, max_total_new_tokens=4096,
        max_calls_per_turn=4, wall_seconds=1800.0, temperature=0.0,
        top_p=1.0, reveal_target=False, normalize_tool_calls=True,
        max_no_tool_retries=2, recover_bare_edits=True,
        auto_public_tests_after_edit=True, finish_gate=True,
    )
    return {
        "schema": "vare_ig_long_horizon_dev_protocol_v1",
        "protocol_id": "vare-ig-qwen3-4b-long-horizon-development-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistered_before_model_inference": True,
        "model": {"id": runner.MODEL_ID, "revision": runner.MODEL_REVISION,
                  "local_only": True, "inference": "MLX/Metal"},
        "split": "development",
        "claim_boundary": "Four synthetic Development repository-repair tasks; descriptive agent-behavior measurement, not held-out capability evidence.",
        "controller": {"config": controller.to_dict(), "sha256": controller.sha256()},
        "execution": {
            "task_ids": [task.task_id for task in TASKS],
            "task_sha256": {task.task_id: task.sha256() for task in TASKS},
            "workspace_sha256": {task_id: _workspace_digest(files) for task_id, files in FILES.items()},
            "grader_config_sha256": {task.task_id: grader_for(task.task_id).config_sha256() for task in TASKS},
            "public_case_ids": {task_id: [call.case_id for call in CALLS_BY_ID[task_id][:2]] for task_id in CALLS_BY_ID},
            "final_case_ids": {task_id: [call.case_id for call in CALLS_BY_ID[task_id][2:]] for task_id in CALLS_BY_ID},
            "seeds_by_task": dict(zip([task.task_id for task in TASKS], SEEDS, strict=True)),
            "same_task_seed_for_base_and_trained": True,
            "controller_arm": "fixed across policy checkpoints",
        },
        "budgets": {"max_turns": controller.max_turns,
                    "max_new_tokens_per_turn": controller.max_new_tokens_per_turn,
                    "max_total_new_tokens": controller.max_total_new_tokens,
                    "max_tool_calls_per_turn": controller.max_calls_per_turn,
                    "max_wall_seconds_per_episode": controller.wall_seconds,
                    "temperature": controller.temperature, "top_p": controller.top_p,
                    "max_prompt_tokens": 8192,
                    "public_tests_after_successful_edit": True,
                    "public_test_cases_per_task": 2,
                    "hidden_parent_grade_cases_per_task": "3 or 4, disjoint from public examples"},
        "source_hashes": source_hashes(),
        "grader": {"schema": "vare_function_oracle_v1", "image": runner.CONTAINER_IMAGE,
                   "network_disabled": True, "expected_values_parent_owned": True,
                   "candidate_pytest_authority": False},
        "resources": {"one_model_process_at_a_time": True,
                      "preflight": "existing Qwen runner memory, swap, and disk checks",
                      "max_phase_seconds": runner.MAX_RUN_SECONDS},
        "study_questions": [
            "How often does the real policy inspect more than one context file before editing?",
            "How often are failed tool actions or public-test failures recovered from?",
            "What fraction of Development tasks pass the independent parent oracle?",
            "Does a genuinely trained adapter change these behaviors under the same controller and task seeds?",
        ],
    }


def write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {path}")
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(data, encoding="utf-8")
    temp.replace(path)


def freeze() -> None:
    payload = protocol_payload()
    write_once(PROTOCOL_PATH, payload)
    print(json.dumps({"protocol": str(PROTOCOL_PATH),
                      "sha256": _source_sha(PROTOCOL_PATH),
                      "tasks": payload["execution"]["task_ids"]}, sort_keys=True))


def validate_protocol(path: Path) -> tuple[dict[str, Any], lh.ControllerConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    current = protocol_payload()
    for key in ("schema", "protocol_id", "preregistered_before_model_inference", "model", "split",
                "claim_boundary", "controller", "execution", "budgets", "source_hashes", "grader", "resources"):
        if payload.get(key) != current.get(key):
            raise RuntimeError(f"long-horizon protocol mismatch: {key}")
    if payload.get("preregistered_before_model_inference") is not True:
        raise RuntimeError("the Development cohort must be frozen before Qwen inference")
    return payload, lh.ControllerConfig(**payload["controller"]["config"])


class _GuardedPolicy:
    def __init__(self, policy, run: Path, started: float):
        self._policy, self.run, self.started = policy, run, started
        self.policy_id = policy.policy_id
        self.synthetic = False

    def generate(self, *args, **kwargs):
        runner.resource_guard(self.run, self.started)
        result = self._policy.generate(*args, **kwargs)
        runner.resource_guard(self.run, self.started)
        return result

    def peak_memory_bytes(self):
        return self._policy.peak_memory_bytes()

    def close(self):
        self._policy.close()


def run(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.expanduser().resolve()
    protocol, controller = validate_protocol(protocol_path)
    if args.arm == "base" and args.adapter is not None:
        raise ValueError("base arm must not receive an adapter")
    if args.arm == "trained" and args.adapter is None:
        raise ValueError("trained arm requires an adapter produced by the real VARE-IG learner")
    if args.arm == "trained" and args.training_run is None:
        raise ValueError("trained arm requires --training-run with the learner's final summary")
    if args.adapter is not None and not args.adapter.is_dir():
        raise FileNotFoundError(args.adapter)
    adapter_digest = runner.sha256_directory(args.adapter) if args.adapter else None
    training_summary_sha = None
    if args.training_run is not None:
        training_summary_path = args.training_run.expanduser().resolve() / "final-summary.json"
        training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
        training_summary_sha = _source_sha(training_summary_path)
        if args.arm != "trained":
            raise ValueError("--training-run applies only to the trained arm")
        if Path(training_summary.get("adapter_path", "")).expanduser().resolve() != args.adapter.expanduser().resolve():
            raise ValueError("adapter path does not match the VARE learner's final summary")
        if training_summary.get("adapter_sha256") != adapter_digest:
            raise ValueError("adapter digest differs from the VARE learner's final summary")
        if training_summary.get("adapter_reload_check", {}).get("response_exact_match") is not True:
            raise ValueError("VARE learner did not verify adapter reload equivalence")
        initial_update = training_summary.get("training_update_1", {}).get("update_applied") is True
        recovery_update = training_summary.get("checkpoint_replay", {}).get("update_applied_both") is True
        if not (initial_update or recovery_update):
            raise ValueError("no genuine optimizer update is recorded; trained-policy evaluation is ineligible")
    run_root = args.output.expanduser().resolve() / f"{args.arm}-{controller.name}"
    run_root.mkdir(parents=True, exist_ok=True)
    if (run_root / "long-horizon-summary.json").exists():
        raise FileExistsError(f"a completed episode summary already exists: {run_root}")
    protocol_sha = _source_sha(protocol_path)
    phase_started = time.monotonic()
    try:
        preflight = runner.preflight(run_root, "vare-ig-long-horizon")
    except runner.ResourceStop as exc:
        runner.atomic_json(run_root / "resource-stop-vare-ig-long-horizon.json", {
            "reason": str(exc), "resource": runner.resource_snapshot(run_root), "time": time.time()
        })
        print(json.dumps({"resource_stop": str(exc)}, sort_keys=True), flush=True)
        return 75
    policy_digest = lh.sha256_json({"model": runner.MODEL_ID, "revision": runner.MODEL_REVISION,
                                    "adapter_sha256": adapter_digest})
    policy_version = 0 if args.arm == "base" else 1
    checkpoint = lh.PolicyCheckpoint(args.arm, policy_version, policy_digest)
    raw_policy = lh.mlx_policy(runner.MODEL_ID, str(args.adapter) if args.adapter else None,
                               revision=runner.MODEL_REVISION, max_prompt_tokens=8192)
    runner.resource_guard(run_root, phase_started)
    policy = _GuardedPolicy(raw_policy, run_root, phase_started)
    graders = {task.task_id: grader_for(task.task_id) for task in TASKS}
    harness = lh.EpisodeHarness(
        run_root, lh.PolicyArm(args.arm, policy, checkpoint), controller,
        factories={task.task_id: workspace_factory(task.task_id) for task in TASKS},
        public_tests={task.task_id: public_test_for(task.task_id) for task in TASKS},
        grader_for=lambda task: graders[task.task_id],
        protocol_sha256=protocol_sha,
    )
    specs = [lh.EpisodeSpec(task, protocol["execution"]["seeds_by_task"][task.task_id]) for task in TASKS]
    harness.enroll(specs)
    try:
        harness.run(wait_for_expired=True)
    except runner.ResourceStop as exc:
        runner.atomic_json(run_root / "resource-stop-vare-ig-long-horizon.json", {
            "reason": str(exc), "resource": runner.resource_snapshot(run_root), "time": time.time()
        })
        policy.close()
        print(json.dumps({"resource_stop": str(exc)}, sort_keys=True), flush=True)
        return 75
    metrics = lh.run_metrics(run_root)
    summary = {
        "schema": "vare_ig_long_horizon_run_summary_v1",
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha,
        "arm": args.arm, "model": protocol["model"],
        "adapter_sha256": adapter_digest, "policy_checkpoint": asdict(checkpoint),
        "training_run_summary_sha256": training_summary_sha,
        "controller": controller.to_dict(), "task_count": len(TASKS),
        "task_ids": [task.task_id for task in TASKS],
        "metrics": metrics,
        "resource_start": preflight, "resource_end": runner.resource_snapshot(run_root),
        "wall_seconds": round(time.monotonic() - phase_started, 3),
        "claim_boundary": protocol["claim_boundary"],
    }
    runner.atomic_json(run_root / "long-horizon-summary.json", summary)
    policy.close()
    print(json.dumps({"summary": str(run_root / "long-horizon-summary.json"),
                      "episodes": len(metrics.get("episodes", [])),
                      "success_rate": metrics.get("aggregate", {}).get("final_success_rate"),
                      "wall_seconds": summary["wall_seconds"]}, sort_keys=True), flush=True)
    return 0


def audit_oracles(output: Path) -> int:
    rows = []
    for task in TASKS:
        task_id = task.task_id
        for label, sources in (("reference", REFERENCES), ("mutant", MUTANTS)):
            with tempfile.TemporaryDirectory(prefix="vare-ig-lh-oracle-audit-") as directory:
                subject = Path(directory)
                (subject / TARGET).write_text(sources[task_id], encoding="utf-8")
                report = grade_calls(subject, CALLS_BY_ID[task_id], task_kind="correctness",
                                     image=runner.CONTAINER_IMAGE, timeout_s=runner.EVAL_TIMEOUT_SECONDS,
                                     memory_mb=runner.EVAL_MEMORY_MB)
            rows.append({"task_id": task_id, "control": label, "passed": report.passed,
                         "failed_cases": [case.case_id for case in report.cases if not case.passed]})
    result = {"schema": "vare_ig_long_horizon_oracle_audit_v1",
              "all_references_passed": all(row["passed"] for row in rows if row["control"] == "reference"),
              "all_mutants_rejected": all(not row["passed"] for row in rows if row["control"] == "mutant"),
              "rows": rows,
              "claim_boundary": "Synthetic Development function contracts only; Docker isolation and parent-owned expected values."}
    runner.atomic_json(output, result)
    print(json.dumps({"audit": str(output), "all_references_passed": result["all_references_passed"],
                      "all_mutants_rejected": result["all_mutants_rejected"]}, sort_keys=True))
    return 0 if result["all_references_passed"] and result["all_mutants_rejected"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="write the Development protocol once, before model inference")
    execute = sub.add_parser("run", help="execute one real Qwen policy arm")
    execute.add_argument("--arm", choices=("base", "trained"), required=True)
    execute.add_argument("--adapter", type=Path)
    execute.add_argument("--training-run", type=Path,
                         help="VARE learner run directory; required for trained arm provenance")
    execute.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    execute.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    audit = sub.add_parser("audit-oracles", help="run reference and incorrect-mutant controls")
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        freeze()
        return 0
    if args.command == "audit-oracles":
        return audit_oracles(args.output.expanduser().resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
