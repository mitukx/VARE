#!/usr/bin/env python3
"""Run a prospective full-trajectory GRPO continuation on a frozen Train task."""
from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


def find_experiment_root() -> Path:
    configured = os.environ.get("VARE_EXPERIMENT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "outputs/vare-rsi-heldout-qualification-v2-20261105").is_dir()
            and (candidate / "work/vare-rsi-qwen3-4b-mlx-20261106").is_dir()
        ):
            return candidate
    raise RuntimeError("set VARE_EXPERIMENT_ROOT to the workspace containing frozen task assets")


ROOT = find_experiment_root()
WORK = ROOT / "work/vare-rsi-qwen3-4b-mlx-20261106"
OUTPUTS = ROOT / "outputs/vare-rsi-heldout-qualification-v2-20261105"
RUN_DIR = OUTPUTS / "runs/vare-pydantic-qwen3-4b-train14138-attempt-010"
ASSIGNMENT_PATH = OUTPUTS / "train-dev-assignment-v19-pydantic-14138-scope-guard.json"
RUN_LOCK_PATH = OUTPUTS / "real-model-run-v27-14138-scope-guard-grpo-group.lock.json"
MODEL_PATH = WORK / "model"
MODEL_MANIFEST_PATH = WORK / "model-manifest.json"
INTEGRATION = ROOT / "work/VARE-rsi-integration-20261105"
GRPO_PATH = INTEGRATION / "src/vare/grpo_mlx.py"
CONTAINER_RUNNER = ROOT / "work/container_pytest_runner.py"
TRAIN_TASK_ID = "pydantic-14138-direct-builtin-scalar-union-schema"
MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
BASE_MODEL_ID = f"{MODEL_ID}@{MODEL_REVISION}"
BASE_SEED = 20261106
TEMPERATURE = 0.8
GROUP_SIZE = 8
NEW_ROLLOUT_COUNT = 8
NEW_ROLLOUT_START_INDEX = 48
MAX_TURNS = 16
MAX_TOTAL_TOKENS = 4096
MAX_ACTION_TOKENS = 512
MAX_PROMPT_TOKENS = 4096
MAX_WALL_SECONDS = 900
READ_FILE_PAGE_LINES = 20
TASK_SCOPE_GUARD_TASK_ID = "pydantic-14138-direct-builtin-scalar-union-schema"
TOOL_ADAPTER_VERSION = "mlx-local-coding-agent-tool-adapter-v7-union-loop-scope"
DEV_TEMPERATURE = 0.0
LORA = {"num_layers": 8, "rank": 8, "scale": 20.0, "dropout": 0.0}
OPTIMIZER_CONFIG = {"name": "mlx.optimizers.Adam", "learning_rate": 1e-5}
GIB = 1024**3

TASK_ARM_DIRS = {
    "pydantic-14195-model-config-annotation-non-evaluation":
        ROOT / "work/fresh-train-dev/task-arms-v10/pydantic-14195/baseline",
    "pydantic-14103-ignore-internal-dataclass-extras":
        ROOT / "work/fresh-train-dev/task-arms-v7/pydantic-14103/baseline",
    "pydantic-14138-direct-builtin-scalar-union-schema":
        ROOT / "work/fresh-train-dev/task-arms-v3/pydantic-14138/baseline",
    "pydantic-13997-explicit-json-schema-by-alias-override":
        ROOT / "work/fresh-train-dev/task-arms-v3/pydantic-13997-correct/baseline",
}
TASK_ENV_LOCK_OVERRIDES = {}

SYSTEM_PROMPT = (
    "You are an autonomous coding agent in a clean constrained checkout. Implement the "
    "requested change with a small correct patch. Tests are unavailable. You may inspect "
    "production source and edit only the exact path or paths listed in the task. Search is "
    "restricted to those editable paths. Start with an exact nested JSON tool call whose "
    "arguments value is always an object, never a string. Read only short relevant source "
    "windows. Never patch source lines you have not read. Use standard unified diff syntax "
    "only, with exact file headers and accurate @@ hunk counts; never use context-diff syntax. "
    "If apply_patch fails, inspect its error and the source context, then submit a different "
    "smaller diff. After a successful patch, compile_file and show_changes before finishing. "
    "Do not claim changes you did not make. On every turn return exactly one JSON object with "
    "keys tool and arguments, no Markdown or prose. Use one tool action per assistant turn. "
    "Available tools: search_code, read_file, apply_patch, compile_file, show_changes, finish. "
    "Do not create files, access tests, or use external services."
)

TOOL_HELP = (
    "Task: {prompt}\n"
    "Editable production source path(s): {paths}\n\n"
    "Return one JSON object with keys tool and arguments. The arguments value must always be "
    "a JSON object, never a string. Use real values; do not copy placeholders. Example search "
    "call: {{\"tool\":\"search_code\",\"arguments\":{{\"query\":\"get_model_type_hints\"}}}}.\n"
    "search_code: query must be an exact identifier or expression present in an editable file. "
    "Search is already scoped to the editable path; do not add a paths member.\n"
    "read_file: path must be exactly one editable path; start_line and end_line are positive "
    "source line numbers. Each response returns at most 20 lines. For a request wider than "
    "20 lines, the reader tracks lines already returned from that file and gives the next "
    "contiguous unread page, even when the new request overlaps or widens an earlier one. "
    "Use the actual returned start_line/end_line and next_start_line; do not request already "
    "seen lines again. If an oversized range is fully read already, the first 20 lines are "
    "replayed with an already-read marker so source context survives compaction. Requests of "
    "20 or fewer lines may be reread for patch context.\n"
    "apply_patch arguments must be a JSON object with path and patch string fields. For example: "
    "{\"tool\":\"apply_patch\",\"arguments\":{\"path\":\"pydantic/example.py\","
    "\"patch\":\"--- a/pydantic/example.py\\n+++ b/pydantic/example.py\\n"
    "@@ -10,1 +10,1 @@\\n-old\\n+new\\n\"}}. Put the complete unified diff under patch; "
    "include --- a/path, +++ b/path, and accurate @@ hunk headers with exact copied context. "
    "Do not use hunk as the argument key or context-diff syntax such as 235c235. JSON-escape "
    "newlines normally and leave apostrophes unescaped. Escape every double quote that appears "
    "inside the patch string.\n"
    "compile_file: path must be editable. show_changes: no arguments. finish: summarize only "
    "the actual change. If a patch fails, inspect its error and source context, then write a "
    "different minimal unified diff instead of resending it."
)



def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def hash_tree(directory: Path) -> tuple[dict[str, str], str]:
    entries: dict[str, str] = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or ".cache" in path.parts or path.suffix == ".pyc":
            continue
        entries[path.relative_to(directory).as_posix()] = sha256_file(path)
    return entries, sha256_json(entries)


def current_memory_state() -> dict[str, Any]:
    result: dict[str, Any] = {}
    pressure = subprocess.run(
        ["memory_pressure", "-Q"], capture_output=True, text=True, timeout=10
    )
    text = pressure.stdout + pressure.stderr
    match = re.search(r"free percentage:\s*([0-9.]+)%", text)
    result["memory_pressure_free_pct"] = float(match.group(1)) if match else None
    result["memory_pressure_output"] = text.strip()
    swap = subprocess.run(
        ["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=10
    )
    swap_text = swap.stdout.strip()
    used_match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", swap_text)
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}
    result["swap_output"] = swap_text
    result["swap_used_bytes"] = (
        int(float(used_match.group(1)) * scale[used_match.group(2)])
        if used_match
        else None
    )
    return result


class ResourceGuard:
    def __init__(
        self,
        start_state: dict[str, Any],
        free_floor_pct: float = 15.0,
        swap_growth_allowed_bytes: int = 0,
    ):
        self.baseline_swap = start_state["swap_used_bytes"]
        self.free_floor_pct = free_floor_pct
        self.swap_growth_allowed_bytes = swap_growth_allowed_bytes
        self.last_check = 0.0
        self.last_state = start_state

    def check(self, force: bool = False) -> dict[str, Any] | None:
        now = time.monotonic()
        if not force and now - self.last_check < 10:
            return None
        self.last_check = now
        state = current_memory_state()
        self.last_state = state
        free_pct = state.get("memory_pressure_free_pct")
        if free_pct is None or free_pct < self.free_floor_pct:
            return {"reason": "memory_pressure_below_floor", "state": state}
        if (
            self.baseline_swap is not None
            and state.get("swap_used_bytes") is not None
            and state["swap_used_bytes"]
            > self.baseline_swap + self.swap_growth_allowed_bytes
        ):
            return {
                "reason": "swap_growth_exceeded_allowance",
                "baseline_swap_used_bytes": self.baseline_swap,
                "allowed_growth_bytes": self.swap_growth_allowed_bytes,
                "observed_growth_bytes": state["swap_used_bytes"] - self.baseline_swap,
                "state": state,
            }
        return None


def assignment_index_guard(path: Path) -> str | None:
    assignment = json.loads(path.read_text(encoding="utf-8"))
    training = assignment.get("training", {})
    expected_status = (
        "FROZEN_BEFORE_CONTINUATION_GROUP_PROMPT_EXPOSURE"
        if NEW_ROLLOUT_START_INDEX > 0
        else "FROZEN_BEFORE_FIRST_MODEL_PROMPT_EXPOSURE"
    )
    if assignment.get("status") != expected_status:
        return "Train/Development assignment was not frozen before this group prompt exposure"
    expected_indices = list(range(NEW_ROLLOUT_START_INDEX, NEW_ROLLOUT_START_INDEX + NEW_ROLLOUT_COUNT))
    if (
        training.get("group_size") != GROUP_SIZE
        or training.get("indices") != expected_indices
        or training.get("prior_train_rollouts_attempted") != NEW_ROLLOUT_START_INDEX
    ):
        return "frozen Train group size or new sample indices differ from this run"
    if training.get("max_rollouts") != NEW_ROLLOUT_START_INDEX + NEW_ROLLOUT_COUNT or training.get("indices") != expected_indices or training.get("task_ids") != [TRAIN_TASK_ID]:
        return "frozen Train task, cumulative rollout cap, or new sample indices differ from this run"
    return None


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not RUN_LOCK_PATH.is_file():
        raise FileNotFoundError(f"frozen run lock is required: {RUN_LOCK_PATH}")
    lock = json.loads(RUN_LOCK_PATH.read_text(encoding="utf-8"))
    if lock["run_output_dir"] != str(RUN_DIR.relative_to(ROOT)):
        raise RuntimeError("run output directory differs from the frozen run lock")
    if lock["model"]["id"] != MODEL_ID or lock["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("model id or revision differs from the frozen run lock")
    if lock["rollout"] != {
        "max_rollouts": NEW_ROLLOUT_COUNT, "max_tool_turns": MAX_TURNS,
        "max_generated_tokens": MAX_TOTAL_TOKENS, "max_action_tokens": MAX_ACTION_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS, "max_wall_seconds": MAX_WALL_SECONDS,
    }:
        raise RuntimeError("rollout budgets differ from the frozen run lock")
    if lock["sampling"] != {
        "temperature": TEMPERATURE, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
        "distribution": "untruncated softmax(logits / temperature)",
    }:
        raise RuntimeError("training sampling distribution differs from the frozen run lock")
    if lock["evaluation"] != {
        "max_attempts_per_policy_per_task": 1, "max_tool_turns": MAX_TURNS,
        "max_generated_tokens": MAX_TOTAL_TOKENS, "max_action_tokens": MAX_ACTION_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS, "max_wall_seconds": MAX_WALL_SECONDS,
        "temperature": DEV_TEMPERATURE, "decode": "deterministic greedy",
    }:
        raise RuntimeError("Development evaluation budgets differ from the frozen run lock")
    if lock["optimizer"] != OPTIMIZER_CONFIG or lock["lora"] != LORA:
        raise RuntimeError("optimizer or LoRA configuration differs from the frozen run lock")
    if lock["grpo"].get("group_size") != GROUP_SIZE or lock["grpo"].get("credit_assignment") != "full_trajectory_terminal_reward":
        raise RuntimeError("GRPO group or credit assignment differs from the frozen run lock")
    expected_budget_accounting = {
        "authorized_train_rollout_max": NEW_ROLLOUT_START_INDEX + NEW_ROLLOUT_COUNT,
        "prior_train_rollouts_attempted": NEW_ROLLOUT_START_INDEX,
        "remaining_train_rollouts": NEW_ROLLOUT_COUNT,
        "first_new_sample_index": NEW_ROLLOUT_START_INDEX,
        "last_new_sample_index": NEW_ROLLOUT_START_INDEX + NEW_ROLLOUT_COUNT - 1,
        "group_size": GROUP_SIZE,
    }
    if lock["budget_accounting"] != expected_budget_accounting:
        raise RuntimeError("new-task Train rollout budget or sample indices differ from the frozen lock")
    if lock["system_prompt_sha256"] != sha256_bytes(SYSTEM_PROMPT.encode()):
        raise RuntimeError("agent system prompt differs from the frozen run lock")
    if lock["tool_help_sha256"] != sha256_bytes(TOOL_HELP.encode()):
        raise RuntimeError("agent tool help differs from the frozen run lock")
    if sha256_file(ASSIGNMENT_PATH) != lock["assignment_sha256"]:
        raise RuntimeError("assignment manifest hash differs from the frozen run lock")
    assignment = json.loads(ASSIGNMENT_PATH.read_text(encoding="utf-8"))
    index_error = assignment_index_guard(ASSIGNMENT_PATH)
    if index_error:
        raise RuntimeError(index_error)
    if assignment.get("no_go_gate", {}).get("sha256") != lock["no_go_gate"]["sha256"]:
        raise RuntimeError("assignment and run lock bind different frozen NO-GO gates")
    no_go_path = ROOT / lock["no_go_gate"]["path"]
    if sha256_file(no_go_path) != lock["no_go_gate"]["sha256"]:
        raise RuntimeError("frozen held-out NO-GO gate changed")
    if lock["no_go_gate"].get("status") != "NO_GO remains frozen and unchanged":
        raise RuntimeError("run lock does not preserve the frozen NO-GO status")
    binding_path = ROOT / lock["environment_binding_path"]
    if sha256_file(binding_path) != lock["environment_binding_sha256"]:
        raise RuntimeError("Train/Development environment binding changed")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    continuation = assignment.get("status") == "FROZEN_BEFORE_CONTINUATION_GROUP_PROMPT_EXPOSURE"
    expected_exposure = continuation
    if (
        binding["assignment_manifest_sha256"] != sha256_file(ASSIGNMENT_PATH)
        or binding.get("model_or_agent_exposure_before_freeze") is not expected_exposure
        or binding.get("new_group_prompt_exposure_before_freeze") is not False
    ):
        raise RuntimeError("environment binding does not accurately freeze the next group's exposure state")
    if binding.get("no_go_gate_sha256") != lock["no_go_gate"]["sha256"]:
        raise RuntimeError("environment binding references a different NO-GO gate")
    if sha256_file(Path(__file__)) != lock["source_code"]["runner_sha256"]:
        raise RuntimeError("runner code differs from its frozen run lock")
    if sha256_file(GRPO_PATH) != lock["source_code"]["grpo_mlx_sha256"]:
        raise RuntimeError("GRPO implementation differs from its frozen run lock")
    if sha256_file(CONTAINER_RUNNER) != lock["source_code"]["container_runner_sha256"]:
        raise RuntimeError("independent grader wrapper differs from the frozen run lock")
    if sha256_file(MODEL_MANIFEST_PATH) != lock["model"]["manifest_sha256"]:
        raise RuntimeError("model manifest differs from the frozen run lock")
    model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    actual_files, _ = hash_tree(MODEL_PATH)
    if actual_files != model_manifest["files"]:
        raise RuntimeError("local pretrained model files differ from their frozen manifest")
    if model_manifest["model_id"] != MODEL_ID or model_manifest["revision"] != MODEL_REVISION:
        raise RuntimeError("local model revision differs from the frozen run lock")
    if lock["model"]["files_tree_sha256"] != model_manifest["tree_sha256"]:
        raise RuntimeError("pretrained model file tree hash differs from the run lock")

    task_records = assignment["task_records"]
    if len(task_records) != 3 or sum(r["split"] == "train" for r in task_records) != 1 or sum(r["split"] == "development" for r in task_records) != 2:
        raise RuntimeError("assignment must have one Train and two Development tasks")
    if any(r.get("qualification_passed") is not True for r in task_records):
        raise RuntimeError("an assigned task lacks passed source/test qualification")
    train = [r for r in task_records if r["split"] == "train"]
    if [r["task_id"] for r in train] != [TRAIN_TASK_ID]:
        raise RuntimeError("unexpected Train task membership")
    if lock["training_task"]["task_id"] != train[0]["task_id"]:
        raise RuntimeError("run lock Train task differs from the frozen assignment")
    if [r["task_id"] for r in lock["development_tasks"]] != assignment["evaluation"]["task_ids"]:
        raise RuntimeError("run lock Development tasks differ from the frozen assignment")
    families = [r["task_family"] for r in task_records]
    owned_paths = [path for r in task_records for path in r["editable_source_paths"]]
    if len(set(families)) != 3 or len(set(owned_paths)) != len(owned_paths):
        raise RuntimeError("Train/Development source paths or behavior families overlap")

    binding_by_task = {row["task_id"]: row for row in binding["qualification_bindings"]}
    if set(binding_by_task) != {row["task_id"] for row in task_records}:
        raise RuntimeError("environment binding task set differs from assignment")
    for record in task_records:
        task_path = ROOT / record["task_definition_path"]
        summary_path = ROOT / record["qualification_summary_path"]
        if sha256_file(task_path) != record["task_definition_sha256"]:
            raise RuntimeError(f"task definition changed: {record['task_id']}")
        if sha256_file(summary_path) != record["qualification_summary_sha256"]:
            raise RuntimeError(f"qualification record changed: {record['task_id']}")
        definition = json.loads(task_path.read_text(encoding="utf-8"))
        env_path = ROOT / definition["execution"]["environment_lock_path"]
        bound = binding_by_task[record["task_id"]]
        if str(env_path.relative_to(ROOT)) != bound["environment_lock_path"] or sha256_file(env_path) != bound["environment_lock_sha256"]:
            raise RuntimeError(f"grader environment changed: {record['task_id']}")
        lock_env = lock["task_environments"][record["task_id"]]
        if lock_env["path"] != bound["environment_lock_path"] or lock_env["sha256"] != bound["environment_lock_sha256"] or lock_env["image_id"] != bound["image_id"]:
            raise RuntimeError(f"run lock environment mismatch: {record['task_id']}")
    return lock, assignment, task_records


def load_task(record: dict[str, Any]) -> dict[str, Any]:
    definition_path = ROOT / record["task_definition_path"]
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    source_test = definition.get("source_test") or definition.get("source_test_manifest")
    prompt = definition.get("task_prompt") or definition.get("agent_task_prompt")
    if record.get("prompt_variant_path"):
        prompt_variant_path = ROOT / record["prompt_variant_path"]
        if sha256_file(prompt_variant_path) != record.get("prompt_variant_sha256"):
            raise RuntimeError(f"frozen prompt variant changed: {record['task_id']}")
        prompt_variant = json.loads(prompt_variant_path.read_text(encoding="utf-8"))
        if prompt_variant.get("base_task_definition_sha256") != record["task_definition_sha256"]:
            raise RuntimeError(f"prompt variant is not bound to the qualified task: {record['task_id']}")
        prompt = prompt_variant["task_prompt"]
    if not prompt or not source_test:
        raise RuntimeError(f"task contract is incomplete: {record['task_id']}")
    env_lock_path = TASK_ENV_LOCK_OVERRIDES.get(
        record["task_id"],
        ROOT / definition["execution"]["environment_lock_path"],
    )
    env_lock = json.loads(env_lock_path.read_text(encoding="utf-8"))
    if "editable_paths" in definition:
        editable = list(definition["editable_paths"])
    else:
        editable = [source_test["editable_path"]]
    return {
        "record": record,
        "definition": definition,
        "definition_sha256": sha256_file(definition_path),
        "baseline_root": TASK_ARM_DIRS[record["task_id"]],
        "prompt": prompt,
        "editable_paths": editable,
        "source_test": source_test,
        "env_lock": env_lock,
        "env_lock_sha256": sha256_file(env_lock_path),
        "test_path": source_test["test_path"],
        "selector": source_test["selector"],
        "test_sha256": source_test["test_sha256"],
    }


def verify_task_workspace(task: dict[str, Any]) -> None:
    baseline = task["baseline_root"]
    if not (baseline / "pydantic").is_dir() or not (baseline / task["test_path"]).is_file():
        raise RuntimeError(f"qualified baseline or grader test is missing: {task['record']['task_id']}")
    source_test = task["source_test"]
    if sha256_file(baseline / task["test_path"]) != task["test_sha256"]:
        raise RuntimeError(f"frozen grader test changed: {task['record']['task_id']}")
    source_hashes = source_test.get("source_hashes", {}).get("baseline")
    if not source_hashes and "baseline_source_sha256" in source_test:
        source_hashes = {source_test["editable_path"]: source_test["baseline_source_sha256"]}
    if not isinstance(source_hashes, dict):
        raise RuntimeError(f"missing baseline source hash binding: {task['record']['task_id']}")
    for rel, expected in source_hashes.items():
        if sha256_file(baseline / rel) != expected:
            raise RuntimeError(f"baseline source hash mismatch: {task['record']['task_id']}:{rel}")
    for rel, expected in source_test.get("support_files", {}).items():
        if sha256_file(baseline / rel) != expected:
            raise RuntimeError(f"grader support hash mismatch: {task['record']['task_id']}:{rel}")
    visible = "\n".join(
        [
            SYSTEM_PROMPT,
            TOOL_HELP.replace("{prompt}", task["prompt"]).replace(
                "{paths}", ", ".join(task["editable_paths"])
            ),
        ]
    )
    if sha256_json(make_agent_messages(task)) != task["record"]["prompt_sha256"]:
        raise RuntimeError(f"candidate-visible prompt differs from frozen assignment: {task['record']['task_id']}")
    forbidden_visible_markers = {
        task["record"]["candidate_id"],
        task["record"]["task_definition_path"],
        task["record"]["qualification_summary_path"],
        task["test_path"],
        task["selector"],
        task["test_sha256"],
    }
    leaked = sorted(marker for marker in forbidden_visible_markers if marker and marker in visible)
    if leaked:
        raise RuntimeError(
            f"candidate-visible task prompt contains grader or allocation metadata: {leaked}"
        )
    if not task["env_lock"].get("image_id", "").startswith("sha256:"):
        raise RuntimeError(f"grader image is not pinned: {task['record']['task_id']}")
    image = task["env_lock"]["image_id"]
    inspect = subprocess.run(
        ["docker", "--context", "colima", "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if inspect.returncode or inspect.stdout.strip() != image:
        raise RuntimeError(f"pinned offline grader image unavailable: {task['record']['task_id']}")


def git_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="VARE Candidate",
        GIT_AUTHOR_EMAIL="vare-candidate@localhost",
        GIT_COMMITTER_NAME="VARE Candidate",
        GIT_COMMITTER_EMAIL="vare-candidate@localhost",
    )
    return env


def create_candidate_workspace(task: dict[str, Any], directory: Path) -> Path:
    if directory.exists():
        raise FileExistsError(f"candidate workspace already exists: {directory}")
    directory.mkdir(parents=True)
    shutil.copytree(
        task["baseline_root"] / "pydantic",
        directory / "pydantic",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    if any("tests" in p.parts for p in directory.rglob("*")):
        raise RuntimeError("candidate workspace unexpectedly contains test material")
    subprocess.run(["git", "init", "-q", str(directory)], check=True, env=git_env(), timeout=10)
    subprocess.run(["git", "-C", str(directory), "add", "pydantic"], check=True, env=git_env(), timeout=10)
    subprocess.run(
        ["git", "-C", str(directory), "commit", "-qm", "frozen task baseline"],
        check=True,
        env=git_env(),
        timeout=10,
    )
    return directory


def safe_source_path(directory: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError("path must be a safe relative production source path")
    if not relative.parts or relative.parts[0] != "pydantic":
        raise ValueError("only the pydantic production package is visible")
    path = directory.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError("source path is missing or is a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to((directory / "pydantic").resolve()):
        raise ValueError("source path escapes the candidate workspace")
    return path


def decode_loose_json_string(value: str) -> str:
    """Decode JSON escapes while preserving unescaped quotes in recovered patch text."""
    output: list[str] = []
    escapes = {
        '"': '"', "\\": "\\", "/": "/", "b": "\b",
        "f": "\f", "n": "\n", "r": "\r", "t": "\t", "'": "'",
    }
    index = 0
    while index < len(value):
        if value[index] != "\\" or index + 1 >= len(value):
            output.append(value[index])
            index += 1
            continue
        escape = value[index + 1]
        if escape == "u" and index + 5 < len(value):
            digits = value[index + 2 : index + 6]
            try:
                output.append(chr(int(digits, 16)))
                index += 6
                continue
            except ValueError:
                pass
        if escape in escapes:
            output.append(escapes[escape])
            index += 2
            continue
        output.extend(("\\", escape))
        index += 2
    return "".join(output)


def salvage_malformed_apply_patch_action(raw: str) -> dict[str, Any] | None:
    """Recover only apply_patch calls whose JSON string body is malformed or truncated."""
    content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    tool_match = re.search(r'"tool"\s*:\s*"apply_patch"', content)
    if tool_match is None:
        return None
    arguments_match = re.search(r'"arguments"\s*:\s*\{', content[tool_match.end() :])
    if arguments_match is None:
        return None
    arguments_start = tool_match.end() + arguments_match.end()
    patch_match = re.search(r'"(patch|hunk)"\s*:\s*"', content[arguments_start:])
    if patch_match is None:
        return None
    patch_key = patch_match.group(1)
    value_start = arguments_start + patch_match.end()
    prefix = content[arguments_start:value_start]
    path_match = re.search(r'"path"\s*:\s*"([^"\r\n]+)"', prefix)
    closing = re.search(r'"\s*\}\}\s*$', content[value_start:])
    incomplete = closing is None
    patch_body = (
        content[value_start : value_start + closing.start()]
        if closing is not None
        else content[value_start:]
    )
    arguments: dict[str, Any] = {patch_key: decode_loose_json_string(patch_body)}
    if path_match is not None:
        arguments["path"] = path_match.group(1)
    action: dict[str, Any] = {
        "tool": "apply_patch",
        "arguments": arguments,
        "output_normalization": "malformed_apply_patch_json_salvaged",
    }
    if incomplete:
        action["output_normalization"] += "_incomplete"
    return action


def action_from_text(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    for value in reversed(candidates):
        tool = value.get("tool") or value.get("name")
        arguments = value.get("arguments")
        if arguments is None and isinstance(value.get("function"), dict):
            function = value["function"]
            tool = tool or function.get("name")
            arguments = function.get("arguments")
        argument_normalization = None
        if isinstance(arguments, str):
            raw_arguments = arguments
            try:
                parsed_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                parsed_arguments = None
            if isinstance(parsed_arguments, dict):
                arguments = parsed_arguments
            elif tool == "search_code" and isinstance(parsed_arguments, str) and parsed_arguments.strip():
                arguments = {"query": parsed_arguments.strip()}
                argument_normalization = "search_code_scalar_to_query_object"
            elif tool == "search_code" and raw_arguments.strip():
                arguments = {"query": raw_arguments.strip()}
                argument_normalization = "search_code_scalar_to_query_object"
            else:
                continue
        if not isinstance(tool, str):
            continue
        if arguments is None:
            arguments = {k: v for k, v in value.items() if k not in {"tool", "name", "function"}}
        if not isinstance(arguments, dict):
            continue
        action = {"tool": tool, "arguments": arguments}
        if argument_normalization:
            action["argument_normalization"] = argument_normalization
        return action, None
    salvaged = salvage_malformed_apply_patch_action(content)
    if salvaged is not None:
        return salvaged, None
    return None, "model output did not contain one valid tool-action JSON object"


def bounded_result(value: dict[str, Any], max_chars: int = 1800) -> dict[str, Any]:
    encoded = json.dumps(value, ensure_ascii=False)
    if len(encoded) <= max_chars:
        return value
    return {
        "truncated": True,
        "message": "Tool output exceeded its response limit; inspect a smaller source range.",
        "preview": encoded[: max_chars - 200],
    }


def list_source_files(directory: Path, rel: str) -> dict[str, Any]:
    relative = PurePosixPath(rel)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "pydantic":
        raise ValueError("list_files path must stay inside the production package")
    base = directory.joinpath(*relative.parts)
    if base.is_symlink() or not base.is_dir() or not base.resolve().is_relative_to((directory / "pydantic").resolve()):
        raise ValueError("list_files path must be a production source directory")
    items = sorted(
        p.relative_to(directory).as_posix()
        for p in base.rglob("*.py")
        if p.is_file() and not p.is_symlink()
    )
    return {"files": items[:25], "total": len(items), "truncated": len(items) > 25}


def search_source(directory: Path, query: str, allowed: set[str]) -> dict[str, Any]:
    if not query or len(query) > 200:
        raise ValueError("search query must contain 1 to 200 characters")
    results: list[dict[str, Any]] = []
    for relative in sorted(allowed):
        path = safe_source_path(directory, relative)
        if not path.is_file() or path.is_symlink():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(lines, 1):
            if query in line:
                results.append(
                    {"path": path.relative_to(directory).as_posix(), "line": line_no, "text": line[:180]}
                )
                if len(results) >= 12:
                    return {"matches": results, "truncated": True}
    return {"matches": results, "truncated": False}


def read_source(directory: Path, args: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    requested = str(args.get("path", ""))
    if requested not in allowed:
        raise ValueError("read_file accepts only the task editable source path")
    path = safe_source_path(directory, requested)
    lines = path.read_text(encoding="utf-8").splitlines()
    start = args.get("start_line", 1)
    end = args.get("end_line")
    if isinstance(start, bool) or not isinstance(start, int) or start < 1:
        raise ValueError("start_line must be a positive integer")
    if end is not None and (isinstance(end, bool) or not isinstance(end, int) or end < start):
        raise ValueError("end_line must be an integer at least start_line")
    requested_stop = min(
        end if end is not None else start + READ_FILE_PAGE_LINES - 1,
        len(lines),
    )
    stop = min(requested_stop, start + READ_FILE_PAGE_LINES - 1)
    line_limit_truncated = stop < requested_stop
    selected = lines[start - 1 : stop]
    full_text = "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start))
    character_limit_truncated = len(full_text) > 2600
    content_truncated = line_limit_truncated or character_limit_truncated
    text = full_text[:2600]
    return {
        "path": path.relative_to(directory).as_posix(),
        "source_sha256": sha256_file(path),
        "start_line": start,
        "end_line": stop,
        "total_lines": len(lines),
        "content": text,
        "content_truncated": content_truncated,
        "truncation_reasons": [
            reason
            for condition, reason in (
                (line_limit_truncated, "line_limit"),
                (character_limit_truncated, "character_limit"),
            )
            if condition
        ],
        "next_start_line": stop + 1 if stop < len(lines) else None,
    }


def validate_patch(patch: str, allowed: set[str]) -> None:
    if not patch or len(patch.encode("utf-8")) > 256_000 or "GIT binary patch" in patch:
        raise ValueError("patch is empty, binary, or larger than 256 KB")
    if any(line.startswith(("rename from ", "rename to ", "new file mode ", "deleted file mode ")) for line in patch.splitlines()):
        raise ValueError("file creation, deletion, and rename are forbidden")
    diff_paths: list[tuple[str, str]] = []
    file_headers: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) != 4 or not parts[2].startswith("a/") or not parts[3].startswith("b/"):
                raise ValueError("malformed diff header")
            diff_paths.append((parts[2][2:], parts[3][2:]))
        elif line.startswith("--- ") or line.startswith("+++ "):
            name = line[4:].split("\t", 1)[0]
            if name == "/dev/null" or not (name.startswith("a/") or name.startswith("b/")):
                raise ValueError("new or deleted files are forbidden")
            relative = name[2:]
            parsed = PurePosixPath(relative)
            if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
                raise ValueError("patch path traversal is forbidden")
            if relative not in allowed:
                raise ValueError(f"path is not editable: {relative}")
            file_headers.append(relative)
    if not file_headers or len(file_headers) % 2:
        raise ValueError("patch must contain complete old/new file headers")
    pairs = list(zip(file_headers[::2], file_headers[1::2], strict=True))
    if any(left != right for left, right in pairs):
        raise ValueError("renames are forbidden")
    if diff_paths and diff_paths != pairs:
        raise ValueError("diff headers do not match the editable file headers")


def apply_source_patch(directory: Path, patch: str, allowed: set[str]) -> dict[str, Any]:
    validate_patch(patch, allowed)
    checked = subprocess.run(
        ["git", "-C", str(directory), "apply", "--check", "--whitespace=nowarn", "-"],
        input=patch,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if checked.returncode:
        return {"applied": False, "error": checked.stderr[-3000:] or checked.stdout[-3000:]}
    applied = subprocess.run(
        ["git", "-C", str(directory), "apply", "--whitespace=nowarn", "-"],
        input=patch,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if applied.returncode:
        return {"applied": False, "error": applied.stderr[-3000:] or applied.stdout[-3000:]}
    return {
        "applied": True,
        "changed_files": changed_files(directory, allowed),
        "source_hashes": {path: sha256_file(directory / path) for path in sorted(allowed)},
    }


def validate_pydantic_union_patch_scope(
    directory: Path, baseline: Path, patch: str
) -> None:
    """Keep the pydantic-14138 fast path inside its requested loop."""
    relative = "pydantic/_internal/_generate_schema.py"
    current_source = (directory / relative).read_text(encoding="utf-8")
    try:
        tree = ast.parse(current_source, filename=relative)
    except SyntaxError:
        tree = ast.parse((baseline / relative).read_text(encoding="utf-8"), filename=relative)
    method = next(
        (
            node
            for cls in ast.walk(tree)
            if isinstance(cls, ast.ClassDef) and cls.name == "GenerateSchema"
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_union_schema"
        ),
        None,
    )
    if method is None:
        raise ValueError("task-scope validation could not locate GenerateSchema._union_schema")
    loops = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "arg"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "args"
        and node.end_lineno is not None
    ]
    if len(loops) != 1:
        raise ValueError("task-scope validation could not uniquely locate the for arg in args loop")
    first_line, last_line = loops[0].lineno, loops[0].end_lineno
    hunk = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    old_line: int | None = None
    changed_old_lines: list[int] = []
    insertion_anchors: list[int] = []
    for line in patch.splitlines():
        match = hunk.match(line)
        if match:
            old_line = int(match.group(1))
            continue
        if old_line is None:
            continue
        if line.startswith(" "):
            old_line += 1
        elif line.startswith("-") and not line.startswith("--- "):
            changed_old_lines.append(old_line)
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++ "):
            insertion_anchors.append(old_line)
        elif line.startswith("\\"):
            continue
        elif line.startswith(("diff --git ", "--- ", "+++ ")):
            old_line = None
        else:
            old_line = None
    outside = [
        line_number
        for line_number in changed_old_lines + insertion_anchors
        if not first_line <= line_number <= last_line
    ]
    if not changed_old_lines and not insertion_anchors:
        raise ValueError("task-scope validation found no changed source lines")
    if outside:
        raise ValueError(
            "task-scope violation: changed lines must stay inside "
            f"GenerateSchema._union_schema's for arg in args loop (baseline lines "
            f"{first_line}-{last_line}); the post-loop union assembly is out of scope"
        )


def normalize_unified_hunk_counts(patch: str) -> tuple[str, bool]:
    """Recount old and new lines in complete unified hunks before Git validation."""
    lines = patch.splitlines()
    header_pattern = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
    )
    changed = False
    for index, header in enumerate(lines):
        match = header_pattern.match(header)
        if match is None:
            continue
        old_start, new_start, context = match.group(1), match.group(3), match.group(5)
        old_count = 0
        new_count = 0
        cursor = index + 1
        while cursor < len(lines):
            line = lines[cursor]
            if header_pattern.match(line) or line.startswith("diff --git "):
                break
            if line.startswith(("--- a/", "+++ b/")):
                break
            if line.startswith(" "):
                old_count += 1
                new_count += 1
            elif line.startswith("-"):
                old_count += 1
            elif line.startswith("+"):
                new_count += 1
            elif line.startswith("\\"):
                pass
            else:
                break
            cursor += 1
        corrected = (
            f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{context}"
        )
        if corrected != header:
            lines[index] = corrected
            changed = True
    return "\n".join(lines) + ("\n" if patch.endswith("\n") else ""), changed


def normalize_patch_arguments(
    arguments: dict[str, Any], allowed: set[str]
) -> tuple[str, str, list[str]]:
    """Normalize the narrow patch-call variants emitted by the pretrained policy."""
    normalizations: list[str] = []
    patch = arguments.get("patch")
    if not isinstance(patch, str) or not patch.strip():
        hunk = arguments.get("hunk")
        if isinstance(hunk, str) and hunk.strip():
            patch = hunk
            normalizations.append("hunk_argument_aliased_to_patch")
    if not isinstance(patch, str) or not patch.strip():
        raise ValueError("apply_patch requires a non-empty patch string")

    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        if len(allowed) != 1:
            raise ValueError("apply_patch requires an exact editable source path")
        path = next(iter(allowed))
        normalizations.append("single_editable_path_inferred")
    if path not in allowed:
        raise ValueError("apply_patch accepts only the exact task editable source path")

    lines = patch.splitlines()
    has_old_header = any(line.startswith("--- ") for line in lines)
    has_new_header = any(line.startswith("+++ ") for line in lines)
    if not has_old_header and not has_new_header and any(
        line.startswith("@@") for line in lines
    ):
        headers = [f"--- a/{path}", f"+++ b/{path}"]
        if lines and lines[0].startswith("diff --git "):
            patch = "\n".join([lines[0], *headers, *lines[1:]]) + "\n"
        else:
            patch = "\n".join([*headers, *lines]) + "\n"
        normalizations.append("unified_diff_file_headers_synthesized_for_allowlisted_path")
    if not patch.endswith("\n"):
        patch += "\n"
        normalizations.append("patch_final_newline_added")
    patch, counts_changed = normalize_unified_hunk_counts(patch)
    if counts_changed:
        normalizations.append("unified_diff_hunk_line_counts_recomputed")
    return path, patch, normalizations


def changed_files(directory: Path, allowed: set[str], baseline: Path | None = None) -> list[str]:
    result: list[str] = []
    for relative in sorted(allowed):
        current = directory / relative
        if not current.is_file():
            continue
        if baseline is None:
            git_result = subprocess.run(
                ["git", "-C", str(directory), "diff", "--quiet", "--", relative],
                capture_output=True,
            )
            if git_result.returncode != 0:
                result.append(relative)
        elif sha256_file(current) != sha256_file(baseline / relative):
            result.append(relative)
    return result


def show_changes(directory: Path, allowed: set[str], baseline: Path) -> dict[str, Any]:
    rows = []
    for relative in sorted(allowed):
        original = (baseline / relative).read_text(encoding="utf-8").splitlines(keepends=True)
        current = (directory / relative).read_text(encoding="utf-8").splitlines(keepends=True)
        if original != current:
            rows.extend(
                difflib.unified_diff(
                    original,
                    current,
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                    n=3,
                )
            )
    text = "".join(rows)
    return {"diff": text[:20000], "truncated": len(text) > 20000, "changed_files": changed_files(directory, allowed, baseline)}


def execute_tool(
    directory: Path,
    baseline: Path,
    allowed: set[str],
    action: dict[str, Any],
    *,
    task_id: str | None = None,
    read_cursors: dict[tuple[str, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    name = action["tool"]
    args = action["arguments"]
    cursors = read_cursors if read_cursors is not None else {}
    try:
        if name == "search_code":
            return bounded_result(search_source(directory, str(args.get("query", "")), allowed))
        if name == "read_file":
            requested_start = args.get("start_line", 1)
            requested_end = args.get("end_line")
            requested_path = str(args.get("path", ""))
            if (
                isinstance(requested_start, bool)
                or not isinstance(requested_start, int)
                or requested_start < 1
            ):
                raise ValueError("start_line must be a positive integer")
            if requested_end is not None and (
                isinstance(requested_end, bool)
                or not isinstance(requested_end, int)
                or requested_end < requested_start
            ):
                raise ValueError("end_line must be an integer at least start_line")
            if requested_path not in allowed:
                raise ValueError("read_file accepts only the task editable source path")

            source_path = safe_source_path(directory, requested_path)
            source_lines = source_path.read_text(encoding="utf-8").splitlines()
            source_sha256 = sha256_file(source_path)
            coverage_key = ("__read_coverage__", requested_path, "lines")
            coverage = cursors.get(coverage_key)
            if not isinstance(coverage, dict) or coverage.get("source_sha256") != source_sha256:
                coverage = {"source_sha256": source_sha256, "seen_lines": set()}
                cursors[coverage_key] = coverage
            seen_lines = coverage["seen_lines"]

            requested_last = min(
                requested_end
                if requested_end is not None
                else requested_start + READ_FILE_PAGE_LINES - 1,
                len(source_lines),
            )
            requested_width = (
                requested_end - requested_start + 1
                if requested_end is not None
                else READ_FILE_PAGE_LINES
            )
            read_args = dict(args)
            page_start = requested_start
            auto_paged = False
            oversized = requested_width > READ_FILE_PAGE_LINES
            if requested_last < requested_start:
                result = read_source(directory, read_args, allowed)
                result["requested_start_line"] = requested_start
                result["requested_end_line"] = requested_end
                result["auto_paged"] = False
                return bounded_result(result, max_chars=3200)
            if oversized:
                page_start = next(
                    (
                        line_number
                        for line_number in range(requested_start, requested_last + 1)
                        if line_number not in seen_lines
                    ),
                    None,
                )
                if page_start is None:
                    replay_end = min(
                        requested_last,
                        requested_start + READ_FILE_PAGE_LINES - 1,
                    )
                    replay = read_source(
                        directory,
                        {
                            "path": requested_path,
                            "start_line": requested_start,
                            "end_line": replay_end,
                        },
                        allowed,
                    )
                    replay.update(
                        {
                            "requested_start_line": requested_start,
                            "requested_end_line": requested_end,
                            "auto_paged": False,
                            "already_read": True,
                            "already_paged": True,
                            "content_replayed": True,
                        }
                    )
                    return bounded_result(replay, max_chars=3200)
                page_end = min(requested_last, page_start + READ_FILE_PAGE_LINES - 1)
                next_covered = next(
                    (
                        line_number
                        for line_number in range(page_start, page_end + 1)
                        if line_number in seen_lines
                    ),
                    None,
                )
                if next_covered is not None:
                    page_end = next_covered - 1
                read_args["start_line"] = page_start
                read_args["end_line"] = page_end
                auto_paged = page_start != requested_start

            result = read_source(directory, read_args, allowed)
            result["requested_start_line"] = requested_start
            result["requested_end_line"] = requested_end
            result["auto_paged"] = auto_paged
            if result["end_line"] >= result["start_line"]:
                seen_lines.update(range(result["start_line"], result["end_line"] + 1))
            if oversized:
                next_unread = next(
                    (
                        line_number
                        for line_number in range(result["end_line"] + 1, requested_last + 1)
                        if line_number not in seen_lines
                    ),
                    None,
                )
                if next_unread is not None:
                    result["content_truncated"] = True
                    reasons = list(result["truncation_reasons"])
                    if "unread_page" not in reasons:
                        reasons.append("unread_page")
                    result["truncation_reasons"] = reasons
                    result["next_start_line"] = next_unread
                else:
                    result["next_start_line"] = None
            return bounded_result(result, max_chars=3200)
        if name == "apply_patch":
            path, patch, normalizations = normalize_patch_arguments(args, allowed)
            if task_id == TASK_SCOPE_GUARD_TASK_ID:
                validate_pydantic_union_patch_scope(directory, baseline, patch)
            result = apply_source_patch(directory, patch, allowed)
            result["path"] = path
            if normalizations:
                result["argument_normalizations"] = normalizations
            if result.get("applied"):
                cursors.clear()
            return bounded_result(result)
        if name == "compile_file":
            path = str(args.get("path", ""))
            if path not in allowed:
                raise ValueError("compile_file accepts only editable source paths")
            source = safe_source_path(directory, path).read_text(encoding="utf-8")
            ast.parse(source, filename=path)
            return {"compiled": True, "path": path}
        if name == "show_changes":
            return bounded_result(show_changes(directory, allowed, baseline))
        if name == "finish":
            return {"finished": True, "summary": str(args.get("summary", ""))[:1000]}
        return {"error": f"unknown tool: {name}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def extract_sampling_behavior_logprob(item: Any, temperature: float) -> float | None:
    """Selected token log-probability under the actual mlx-lm sampler.

    GenerationResponse.logprobs are raw temperature-1 model log-probabilities;
    mlx-lm divides them by temperature before categorical sampling.
    """
    if temperature <= 0:
        return None
    value = getattr(item, "logprobs", None)
    if value is None:
        return None
    try:
        import mlx.core as mx

        scaled = value.astype(mx.float32) / temperature
        selected = scaled[int(item.token)] - mx.logsumexp(scaled, axis=-1)
        mx.eval(selected)
        picked = float(selected.item())
        return picked if math.isfinite(picked) else None
    except Exception:
        return None


def generate_action(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    seed: int,
    temperature: float,
    max_tokens: int,
    guard: ResourceGuard,
    deadline: float,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    tokenized = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    prompt_ids = [int(token) for token in tokenized]
    if len(prompt_ids) > MAX_PROMPT_TOKENS:
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": [],
            "sampling_behavior_logprobs": [],
            "text": "",
            "finish": "prompt_too_long",
            "seconds": 0.0,
            "resource_violation": None,
        }
    mx.random.seed(seed)
    sampler = make_sampler(
        temp=temperature,
        top_p=1.0 if temperature > 0 else 0.0,
        top_k=0,
        min_p=0.0,
    )
    started = time.monotonic()
    completion: list[int] = []
    logprobs: list[float | None] = []
    text = ""
    finish = "length"
    violation = None
    for item in stream_generate(
        model,
        tokenizer,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        completion.append(int(item.token))
        logprobs.append(extract_sampling_behavior_logprob(item, temperature))
        text += item.text
        if item.finish_reason:
            finish = str(item.finish_reason)
        if time.monotonic() >= deadline:
            finish = "wall_time_limit"
            break
        violation = guard.check()
        if violation:
            finish = "resource_stop"
            break
    mx.clear_cache()
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion,
        "sampling_behavior_logprobs": logprobs,
        "text": text,
        "finish": finish,
        "seconds": round(time.monotonic() - started, 6),
        "resource_violation": violation,
    }


def make_agent_messages(
    task: dict[str, Any], progress_summary: str | None = None
) -> list[dict[str, str]]:
    paths = ", ".join(task["editable_paths"])
    user_content = TOOL_HELP.replace("{prompt}", task["prompt"]).replace("{paths}", paths)
    if progress_summary:
        user_content += "\n\nCompact progress from earlier tool turns:\n" + progress_summary
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def compact_action_summary(turns: list[dict[str, Any]]) -> str:
    """Summarize prior tool events while retaining the latest source page."""
    rows: list[str] = []
    last_source_page: str | None = None
    for row in turns:
        action = row.get("action") or {}
        tool = action.get("tool", "invalid_action") if isinstance(action, dict) else "invalid_action"
        args = action.get("arguments", {}) if isinstance(action, dict) else {}
        result = row.get("tool_result") or {}
        if tool == "search_code":
            query = str(args.get("query", ""))[:80]
            matches = result.get("matches", []) if isinstance(result, dict) else []
            locations = ", ".join(
                f"{m.get('path')}:{m.get('line')}" for m in matches[:4]
                if isinstance(m, dict)
            )
            rows.append(f"turn {row.get('turn')}: searched {query!r}; locations={locations or 'none'}")
        elif tool == "read_file":
            if isinstance(result, dict) and result.get("content"):
                last_source_page = (
                    f"{result.get('path')}:{result.get('start_line')}-{result.get('end_line')}\n"
                    + str(result["content"])[:1250]
                )
            if isinstance(result, dict) and (
                result.get("already_paged") or result.get("already_read")
            ):
                rows.append(
                    f"turn {row.get('turn')}: requested {args.get('path')} lines "
                    f"{args.get('start_line', 1)}-{args.get('end_line', 'default')}; "
                    "that range was already read completely"
                )
            else:
                actual_start = (
                    result.get("start_line", args.get("start_line", 1))
                    if isinstance(result, dict)
                    else args.get("start_line", 1)
                )
                actual_end = (
                    result.get("end_line", args.get("end_line", "default"))
                    if isinstance(result, dict)
                    else args.get("end_line", "default")
                )
                page = "; auto-paged" if isinstance(result, dict) and result.get("auto_paged") else ""
                next_line = result.get("next_start_line") if isinstance(result, dict) else None
                continuation = f"; next_start_line={next_line}" if next_line is not None else ""
                rows.append(
                    f"turn {row.get('turn')}: read {args.get('path')} requested "
                    f"{args.get('start_line', 1)}-{args.get('end_line', 'default')}; "
                    f"returned {actual_start}-{actual_end}{page}{continuation}"
                )
        elif tool == "apply_patch":
            if isinstance(result, dict) and result.get("applied"):
                rows.append(f"turn {row.get('turn')}: patch applied; changed={result.get('changed_files', [])}")
            else:
                error = str(result.get("error", "patch not applied"))[:120] if isinstance(result, dict) else "patch not applied"
                rows.append(f"turn {row.get('turn')}: patch not applied; {error}")
        elif tool == "compile_file":
            rows.append(f"turn {row.get('turn')}: compile {args.get('path')} -> {bool(result.get('compiled'))}")
        elif tool == "show_changes":
            changed = result.get("changed_files", []) if isinstance(result, dict) else []
            rows.append(f"turn {row.get('turn')}: current changed paths={changed}")
        else:
            rows.append(f"turn {row.get('turn')}: {tool} completed")
    summary = "\n".join(rows)
    if last_source_page:
        summary = summary[-450:] + "\nLast visible source page:\n" + last_source_page
    return summary[-1800:]


def run_agent(
    model: Any,
    tokenizer: Any,
    task: dict[str, Any],
    *,
    workspace: Path,
    artifact_dir: Path,
    policy_id: str,
    phase: str,
    sample_index: int,
    seed: int,
    temperature: float,
    guard: ResourceGuard,
) -> dict[str, Any]:
    baseline = create_candidate_workspace(task, workspace)
    allowed = set(task["editable_paths"])
    for rel in allowed:
        if not (baseline / rel).is_file():
            raise RuntimeError(f"editable source path is absent: {rel}")
    messages = make_agent_messages(task)
    turns: list[dict[str, Any]] = []
    read_cursors: dict[tuple[str, str, str], dict[str, Any]] = {}
    total_tokens = 0
    first_prompt_ids: list[int] = []
    first_completion_ids: list[int] = []
    first_sampling_behavior_logprobs: list[float | None] = []
    submitted = False
    submit_summary = ""
    stop_reason = "turn_budget_exhausted"
    started = time.monotonic()
    deadline = started + MAX_WALL_SECONDS

    for turn_index in range(MAX_TURNS):
        violation = guard.check(force=True)
        if violation:
            stop_reason = "resource_stop"
            turns.append({"turn": turn_index + 1, "resource_violation": violation})
            break
        remaining = MAX_TOTAL_TOKENS - total_tokens
        if remaining <= 0:
            stop_reason = "token_budget_exhausted"
            break
        generation = generate_action(
            model,
            tokenizer,
            messages,
            seed=seed + turn_index,
            temperature=temperature,
            max_tokens=min(MAX_ACTION_TOKENS, remaining),
            guard=guard,
            deadline=deadline,
        )
        if generation["finish"] == "prompt_too_long":
            turns.append(
                {
                    "turn": turn_index + 1,
                    "prompt_tokens": len(generation["prompt_ids"]),
                    "completion_tokens": 0,
                    "finish": "prompt_too_long",
                    "action": None,
                    "parse_error": "tokenized prompt exceeded the frozen per-turn prompt budget",
                }
            )
            stop_reason = "prompt_token_budget_exceeded"
            break
        total_tokens += len(generation["completion_ids"])
        if turn_index == 0:
            first_prompt_ids = list(generation["prompt_ids"])
            first_completion_ids = list(generation["completion_ids"])
            first_sampling_behavior_logprobs = list(
                generation["sampling_behavior_logprobs"]
            )
        action, parse_error = action_from_text(generation["text"])
        row: dict[str, Any] = {
            "turn": turn_index + 1,
            "prompt_token_ids": generation["prompt_ids"],
            "completion_token_ids": generation["completion_ids"],
            "sampling_behavior_logprobs": generation[
                "sampling_behavior_logprobs"
            ],
            "raw_output": generation["text"],
            "finish": generation["finish"],
            "prompt_tokens": len(generation["prompt_ids"]),
            "completion_tokens": len(generation["completion_ids"]),
            "generation_seconds": generation["seconds"],
            "context_strategy": (
                "initial_frozen_task_prompt" if turn_index == 0 else
                "task_plus_compact_prior_tool_trace_plus_latest_exchange"
            ),
            "compacted_prior_tool_turns": turn_index,
            "action": action,
        }
        if parse_error:
            row["parse_error"] = parse_error
            result = {"error": parse_error}
        elif action["tool"] == "finish":
            submitted = True
            submit_summary = str(action["arguments"].get("summary", ""))[:1000]
            row["tool_result"] = {"finished": True}
            turns.append(row)
            stop_reason = "submitted"
            break
        else:
            result = execute_tool(
                baseline,
                task["baseline_root"],
                allowed,
                action,
                task_id=task["record"]["task_id"],
                read_cursors=read_cursors,
            )
            row["tool_result"] = result
        turns.append(row)
        # Retain the task, a short summary of earlier tool turns, and the complete
        # latest action/result pair. The full tokenized prompts and all tool evidence
        # remain in the trajectory record for audit and full-trajectory GRPO credit.
        messages = make_agent_messages(task, compact_action_summary(turns[:-1]) or None)
        messages.append({"role": "assistant", "content": generation["text"]})
        messages.append(
            {
                "role": "user",
                "content": "Tool result:\n"
                + json.dumps(result, ensure_ascii=False, sort_keys=True),
            }
        )
        if generation["resource_violation"]:
            row["resource_violation"] = generation["resource_violation"]
            stop_reason = "resource_stop"
            break
        if generation["finish"] == "wall_time_limit":
            stop_reason = "wall_time_limit"
            break
        if total_tokens >= MAX_TOTAL_TOKENS:
            stop_reason = "token_budget_exhausted"
            break
        if time.monotonic() >= deadline:
            stop_reason = "wall_time_limit"
            break

    source_hashes = {path: sha256_file(baseline / path) for path in sorted(allowed)}
    changed = changed_files(baseline, allowed, task["baseline_root"])
    diff = show_changes(baseline, allowed, task["baseline_root"])["diff"]
    record = {
        "schema": "vare_real_pydantic_agent_trajectory_v1",
        "phase": phase,
        "task_id": task["record"]["task_id"],
        "task_family": task["record"]["task_family"],
        "split": task["record"]["split"],
        "policy_checkpoint_id": policy_id,
        "sample_index": sample_index,
        "seed": seed,
        "temperature": temperature,
        "sampling": {
            "top_p": 1.0 if temperature > 0 else 0.0,
            "top_k": 0,
            "min_p": 0.0,
            "distribution": (
                "untruncated softmax(logits / temperature)"
                if temperature > 0
                else "greedy"
            ),
        },
        "agent_visible_prompt": {
            "system": SYSTEM_PROMPT,
            "user": TOOL_HELP.replace("{prompt}", task["prompt"]).replace(
                "{paths}", ", ".join(task["editable_paths"])
            ),
        },
        "prompt_sha256": sha256_json(make_agent_messages(task)),
        "initial_prompt_token_ids": first_prompt_ids,
        "first_action_completion_token_ids": first_completion_ids,
        "first_action_sampling_behavior_logprobs": first_sampling_behavior_logprobs,
        "tool_turns": len(turns),
        "context_compaction_policy": {
            "strategy": "frozen task plus prior tool-event summary plus latest full exchange",
            "prior_summary_character_cap": 1800,
            "full_tool_trace_preserved": True,
        },
        "generated_tokens": total_tokens,
        "wall_seconds": round(time.monotonic() - started, 6),
        "stop_reason": stop_reason,
        "submitted": submitted,
        "submit_summary": submit_summary,
        "changed_files": changed,
        "candidate_source_hashes": source_hashes,
        "candidate_source_tree_sha256": hash_tree(baseline / "pydantic")[1],
        "candidate_diff": diff,
        "tool_trace": turns,
        "workspace_path": str(baseline),
    }
    record["trajectory_sha256"] = sha256_json(record)
    atomic_json(artifact_dir / "trajectory.json", record)
    return {
        "record": record,
        "workspace": baseline,
        "first_prompt_ids": first_prompt_ids,
        "first_completion_ids": first_completion_ids,
        "first_sampling_behavior_logprobs": first_sampling_behavior_logprobs,
    }


def grade_candidate(
    task: dict[str, Any],
    candidate_workspace: Path,
    artifact_dir: Path,
    *,
    grader_name: str,
) -> dict[str, Any]:
    grade_workspace = artifact_dir / "grader-workspace"
    if grade_workspace.exists():
        raise FileExistsError(f"grader workspace already exists: {grade_workspace}")
    shutil.copytree(
        task["baseline_root"],
        grade_workspace,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    for relative in task["editable_paths"]:
        destination = grade_workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_workspace / relative, destination)
    if sha256_file(grade_workspace / task["test_path"]) != task["test_sha256"]:
        raise RuntimeError("independent grader test changed while preparing candidate")
    for relative, expected in task["source_test"].get("support_files", {}).items():
        if sha256_file(grade_workspace / relative) != expected:
            raise RuntimeError("independent grader support file changed while preparing candidate")
    env_lock = task["env_lock"]
    raw_caps = (
        env_lock.get("resource_caps")
        or env_lock.get("caps")
        or env_lock.get("container_policy")
    )
    if not isinstance(raw_caps, dict):
        raise RuntimeError("pinned grader environment has no resource limits")
    caps = {
        "cpus": raw_caps["cpus"],
        "memory_bytes": raw_caps["memory_bytes"],
        "memory_swap_bytes": raw_caps["memory_swap_bytes"],
        "pids": raw_caps.get("pids", raw_caps.get("pids_limit")),
        "timeout_seconds_per_arm": raw_caps.get(
            "timeout_seconds_per_arm",
            task["definition"]["execution"]["timeout_seconds_per_arm"],
        ),
    }
    if not caps["pids"]:
        raise RuntimeError("pinned grader environment has no PID limit")
    image = env_lock["image_id"]
    container_name = re.sub(r"[^a-zA-Z0-9_.-]", "-", grader_name)[:120]
    command = task["definition"]["execution"]["command"]
    cmd = [
        "docker",
        "--context",
        "colima",
        "run",
        "--name",
        container_name,
        "--pull=never",
        "--network=none",
        f"--memory={caps['memory_bytes']}",
        f"--memory-swap={caps['memory_swap_bytes']}",
        f"--cpus={caps['cpus']}",
        f"--pids-limit={caps['pids']}",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--shm-size=64m",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=256m,mode=1777",
        "--workdir",
        "/workspace",
        "--user=65534:65534",
        "-v",
        f"{grade_workspace}:/workspace:ro",
        "-v",
        f"{CONTAINER_RUNNER}:/container_pytest_runner.py:ro",
        "-e",
        "PYTHONPATH=/workspace",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "-e",
        "HOME=/tmp",
        "-e",
        "TMPDIR=/tmp",
        "--entrypoint=python",
        image,
        *command[1:],
    ]
    started = time.monotonic()
    timeout = int(caps["timeout_seconds_per_arm"])
    timed_out = False
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        exit_code = completed.returncode
        log_text = completed.stdout + completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        partial = exc.stdout or b""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        log_text = str(partial) + f"\nQUAL_TIMEOUT_SECONDS={timeout}\n"
        subprocess.run(
            ["docker", "--context", "colima", "kill", container_name],
            capture_output=True,
            text=True,
            timeout=15,
        )
    elapsed = round(time.monotonic() - started, 3)
    atomic_json(artifact_dir / "grader-command.json", {"command": cmd, "wall_seconds": elapsed})
    log_path = artifact_dir / "grader.log"
    log_path.write_text(log_text, encoding="utf-8")
    try:
        state_result = subprocess.run(
            ["docker", "--context", "colima", "inspect", container_name, "--format", "{{json .State}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        state = json.loads(state_result.stdout) if state_result.returncode == 0 else {}
    except Exception:
        state = {}
    selector_pattern = re.escape(task["selector"]) + r"(?:\[[^\]\r\n]*\])?"
    selector_failure = re.search(
        rf"(?m)^(?:FAILED|ERROR)\s+{selector_pattern}\s*$", log_text
    ) is not None
    completion_marker = re.search(r"(?m)^QUAL_MEMORY_PEAK_BYTES=\d+\s*$", log_text) is not None
    passed_summary = re.search(
        r"(?m)^[^\r\n]*\b\d+\s+passed\b[^\r\n]*\bin \d+(?:\.\d+)?s\b[^\r\n]*$", log_text
    ) is not None
    failed_summary = re.search(
        r"(?m)^[^\r\n]*\b\d+\s+(?:failed|errors?)\b[^\r\n]*\bin \d+(?:\.\d+)?s\b[^\r\n]*$", log_text
    ) is not None
    complete_container = state.get("Status") == "exited" and state.get("OOMKilled") is False
    if exit_code == 0 and not timed_out and completion_marker and passed_summary and complete_container:
        reward = 1.0
        status = "pass"
    elif exit_code in (1, 2) and not timed_out and selector_failure and completion_marker and failed_summary and complete_container:
        reward = 0.0
        status = "candidate_failure"
    else:
        raise RuntimeError(
            f"grader infrastructure or incomplete evidence for {task['record']['task_id']}: "
            f"exit={exit_code}, timed_out={timed_out}, completion_marker={completion_marker}, "
            f"pass_summary={passed_summary}, fail_summary={failed_summary}, "
            f"selector_failure={selector_failure}, container_state={state}"
        )
    test_id = task["definition_sha256"] + ":" + task["test_sha256"] + ":" + image
    result = {
        "schema": "vare_independent_trusted_reward_v1",
        "task_id": task["record"]["task_id"],
        "task_definition_sha256": task["definition_sha256"],
        "grader_test_sha256": task["test_sha256"],
        "environment_lock_sha256": task["env_lock_sha256"],
        "grader_image_id": image,
        "grader_identity": sha256_bytes(test_id.encode()),
        "candidate_source_hashes": {
            relative: sha256_file(candidate_workspace / relative)
            for relative in task["editable_paths"]
        },
        "exit_code": exit_code,
        "container_status": state.get("Status"),
        "oom_killed": state.get("OOMKilled"),
        "completion_marker_present": completion_marker,
        "pytest_pass_summary_present": passed_summary,
        "pytest_failure_summary_present": failed_summary,
        "exact_selector_failure_present": selector_failure,
        "timed_out": timed_out,
        "wall_seconds": elapsed,
        "reward": reward,
        "grade_status": status,
        "candidate_feedback": False,
        "network": "none",
        "grader_log_path": str(log_path),
        "grader_log_sha256": sha256_file(log_path),
    }
    atomic_json(artifact_dir / "grade.json", result)
    return result


def sample_seed(phase: str, task_id: str, index: int) -> int:
    value = f"{BASE_SEED}:{phase}:{task_id}:{index}".encode()
    return int(hashlib.sha256(value).hexdigest()[:8], 16)


def load_policy(*, lora: bool, seed: int) -> tuple[Any, Any]:
    from mlx_lm import load
    import mlx.core as mx
    from vare import grpo_mlx

    model, tokenizer = load(str(MODEL_PATH))
    if lora:
        grpo_mlx.make_lora_policy(model, **LORA, seed=seed)
    mx.eval(model.parameters())
    return model, tokenizer


def gradient_checkpointing(model: Any):
    from contextlib import contextmanager
    from mlx_lm.tuner.trainer import grad_checkpoint

    @contextmanager
    def active():
        layer = model.layers[0]
        cls = type(layer)
        original = cls.__call__
        grad_checkpoint(layer)
        try:
            yield
        finally:
            cls.__call__ = original

    return active()


def parameter_snapshot(model: Any) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    mx.eval(model.trainable_parameters())
    return {
        key: np.asarray(value).astype(np.float32, copy=True)
        for key, value in tree_flatten(model.trainable_parameters())
    }


def parameter_delta(before: dict[str, Any], model: Any) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    mx.eval(model.trainable_parameters())
    after = {
        key: np.asarray(value).astype(np.float32, copy=False)
        for key, value in tree_flatten(model.trainable_parameters())
    }
    if set(before) != set(after):
        raise RuntimeError("trainable parameter key set changed during update")
    l2_sq = 0.0
    max_abs = 0.0
    changed = 0
    total = 0
    finite = True
    for key in before:
        delta = after[key] - before[key]
        finite = finite and bool(np.isfinite(delta).all())
        l2_sq += float(np.square(delta, dtype=np.float64).sum())
        max_abs = max(max_abs, float(np.abs(delta).max(initial=0.0)))
        changed += int(np.count_nonzero(delta))
        total += int(delta.size)
    return {
        "l2": math.sqrt(l2_sq),
        "max_abs": max_abs,
        "changed_elements": changed,
        "total_elements": total,
        "finite": finite,
    }


def compare_sampling_and_grpo_behavior_logprobs(samples: list[Any]) -> dict[str, Any]:
    differences: list[float] = []
    action_segments = 0
    for sample in samples:
        segments = sample.trajectory_segments
        if not segments:
            raise RuntimeError("GRPO sample is missing its full policy trajectory")
        for segment in segments:
            sampled = segment.sampling_behavior_logprobs
            trainer = segment.behavior_logprobs
            if not isinstance(sampled, list) or not isinstance(trainer, list):
                raise RuntimeError("trajectory segment is missing sampling-time or GRPO behavior log-probabilities")
            if len(sampled) != len(segment.completion_ids) or len(trainer) != len(segment.completion_ids):
                raise RuntimeError("sampling-time and GRPO behavior log-probability lengths differ")
            for left, right in zip(sampled, trainer, strict=True):
                if not math.isfinite(left) or not math.isfinite(right):
                    raise RuntimeError("behavior log-probabilities must be finite")
                differences.append(abs(float(left) - float(right)))
            action_segments += 1
    tolerance = 1e-6
    return {
        "token_count": len(differences),
        "action_segment_count": action_segments,
        "max_abs_difference": max(differences, default=0.0),
        "mean_abs_difference": math.fsum(differences) / len(differences) if differences else 0.0,
        "matches_within_grpo_tolerance": all(value <= tolerance for value in differences),
        "grpo_on_policy_tolerance": tolerance,
    }



def grade_rollout_set(
    model: Any,
    tokenizer: Any,
    task: dict[str, Any],
    *,
    phase: str,
    policy_id: str,
    count: int,
    start_index: int = 0,
    temperature: float,
    guard: ResourceGuard,
) -> list[dict[str, Any]]:
    results = []
    for index in range(start_index, start_index + count):
        artifact_dir = RUN_DIR / phase / task["record"]["task_id"] / f"sample-{index:02d}"
        if artifact_dir.exists():
            raise FileExistsError(f"refusing to overwrite rollout artifacts: {artifact_dir}")
        artifact_dir.mkdir(parents=True)
        workspace = WORK / "candidate-workspaces" / RUN_DIR.name / phase / task["record"]["task_id"] / f"sample-{index:02d}"
        agent = run_agent(
            model,
            tokenizer,
            task,
            workspace=workspace,
            artifact_dir=artifact_dir,
            policy_id=policy_id,
            phase=phase,
            sample_index=index,
            seed=sample_seed(phase, task["record"]["task_id"], index),
            temperature=temperature,
            guard=guard,
        )
        if agent["record"]["stop_reason"] == "resource_stop":
            return [
                {
                    "resource_violation": True,
                    "task_id": task["record"]["task_id"],
                    "sample_index": index,
                    "trajectory_path": str(artifact_dir / "trajectory.json"),
                }
            ]
        grade = grade_candidate(
            task,
            agent["workspace"],
            artifact_dir,
            grader_name=f"vare-pydantic-{phase}-{task['record']['task_id']}-{index}",
        )
        result = {
            "task_id": task["record"]["task_id"],
            "sample_index": index,
            "reward": grade["reward"],
            "grade_status": grade["grade_status"],
            "trajectory_path": str(artifact_dir / "trajectory.json"),
            "grade_path": str(artifact_dir / "grade.json"),
            "trajectory_sha256": sha256_file(artifact_dir / "trajectory.json"),
            "grade_sha256": sha256_file(artifact_dir / "grade.json"),
            "workspace": str(agent["workspace"]),
            "first_prompt_ids": agent["first_prompt_ids"],
            "first_completion_ids": agent["first_completion_ids"],
            "first_sampling_behavior_logprobs": agent[
                "first_sampling_behavior_logprobs"
            ],
            "grader_identity": grade["grader_identity"],
            "policy_checkpoint_id": policy_id,
            "behavior_step": 0,
        }
        atomic_json(artifact_dir / "sample-result.json", result)
        results.append(result)
        violation = guard.check(force=True)
        if violation:
            return [{"resource_violation": True, "task_id": task["record"]["task_id"], "violation": violation}]
    return results



def make_grpo_samples(results: list[dict[str, Any]], task: dict[str, Any]):
    from vare.grpo_mlx import Sample, TrajectorySegment

    samples = []
    for row in results:
        if row.get("resource_violation"):
            raise RuntimeError("cannot form GRPO samples after a resource stop")
        trajectory = json.loads(Path(row["trajectory_path"]).read_text(encoding="utf-8"))
        segments = []
        for turn in trajectory["tool_trace"]:
            prompt = turn.get("prompt_token_ids")
            completion = turn.get("completion_token_ids")
            sampling_logprobs = turn.get("sampling_behavior_logprobs")
            if not completion:
                continue
            if not isinstance(prompt, list) or not isinstance(completion, list):
                raise RuntimeError("trajectory action segment lacks prompt or completion token IDs")
            if not isinstance(sampling_logprobs, list) or len(sampling_logprobs) != len(completion):
                raise RuntimeError("trajectory action segment has incomplete sampling behavior log-probabilities")
            if any(
                value is None or not math.isfinite(float(value)) or float(value) > 0
                for value in sampling_logprobs
            ):
                raise RuntimeError("trajectory action segment has invalid sampling behavior log-probabilities")
            segments.append(TrajectorySegment(
                prompt_ids=[int(x) for x in prompt],
                completion_ids=[int(x) for x in completion],
                sampling_behavior_logprobs=[float(value) for value in sampling_logprobs],
            ))
        if not segments:
            raise RuntimeError("rollout contains no policy-generated action segments")
        if segments[0].prompt_ids != row["first_prompt_ids"] or segments[0].completion_ids != row["first_completion_ids"]:
            raise RuntimeError("sample-result first action differs from the retained full trajectory")
        samples.append(Sample(
            instance_id=task["record"]["task_id"],
            behavior_checkpoint_id=row["policy_checkpoint_id"],
            behavior_step=0,
            prompt_ids=segments[0].prompt_ids,
            completion_ids=segments[0].completion_ids,
            reward=float(row["reward"]),
            verifier=row["grader_identity"],
            sample_index=int(row["sample_index"]),
            behavior_logprobs=None,
            meta={
                "task_definition_sha256": task["definition_sha256"],
                "trajectory_sha256": row["trajectory_sha256"],
                "grader_identity": row["grader_identity"],
                "credit_assignment": "full_trajectory_terminal_reward",
                "action_segment_count": len(segments),
            },
            trajectory_segments=segments,
        ))
    return samples


def save_training_sample_rows(samples: list[Any], results: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for sample, result in zip(samples, results, strict=True):
        segments = sample.trajectory_segments or []
        rows.append({
            "schema": "vare_real_grpo_sample_full_trajectory_v1",
            "instance_id": sample.instance_id,
            "behavior_checkpoint_id": sample.behavior_checkpoint_id,
            "behavior_step": sample.behavior_step,
            "sample_index": sample.sample_index,
            "action_segments": [
                {
                    "turn_index": index + 1,
                    "prompt_token_ids": segment.prompt_ids,
                    "completion_token_ids": segment.completion_ids,
                    "sampling_behavior_logprobs": segment.sampling_behavior_logprobs,
                    "grpo_ratio_behavior_logprobs": segment.behavior_logprobs,
                }
                for index, segment in enumerate(segments)
            ],
            "action_segment_count": len(segments),
            "grpo_ratio_behavior_logprob_source": (
                "same-policy differentiable MLX forward used by the #195 GRPO integrity path"
            ),
            "trusted_reward": sample.reward,
            "grader_identity": sample.verifier,
            "trajectory_sha256": result["trajectory_sha256"],
            "credit_assignment": "full_trajectory_terminal_reward",
        })
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")



def evaluate_policy(
    *,
    updated: bool,
    checkpoint_path: Path | None,
    guard: ResourceGuard,
    run_lock: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import mlx.core as mx
    import mlx.optimizers as optim
    from vare import grpo_mlx

    model, tokenizer = load_policy(lora=updated, seed=BASE_SEED)
    optimizer = None
    reload_record = {"reloaded": False}
    if updated:
        if checkpoint_path is None:
            raise RuntimeError("updated policy evaluation requires a checkpoint")
        optimizer = optim.Adam(learning_rate=OPTIMIZER_CONFIG["learning_rate"])
        manifest = grpo_mlx.load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            base_model_id=BASE_MODEL_ID,
        )
        loaded_digest = grpo_mlx.trainable_digest(model)
        if loaded_digest != manifest["trainable_digest"]:
            raise RuntimeError("fresh checkpoint reload parameter digest mismatch")
        if int(optimizer.state["step"].item()) != 1 or manifest["step"] != 1:
            raise RuntimeError("fresh checkpoint reload did not restore exactly one optimizer step")
        policy_id = f"{BASE_MODEL_ID}:update-1:{loaded_digest[:20]}"
        reload_record = {
            "reloaded": True,
            "checkpoint_manifest_sha256": sha256_file(checkpoint_path / "checkpoint.json"),
            "checkpoint_adapter_sha256": sha256_file(checkpoint_path / "adapters.safetensors"),
            "checkpoint_optimizer_sha256": sha256_file(checkpoint_path / "optimizer.safetensors"),
            "loaded_trainable_parameters_sha256": loaded_digest,
            "optimizer_step": int(optimizer.state["step"].item()),
            "manifest_step": int(manifest["step"]),
            "base_model_id": manifest["base_model_id"],
        }
    else:
        policy_id = BASE_MODEL_ID + ":unchanged"
    all_results = []
    for index, record in enumerate(run_lock["development_tasks"]):
        task = load_task(record)
        verify_task_workspace(task)
        phase = "development-updated" if updated else "development-unchanged"
        attempt_dir = RUN_DIR / phase / record["task_id"]
        if attempt_dir.exists():
            raise FileExistsError(f"refusing to overwrite evaluation: {attempt_dir}")
        attempt_dir.mkdir(parents=True)
        workspace = WORK / "candidate-workspaces" / RUN_DIR.name / phase / record["task_id"] / "sample-00"
        agent = run_agent(
            model,
            tokenizer,
            task,
            workspace=workspace,
            artifact_dir=attempt_dir,
            policy_id=policy_id,
            phase=phase,
            sample_index=0,
            seed=sample_seed("development-paired", record["task_id"], 0),
            temperature=DEV_TEMPERATURE,
            guard=guard,
        )
        if agent["record"]["stop_reason"] == "resource_stop":
            raise RuntimeError("resource guard stopped Development evaluation")
        grade = grade_candidate(
            task,
            agent["workspace"],
            attempt_dir,
            grader_name=f"vare-pydantic-{phase}-{record['task_id']}",
        )
        row = {
            "task_id": record["task_id"],
            "task_family": record["task_family"],
            "policy": "one-update candidate policy" if updated else "unchanged pretrained policy",
            "policy_checkpoint_id": policy_id,
            "reward": grade["reward"],
            "grade_status": grade["grade_status"],
            "tool_turns": agent["record"]["tool_turns"],
            "generated_tokens": agent["record"]["generated_tokens"],
            "wall_seconds": agent["record"]["wall_seconds"],
            "changed_files": agent["record"]["changed_files"],
            "candidate_source_hashes": agent["record"]["candidate_source_hashes"],
            "trajectory_sha256": sha256_file(attempt_dir / "trajectory.json"),
            "grader_identity": grade["grader_identity"],
        }
        atomic_json(attempt_dir / "evaluation-result.json", row)
        all_results.append(row)
        violation = guard.check(force=True)
        if violation:
            raise RuntimeError(f"resource guard stopped Development evaluation: {violation}")
    del optimizer
    del model
    mx.clear_cache()
    return all_results, reload_record


def runtime_versions() -> dict[str, str]:
    names = ["mlx", "mlx-lm", "numpy", "huggingface-hub"]
    return {name: importlib.metadata.version(name) for name in names}


def run_experiment() -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(INTEGRATION / "src"))

    lock, assignment, records = verify_frozen_inputs()
    current_arch = platform.machine()
    if current_arch != "arm64":
        raise RuntimeError(f"MLX experiment requires arm64 Apple Silicon, got {current_arch}")
    if platform.system() != "Darwin":
        raise RuntimeError("MLX experiment requires macOS")
    memory = current_memory_state()
    guard = ResourceGuard(
        memory,
        free_floor_pct=float(lock["resources"]["memory_free_floor_pct"]),
        swap_growth_allowed_bytes=int(lock["resources"]["swap_growth_allowed_bytes"]),
    )
    violation = guard.check(force=True)
    if violation:
        raise RuntimeError(f"resource preflight failed: {violation}")
    disk = shutil.disk_usage(ROOT)
    if disk.free < int(lock["resources"]["minimum_disk_free_bytes"]):
        raise RuntimeError("resource preflight failed: insufficient free disk")
    for record in records:
        verify_task_workspace(load_task(record))

    import mlx
    import mlx.core as mx
    from vare import grpo_mlx

    if not mx.metal.is_available() or "gpu" not in str(mx.default_device()).lower():
        raise RuntimeError("MLX Metal GPU is unavailable")
    versions = runtime_versions()
    if versions != lock["runtime"]["versions"]:
        raise RuntimeError(f"MLX runtime versions differ from frozen run lock: {versions}")
    if RUN_DIR.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True)
    run_meta = {
        "schema": "vare_real_model_train_dev_run_v1",
        "experiment_id": lock["experiment_id"],
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assignment_sha256": sha256_file(ASSIGNMENT_PATH),
        "run_lock_sha256": sha256_file(RUN_LOCK_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "grpo_mlx_sha256": sha256_file(GRPO_PATH),
        "model": {
            "id": MODEL_ID, "revision": MODEL_REVISION,
            "files_tree_sha256": lock["model"]["files_tree_sha256"],
        },
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "mlx_module_version": versions["mlx"], "versions": versions,
            "device": str(mx.default_device()), "metal_available": mx.metal.is_available(),
        },
        "resource_baseline": memory, "disk_free_bytes_at_start": disk.free,
        "train_task_ids": assignment["training"]["task_ids"],
        "development_task_ids": assignment["evaluation"]["task_ids"],
        "development_exposed_before_training": False,
        "prior_train_rollouts_attempted": NEW_ROLLOUT_START_INDEX,
        "authorized_train_rollout_exposure_budget": NEW_ROLLOUT_COUNT,
        "task_cumulative_exposure_cap": NEW_ROLLOUT_START_INDEX + NEW_ROLLOUT_COUNT,
        "new_training_sample_indices": list(range(NEW_ROLLOUT_START_INDEX, NEW_ROLLOUT_START_INDEX + NEW_ROLLOUT_COUNT)),
        "prior_negative_train_experiment_path": "work/VARE-rsi-integration-20261105/results/vare-pydantic-14138-full-trajectory-v1/attempt-009/summary.json",
        "prompt_policy_version": "pydantic-14138-prompt-v7-scope-directed",
        "tool_adapter_version": TOOL_ADAPTER_VERSION,
        "network_policy": "model weights local; rollout agent has no shell/network tools; graders use Docker --network=none",
        "credit_assignment": assignment["training"]["credit_assignment"],
    }
    atomic_json(RUN_DIR / "run_meta.json", run_meta)
    atomic_json(RUN_DIR / "progress.json", {"stage": "preflight_passed", "resource_state": memory})

    train_record = next(row for row in records if row["split"] == "train")
    train_task = load_task(train_record)
    mx.reset_peak_memory()
    model, tokenizer = load_policy(lora=True, seed=BASE_SEED)
    violation = guard.check(force=True)
    if violation:
        atomic_json(RUN_DIR / "abort.json", {"stage": "model_loaded", "resource_violation": violation})
        del model
        raise RuntimeError(f"resource stop after model load: {violation}")
    initial_digest = grpo_mlx.trainable_digest(model)
    expected_initial_digest = lock.get("expected_initial_trainable_parameters_sha256")
    if expected_initial_digest and initial_digest != expected_initial_digest:
        raise RuntimeError("fresh pretrained policy adapter differs from the frozen carryover policy")
    policy_id = f"{BASE_MODEL_ID}:trainable-initial:{initial_digest[:20]}"
    atomic_json(
        RUN_DIR / "base-policy.json",
        {"policy_checkpoint_id": policy_id, "base_model_id": BASE_MODEL_ID,
         "trainable_parameters_sha256": initial_digest, "lora": LORA, "optimizer": OPTIMIZER_CONFIG},
    )
    atomic_json(
        RUN_DIR / "progress.json",
        {"stage": "model_loaded", "policy_checkpoint_id": policy_id,
         "peak_mlx_bytes": int(mx.get_peak_memory()), "resource_state": guard.last_state},
    )

    train_results = grade_rollout_set(
        model, tokenizer, train_task, phase="training", policy_id=policy_id,
        count=NEW_ROLLOUT_COUNT, start_index=NEW_ROLLOUT_START_INDEX,
        temperature=TEMPERATURE, guard=guard,
    )
    if any(row.get("resource_violation") for row in train_results):
        atomic_json(RUN_DIR / "abort.json", {"stage": "training_rollouts", "details": train_results})
        del model
        raise RuntimeError("resource guard stopped training rollout generation or grading")
    if len(train_results) != GROUP_SIZE:
        raise RuntimeError(f"incomplete Train group: expected {GROUP_SIZE}, got {len(train_results)}")
    rewards = [float(row["reward"]) for row in train_results]
    atomic_json(
        RUN_DIR / "training-rollout-summary.json",
        {"task_id": train_record["task_id"], "n_rollouts": len(train_results),
         "rewards": rewards, "reward_counts": {"0": rewards.count(0.0), "1": rewards.count(1.0)},
         "all_rollout_sha256": [row["trajectory_sha256"] for row in train_results],
         "all_grader_sha256": [row["grade_sha256"] for row in train_results]},
    )
    samples = make_grpo_samples(train_results, train_task)
    if len(samples) != GROUP_SIZE:
        raise RuntimeError("GRPO sample builder did not preserve the complete frozen group")
    save_training_sample_rows(samples, train_results, RUN_DIR / "training-samples.jsonl")
    groups = grpo_mlx.group_samples(samples, group_size=GROUP_SIZE)
    if len(groups) != 1:
        raise RuntimeError(f"expected one same-task on-policy group, got {len(groups)}")
    advantages, informative = grpo_mlx.group_advantages(rewards, normalize_std=True)
    if not informative:
        atomic_json(
            RUN_DIR / "training-update.json",
            {"update_applied": False, "reason": "no_variable_trusted_rewards",
             "rewards": rewards, "group_size": GROUP_SIZE,
             "policy_checkpoint_id": policy_id, "optimizer_updates": 0,
             "trajectory_hashes": [row["trajectory_sha256"] for row in train_results],
             "grader_hashes": [row["grade_sha256"] for row in train_results],
             "credit_assignment": "full_trajectory_terminal_reward"},
        )
        del samples, model
        mx.clear_cache()
        atomic_json(RUN_DIR / "progress.json", {"stage": "no_informative_group", "rewards": rewards})
        return 2

    config = grpo_mlx.GRPOConfig(
        clip_eps_low=0.2, clip_eps_high=0.2, normalize_std=True,
        temperature=TEMPERATURE, max_policy_lag=0, max_grad_norm=1.0,
        group_size=GROUP_SIZE,
    )
    informative_samples = grpo_mlx.informative_samples(samples, config)
    if len(informative_samples) != GROUP_SIZE:
        raise RuntimeError("informative group does not contain every member of the frozen group")
    grpo_mlx.check_exact_ratio_sampling(temperature=TEMPERATURE, top_p=1.0, top_k=0, min_p=0.0)
    before_snapshot = parameter_snapshot(model)
    with gradient_checkpointing(model):
        grpo_mlx.compute_behavior_logprobs(
            model, informative_samples, config,
            after_sample=lambda: guard.check(force=True)
            and (_ for _ in ()).throw(RuntimeError("resource guard stopped behavior log-probability pass")),
        )
        behavior_logprob_comparison = compare_sampling_and_grpo_behavior_logprobs(informative_samples)
        if not behavior_logprob_comparison["matches_within_grpo_tolerance"]:
            raise RuntimeError("sampling-time and GRPO on-policy behavior log-probabilities differ")
        preflight_grads, gradient_stats = grpo_mlx.policy_gradient(
            model, list(zip(informative_samples, advantages, strict=True)), config,
            n_tokens=sum(len(segment.completion_ids) for sample in informative_samples for segment in sample.trajectory_segments),
            policy_step=0,
        )
        mx.eval(preflight_grads)
        from mlx.utils import tree_flatten
        import numpy as np
        gradient_values = [np.asarray(value).astype(np.float64, copy=False) for _, value in tree_flatten(preflight_grads)]
        grad_l2 = math.sqrt(sum(float(np.square(value).sum()) for value in gradient_values))
        grad_finite = all(bool(np.isfinite(value).all()) for value in gradient_values)
        del preflight_grads
        mx.clear_cache()
        if not grad_finite or not math.isfinite(grad_l2) or grad_l2 <= 0:
            atomic_json(
                RUN_DIR / "training-update.json",
                {"update_applied": False, "reason": "gradient_not_finite_and_nonzero",
                 "gradient_l2": grad_l2, "gradient_finite": grad_finite,
                 "gradient_stats": gradient_stats, "rewards": rewards,
                 "advantages": advantages, "optimizer_updates": 0},
            )
            del samples, model
            mx.clear_cache()
            return 3
        violation = guard.check(force=True)
        if violation:
            atomic_json(RUN_DIR / "abort.json", {"stage": "before_optimizer_update", "resource_violation": violation})
            del samples, model
            raise RuntimeError(f"resource stop before optimizer mutation: {violation}")
        optimizer = __import__("mlx.optimizers", fromlist=["Adam"]).Adam(
            learning_rate=OPTIMIZER_CONFIG["learning_rate"]
        )
        update = grpo_mlx.train_step(
            model, optimizer, samples, config,
            policy_checkpoint_id=policy_id, policy_step=0,
            after_sample=lambda: guard.check(force=True)
            and (_ for _ in ()).throw(RuntimeError("resource guard stopped GRPO update")),
        )
    mx.eval(model.parameters(), optimizer.state)
    delta = parameter_delta(before_snapshot, model)
    update["preflight_gradient_l2"] = grad_l2
    update["preflight_gradient_finite"] = grad_finite
    update["preflight_gradient_stats"] = gradient_stats
    update["sampling_vs_grpo_behavior_logprobs"] = behavior_logprob_comparison
    update["parameter_delta"] = delta
    update["reward_advantages"] = advantages
    update["credit_assignment"] = "full_trajectory_terminal_reward"
    update["params_changed"] = bool(update.get("params_changed") and delta["changed_elements"] > 0)
    if not update.get("update_applied") or not update["params_changed"] or not delta["finite"] or int(optimizer.state["step"].item()) != 1:
        update["update_applied"] = False
        update["reason"] = "optimizer_step_failed_parameter_change_or_finite_delta_check"
        atomic_json(RUN_DIR / "training-update.json", update)
        del optimizer, samples, model
        mx.clear_cache()
        return 4

    save_training_sample_rows(samples, train_results, RUN_DIR / "training-samples.jsonl")
    checkpoint = RUN_DIR / "checkpoints/update-0001"
    checkpoint_manifest = grpo_mlx.save_checkpoint(
        checkpoint, model, optimizer, base_model_id=BASE_MODEL_ID, step=1,
        base_seed=BASE_SEED,
        extra={"experiment_id": lock["experiment_id"],
               "assignment_sha256": sha256_file(ASSIGNMENT_PATH),
               "training_task_id": train_record["task_id"],
               "policy_checkpoint_id_before": policy_id,
               "optimizer": OPTIMIZER_CONFIG, "lora": LORA,
               "credit_assignment": "full_trajectory_terminal_reward",
               "rollout_hashes": [row["trajectory_sha256"] for row in train_results],
               "grader_hashes": [row["grade_sha256"] for row in train_results]},
    )
    update["checkpoint_manifest_sha256"] = sha256_file(checkpoint / "checkpoint.json")
    update["checkpoint_files"] = checkpoint_manifest["files"]
    update["optimizer_step"] = int(optimizer.state["step"].item())
    atomic_json(RUN_DIR / "training-update.json", update)
    del optimizer, samples, before_snapshot, model
    mx.clear_cache()

    violation = guard.check(force=True)
    if violation:
        atomic_json(RUN_DIR / "abort.json", {"stage": "after_checkpoint", "resource_violation": violation})
        raise RuntimeError(f"resource stop before Development evaluation: {violation}")
    base_eval, _ = evaluate_policy(updated=False, checkpoint_path=None, guard=guard, run_lock=lock)
    updated_eval, reload_record = evaluate_policy(updated=True, checkpoint_path=checkpoint, guard=guard, run_lock=lock)
    atomic_json(RUN_DIR / "checkpoint-reload-verification.json", reload_record)
    by_base = {row["task_id"]: row for row in base_eval}
    by_updated = {row["task_id"]: row for row in updated_eval}
    paired = []
    for task_id in sorted(by_base):
        before, after = by_base[task_id], by_updated[task_id]
        paired.append({"task_id": task_id, "task_family": before["task_family"],
                       "unchanged_reward": before["reward"], "updated_reward": after["reward"],
                       "delta": after["reward"] - before["reward"],
                       "matched_attempt_budget": True,
                       "grader_identity_matches": before["grader_identity"] == after["grader_identity"]})
    summary = {
        "schema": "vare_real_model_train_dev_result_v1", "experiment_id": lock["experiment_id"],
        "training": {"task_id": train_record["task_id"], "rollouts": len(train_results),
                     "rewards": rewards, "update_applied": update["update_applied"],
                     "optimizer_step": update["optimizer_step"], "params_changed": update["params_changed"],
                     "parameter_delta": delta, "gradient_l2": grad_l2,
                     "sampling_vs_grpo_behavior_logprobs": behavior_logprob_comparison},
        "checkpoint_reload": reload_record,
        "evaluation": {"split": "development", "tasks": paired,
                       "unchanged_passed": sum(row["unchanged_reward"] == 1 for row in paired),
                       "updated_passed": sum(row["updated_reward"] == 1 for row in paired),
                       "n": len(paired), "matched_budget": True, "promotion": False},
        "credit_assignment": "terminal reward is assigned to every sampled policy action segment in the retained multi-turn trajectory; this is an implementation path, not evidence of completed Long-Horizon RL",
        "heldout_access": False, "heldout_claim": False,
        "started_at_utc": run_meta["started_at_utc"],
        "finished_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resource_end_state": guard.last_state, "peak_mlx_bytes": int(mx.get_peak_memory()),
    }
    atomic_json(RUN_DIR / "evaluation-summary.json", summary)
    atomic_json(RUN_DIR / "progress.json", {"stage": "completed", "summary_sha256": sha256_file(RUN_DIR / "evaluation-summary.json")})
    return 0


def main() -> int:
    try:
        return run_experiment()
    except BaseException as exc:
        if RUN_DIR.exists():
            atomic_json(
                RUN_DIR / "fatal-error.json",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": __import__("traceback").format_exc(),
                    "recorded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
