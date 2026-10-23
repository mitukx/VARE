"""Freeze the Development-only Qwen3-4B feasibility protocol."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/coding_agent_phase_a_v6"
PROTOCOL = STUDY / "protocol-v1.json"
LOCK = STUDY / "protocol-v1.lock.json"
TASKS = tuple(sorted((STUDY / "tasks").glob("*/task.json")))
MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
EXPECTED_MODEL_WEIGHT_SHA256 = "2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f"
IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True, timeout=15,
    ).stdout.strip()


def system_swap() -> dict[str, int]:
    raw = subprocess.run(["sysctl", "vm.swapusage"], check=True, capture_output=True, text=True, timeout=10).stdout
    fields = {}
    for name in ("total", "used", "free"):
        marker = f"{name} = "
        start = raw.index(marker) + len(marker)
        end = raw.index("M", start)
        fields[f"{name}_bytes"] = int(float(raw[start:end]) * 1024**2)
    return fields


def available_memory_bytes() -> int:
    raw = subprocess.run(["vm_stat"], check=True, capture_output=True, text=True, timeout=10).stdout
    page_match = re.search(r"page size of (\d+) bytes", raw)
    if page_match is None:
        raise RuntimeError("vm_stat page size is unavailable")
    page_size = int(page_match.group(1))
    pages: dict[str, int] = {}
    for match in re.finditer(r"Pages (free|inactive|speculative):\s+([\d.]+)", raw):
        pages[match.group(1)] = int(match.group(2).replace(".", ""))
    if not all(name in pages for name in ("free", "inactive", "speculative")):
        raise RuntimeError("vm_stat available page counters are missing")
    return sum(pages.values()) * page_size


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes())
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def freeze(model_snapshot: Path) -> dict[str, Any]:
    if PROTOCOL.exists() or LOCK.exists():
        raise FileExistsError("refusing to rewrite the frozen coding-agent Development protocol")
    if git("status", "--porcelain"):
        raise RuntimeError("freeze the experiment only from a clean worktree")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("the frozen experiment requires Apple Silicon macOS")
    if len(TASKS) != 4:
        raise RuntimeError("exactly four task descriptors are required")
    model_root = model_snapshot.resolve(strict=True)
    if model_root.name != MODEL_REVISION or not model_root.is_dir():
        raise RuntimeError("model directory must be the frozen Qwen3-4B snapshot revision")
    model_files = tree_hashes(model_root)
    if model_files.get("model.safetensors") != EXPECTED_MODEL_WEIGHT_SHA256:
        raise RuntimeError("Qwen3-4B weight hash differs from the expected local checkpoint")
    task_records: dict[str, dict[str, Any]] = {}
    for task_path in TASKS:
        task_bytes = task_path.read_bytes()
        task = json.loads(task_bytes)
        task_id = task["task_id"]
        task_records[task_id] = {
            "descriptor_path": task_path.relative_to(ROOT).as_posix(),
            "descriptor_sha256": sha256(task_bytes),
            "prompt_sha256": sha256(task["prompt"].encode()),
            "task_id": task_id,
            "split": task["metadata"]["split"],
            "family": task["metadata"]["family"],
            "editable_files": task["metadata"]["editable_files"],
            "base_revision": task["source"]["base_revision"],
            "fixed_revision": task["source"]["fixed_revision_sha256"],
            "workspace_snapshot_sha256": task["source"]["snapshot_sha256"],
            "target_sha256": task["source"]["baseline_target_sha256"],
        }
        if task["metadata"]["split"] != "development":
            raise RuntimeError("only Development tasks can enter this protocol")
    available_memory = available_memory_bytes()
    swap = system_swap()
    if available_memory < 4 * 1024**3 or swap["free_bytes"] < 512 * 1024**2:
        raise RuntimeError("resource preflight is below the frozen safe-start floor")
    implementation_paths = (
        "scripts/run_local_agent.py",
        "src/vare/agent_task_contract.py",
        "src/vare/agent_structured_edit.py",
        "scripts/grade_transformers_fresh_dev_v7.py",
        "scripts/transformers_fresh_dev_worker_v3.py",
        "scripts/grade_transformers_phase_a_v1.py",
    )
    protocol = {
        "protocol_id": "vare-qwen3-4b-coding-agent-development-v6-2026-10-21",
        "implementation_commit": git("rev-parse", "HEAD"),
        "model": {
            "repository": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_files_sha256": model_files,
            "backend": "MLX/Metal",
            "trust_remote_code": False,
        },
        "hardware": {
            "model": subprocess.run(["sysctl", "-n", "hw.model"], check=True, capture_output=True, text=True).stdout.strip(),
            "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=True, capture_output=True, text=True).stdout.strip(),
            "unified_memory_bytes": int(subprocess.run(["sysctl", "-n", "hw.memsize"], check=True, capture_output=True, text=True).stdout.strip()),
            "platform": platform.platform(),
        },
        "software": {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "docker_context": subprocess.run(["docker", "context", "show"], check=True, capture_output=True, text=True).stdout.strip(),
            "grading_image": IMAGE,
        },
        "resource_preflight": {
            "available_memory_bytes": available_memory,
            "swap": swap,
        },
        "task_split": {
            "train": [],
            "development": list(task_records),
            "heldout": [],
            "heldout_access": "forbidden",
        },
        "tasks": task_records,
        "sampling": {
            "greedy": {"seeds": [0], "do_sample": False},
            "sampled": {"seeds": [101, 202], "do_sample": True, "temperature": 0.3, "top_p": 0.9, "top_k": 20},
            "rollouts_per_task": 3,
            "total_planned_rollouts": len(task_records) * 3,
        },
        "agent": {
            "edit_format": "structured",
            "public_tests": "run immutable public test functions in the network-disabled candidate container after every applied edit",
            "hidden_grading": "host-side independent v6 grader in a separate execution boundary after each rollout",
            "resource_enforcement": "run_local_agent checks available memory and swap before model load and each turn, enforces the per-turn MLX peak, and checks the shared absolute study deadline between generations",
            "shared_deadline_argument": "--study-deadline-epoch; every rollout in this frozen study must receive the same start_epoch + maximum_total_wall_seconds deadline",
            "max_turns": 12,
            "max_new_tokens_per_turn": 384,
            "max_total_new_tokens": 2048,
            "max_context_tokens": 6144,
            "wall_seconds_per_rollout": 600,
            "max_no_tool_retries": 2,
            "public_test_memory_mb": 1900,
        },
        "limits": {
            "maximum_total_wall_seconds": 7200,
            "minimum_available_memory_bytes": 4 * 1024**3,
            "minimum_free_swap_bytes": 512 * 1024**2,
            "maximum_swap_used_bytes": 7 * 1024**3,
            "maximum_mlx_peak_bytes_per_turn": 16 * 1024**3,
            "stop_on_first_safety_limit": True,
        },
        "primary_metrics": [
            "distinct Development tasks with an agent-generated patch accepted by the independent grader",
            "task-level pass@3 among one greedy and two sampled rollouts",
            "reward variance within the two-sample group for each task",
        ],
        "secondary_metrics": [
            "valid tool-call rate", "applied-edit rate", "public-test pass rate", "format failures",
            "generation tokens", "prompt/decode tokens per second", "wall time", "peak MLX memory",
            "candidate grading time", "cost per verified success",
        ],
        "decision_rule": {
            "phase_a_go": "at least two distinct Development tasks have at least one model-generated independently verified success and sampled rewards are non-constant across at least one task",
            "training_go": "false unless Phase A GO plus independent unused held-out qualification and validated real optimizer/checkpoint path are separately established",
            "failure": "record no-go without changing tasks, prompts, model, seeds, or budgets",
        },
        "limitations": [
            "Development-only feasibility evidence; no held-out result or capability-improvement claim.",
            "All tasks are from Transformers; only bug families differ.",
            "Upstream PR dates do not establish model training-data novelty.",
            "No SFT, GRPO, or policy update is part of this protocol.",
        ],
    }
    protocol_bytes = (json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode()
    PROTOCOL.write_bytes(protocol_bytes)
    lock = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_bytes),
        "implementation_sha256": {
            name: sha256((ROOT / name).read_bytes()) for name in implementation_paths
        },
        "model_snapshot_sha256": model_files,
        "task_descriptor_sha256": {
            task["task_id"]: task["descriptor_sha256"] for task in task_records.values()
        },
    }
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"protocol_sha256": lock["protocol_sha256"], "rollouts": protocol["sampling"]["total_planned_rollouts"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.model_snapshot), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
