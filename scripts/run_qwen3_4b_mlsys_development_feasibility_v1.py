#!/usr/bin/env python3
"""One-shot Qwen3-4B patch feasibility on the three frozen Development tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare import mlxlm_1912_repo_task, mlxlm_1971_repo_task, transformers_repo_task  # noqa: E402

MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
PROTOCOL = ROOT / "protocols/qwen3_4b_mlsys_development_feasibility_v1.lock.json"
TASKS = (
    "transformers-45852-4d-mask",
    "mlxlm-1912-top-p-min-token",
    "mlxlm-1971-mimo-value-scale",
)
K = 4
MAX_NEW_TOKENS = 512
SAMPLED_TEMPERATURE = 0.7
SAMPLED_TOP_P = 0.9
SEEDS = {
    TASKS[0]: (7401, 7402, 7403, 7404),
    TASKS[1]: (7411, 7412, 7413, 7414),
    TASKS[2]: (7421, 7422, 7423, 7424),
}
LOCKED_FILES = (
    "scripts/run_qwen3_4b_mlsys_development_feasibility_v1.py",
    "src/vare/transformers_repo_task.py",
    "src/vare/mlxlm_1912_repo_task.py",
    "src/vare/mlxlm_1971_repo_task.py",
    "src/vare/isolated_candidate.py",
    "src/vare/container_isolation.py",
    "benchmarks/mlsys_repo_eval_v2/transformers_45852/task.md",
    "benchmarks/mlsys_repo_eval_v2/transformers_45852/grader_adapter.py",
    "benchmarks/mlsys_repo_eval_v2/transformers_45852/protocol-v1.json",
    "benchmarks/mlsys_repo_eval_v2/transformers_45852/Dockerfile",
    "benchmarks/mlsys_repo_eval_v2/transformers_45852/requirements.txt",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/task.md",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/grader_adapter.py",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/protocol-v3.json",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/task.md",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/grader_adapter.py",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/protocol-v1.json",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/expected-output-v1.json",
    "tests/test_qwen3_4b_development_feasibility.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_status() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout


def verify_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected = protocol.get("implementation_sha256")
    actual = {name: sha256(ROOT / name) for name in LOCKED_FILES}
    if expected != actual:
        raise RuntimeError("development protocol source hash lock mismatch")
    if protocol.get("task_ids") != list(TASKS):
        raise RuntimeError("development task set differs from frozen protocol")
    sampling = protocol.get("sampling", {})
    if sampling != {
        "greedy_trials_per_task": 1,
        "greedy_temperature": 0.0,
        "sampled_trials_per_task": K,
        "sampled_temperature": SAMPLED_TEMPERATURE,
        "sampled_top_p": SAMPLED_TOP_P,
        "sampled_seeds": {key: list(value) for key, value in SEEDS.items()},
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_prompt_tokens": 50000,
        "sampling_order": "one greedy followed by four samples per task; tasks in listed order",
    }:
        raise RuntimeError("sampling settings differ from frozen protocol")
    if protocol.get("model", {}).get("repo_id") != MODEL_ID or protocol.get(
        "model", {}
    ).get("immutable_revision") != MODEL_REVISION:
        raise RuntimeError("model identity differs from frozen protocol")
    lock_paths = set(LOCKED_FILES) | {"protocols/qwen3_4b_mlsys_development_feasibility_v1.lock.json"}
    flags = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-v", "--", *sorted(lock_paths)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.splitlines()
    seen: set[str] = set()
    for line in flags:
        if len(line) < 3 or line[1] != " ":
            raise RuntimeError("cannot parse locked Git index flags")
        tag, path = line[0], line[2:]
        seen.add(path)
        if tag == "S" or tag.islower():
            raise RuntimeError(f"locked file has hidden Git index flag: {path}")
    if seen != lock_paths:
        raise RuntimeError("protocol input is untracked or missing")
    return protocol


def _task_spec(task_id: str):
    task: Any = None
    if task_id == TASKS[0]:
        task = transformers_repo_task
        protocol = json.loads((task.TASK_DIR / "protocol-v1.json").read_text())
        return {
            "module": task,
            "source_repo": Path(os.environ["VARE_TRANSFORMERS_SOURCE"]),
            "target": task.TARGET_PATH,
            "description": task.TASK_DESCRIPTION,
            "image": protocol["image"],
            "baseline": task.BASELINE_COMMIT,
            "source_hash": task.BASELINE_TARGET_SHA256,
            "export": "transformers",
        }
    if task_id == TASKS[1]:
        task = mlxlm_1912_repo_task
        protocol = json.loads(
            (task.TASK_DESCRIPTION.parent / "protocol-v3.json").read_text()
        )
        return {
            "module": task,
            "source_repo": Path(os.environ["VARE_MLXL_1912_SOURCE"]),
            "target": task.TARGET_PATH,
            "description": task.TASK_DESCRIPTION,
            "image": protocol["runtime"]["image"],
            "baseline": task.BASELINE_COMMIT,
            "source_hash": task.BASELINE_SOURCE_SHA256,
            "export": "mlx_lm",
        }
    task = mlxlm_1971_repo_task
    protocol = json.loads((task.TASK_ROOT / "protocol-v1.json").read_text())
    return {
        "module": task,
        "source_repo": Path(os.environ["VARE_MLXL_1971_SOURCE"]),
        "target": task.TARGET_PATH,
        "description": task.TASK_DESCRIPTION,
        "image": protocol["runtime"]["image"],
        "baseline": task.BASELINE_COMMIT,
        "source_hash": task.BASELINE_SOURCE_SHA256,
        "export": "mlx_lm",
    }


def _prepare_source(spec: dict[str, Any], workspace: Path) -> dict[str, Any]:
    module = spec["module"]
    repo = spec["source_repo"].resolve(strict=True)
    if spec["export"] == "transformers":
        module.verify_baseline(repo)
        exported = module.export_baseline(repo, workspace)
    else:
        module.verify_baseline(repo)
        exported = module.export_candidate(repo, workspace)
    target = workspace / spec["target"]
    if not target.is_file() or target.is_symlink() or sha256(target) != spec["source_hash"]:
        raise RuntimeError(f"candidate prompt source differs from pinned baseline: {spec['target']}")
    return {"export": exported, "target_path": target, "source_sha256": sha256(target)}


def _prompt(
    tokenizer: Any, task_id: str, description: str, source_path: Path, source_sha: str
) -> str:
    full_source = source_path.read_text(encoding="utf-8")
    if task_id == TASKS[0]:
        # Keep the long-context Transformers prompt inside the measured local
        # memory budget while retaining the complete frozen pad() method.
        source = "\n".join(full_source.splitlines()[2568:2739])
        source_label = (
            f"lines 2569-2739 of {source_path.name}; "
            f"full-file SHA-256 {source_sha}; excerpt SHA-256 "
            f"{hashlib.sha256(source.encode()).hexdigest()}"
        )
    else:
        source = full_source
        source_label = f"{source_path.name}; SHA-256 {source_sha}"
    user_content = (
        f"Task ID: {task_id}\n\n"
        f"Task statement:\n{description}\n\n"
        f"Pinned candidate-visible source ({source_label}):\n"
        "===== BEGIN SOURCE FILE =====\n"
        f"{source}\n"
        "===== END SOURCE FILE =====\n"
    )
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are a repository coding agent. Make the requested minimal source change. "
                    "Return only one unified Git diff with headers `diff --git a/PATH b/PATH`, "
                    "`--- a/PATH`, and `+++ b/PATH`. Do not include Markdown fences, explanations, "
                    "tests, or changes to any other file. Do not assume access to hidden tests or "
                    "the upstream fix."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _apply_and_grade(task_id: str, spec: dict[str, Any], baseline_repo: Path, raw: str, folder: Path) -> dict[str, Any]:
    if not raw.startswith("diff --git ") or "```" in raw:
        return {"passed": False, "failure": "format_failure"}
    module = spec["module"]
    patch = folder / "candidate.patch"
    patch.write_text(raw, encoding="utf-8")
    try:
        if task_id == TASKS[0]:
            module._validate_diff(raw)
            workspace = folder / "candidate"
            exported = module.apply_patch(baseline_repo, patch, workspace)
            result = module.grade_workspace(
                workspace,
                image=spec["image"],
                baseline_files=exported["baseline_files"],
            )
            return {"patch_sha256": sha256(patch), "passed": result["passed"], "grading": result}
        workspace = folder / "candidate"
        exported = module.export_candidate(baseline_repo, workspace, patch)
        result = module.grade_workspace(workspace, image=spec["image"])
        return {"patch_sha256": sha256(patch), "passed": result["passed"], "grading": result, "export": exported}
    except ValueError as exc:
        return {"patch_sha256": sha256(patch), "passed": False, "failure": "format_failure", "detail": str(exc)[:500]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"patch_sha256": sha256(patch), "passed": False, "failure": "wrong_patch", "detail": str(exc)[:500]}


def _memory_snapshot() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {"total_bytes": vm.total, "available_bytes": vm.available, "swap_used_bytes": swap.used, "swap_free_bytes": swap.free}
    except ImportError:
        if platform.system() != "Darwin":
            return {"error": "memory probe unavailable"}
        vm_stat = subprocess.run(
            ["vm_stat"], check=True, capture_output=True, text=True, timeout=10
        ).stdout
        page_match = re.search(r"page size of (\d+) bytes", vm_stat)
        if page_match is None:
            return {"error": "vm_stat page size missing"}
        page_size = int(page_match.group(1))
        pages: dict[str, int] = {}
        for line in vm_stat.splitlines():
            match = re.match(r"Pages (free|inactive|speculative):\s+([\d.]+)", line)
            if match:
                pages[match.group(1)] = int(match.group(2).replace(".", ""))
        if not all(key in pages for key in ("free", "inactive", "speculative")):
            return {"error": "vm_stat availability fields missing"}
        swap_text = subprocess.run(
            ["sysctl", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        swap_fields: dict[str, int] = {}
        for match in re.finditer(r"(total|used|free) = ([\d.]+)([KMG])", swap_text):
            power = {"K": 1, "M": 2, "G": 3}[match.group(3)]
            swap_fields[match.group(1)] = int(float(match.group(2)) * (1024**power))
        total_memory = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return {
            "total_bytes": total_memory,
            "available_bytes": (pages["free"] + pages["inactive"] + pages["speculative"])
            * page_size,
            "swap_used_bytes": swap_fields.get("used"),
            "swap_free_bytes": swap_fields.get("free"),
            "source": "vm_stat free+inactive+speculative and sysctl vm.swapusage",
        }


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = verify_protocol()
    if _git_status().strip():
        raise RuntimeError("implementation worktree must be clean before inference")
    if platform.machine() != "arm64" or platform.system() != "Darwin":
        raise RuntimeError("frozen protocol requires Apple Silicon macOS")
    docker_context = subprocess.run(["docker", "context", "show"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if docker_context != protocol["limits"]["docker_context"]:
        raise RuntimeError("Docker context differs from frozen protocol")

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    if not mx.metal.is_available() or str(mx.default_device()) != "Device(gpu, 0)":
        raise RuntimeError("MLX Metal GPU is not the default device")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw_records_path = output / "raw-records.jsonl"
    model_root = Path(os.environ["VARE_QWEN_MODEL_SNAPSHOT"]).resolve(strict=True)
    model_files = {
        p.relative_to(model_root).as_posix(): sha256(p)
        for p in sorted(model_root.rglob("*"))
        if p.is_file()
    }
    if model_files != protocol["model"]["snapshot_files_sha256"]:
        raise RuntimeError("downloaded model snapshot does not match frozen hashes")
    memory_start = _memory_snapshot()
    if memory_start.get("available_bytes", 0) < protocol["limits"]["min_available_memory_bytes"]:
        raise RuntimeError("available memory was below the frozen preflight floor")
    if memory_start.get("swap_free_bytes", 0) < protocol["limits"]["min_swap_free_bytes"]:
        raise RuntimeError("free swap was below the frozen preflight floor")
    model_started = time.monotonic()
    model, tokenizer = load(str(model_root), trust_remote_code=False)
    load_wall = time.monotonic() - model_started
    records: list[dict[str, Any]] = []
    run_started = time.monotonic()

    for task_id in TASKS:
        spec = _task_spec(task_id)
        source_root = spec["source_repo"].resolve(strict=True)
        verify = spec["module"].verify_baseline(source_root)
        del verify
        with tempfile.TemporaryDirectory(prefix=f"vare-qwen-{task_id}-") as temp_name:
            temp = Path(temp_name)
            baseline_workspace = temp / "baseline-workspace"
            prepared = _prepare_source(spec, baseline_workspace)
            task_text = spec["description"].read_text(encoding="utf-8")
            prompt = _prompt(
                tokenizer, task_id, task_text, prepared["target_path"], prepared["source_sha256"]
            )
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            encoded = tokenizer.encode(prompt)
            prompt_lock = protocol["prompts"][task_id]
            if prompt_sha != prompt_lock["sha256"] or len(encoded) != prompt_lock["token_count"]:
                raise RuntimeError(f"candidate prompt differs from frozen protocol for {task_id}")
            if len(encoded) > protocol["sampling"]["max_prompt_tokens"]:
                raise RuntimeError(f"prompt exceeded frozen context budget for {task_id}")
            for kind, seed in [("greedy", 0), *(("sampled", s) for s in SEEDS[task_id])]:
                if time.monotonic() - run_started > protocol["limits"]["max_total_wall_seconds"]:
                    raise RuntimeError("total wall time exceeded frozen stop limit")
                mx.random.seed(seed)
                sampler = make_sampler(
                    temp=0.0 if kind == "greedy" else SAMPLED_TEMPERATURE,
                    top_p=1.0 if kind == "greedy" else SAMPLED_TOP_P,
                )
                mx.reset_peak_memory()
                generation_started = time.monotonic()
                pieces: list[str] = []
                last = None
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=MAX_NEW_TOKENS,
                    sampler=sampler,
                ):
                    pieces.append(response.text)
                    last = response
                raw = "".join(pieces)
                generation_wall = time.monotonic() - generation_started
                grade: dict[str, Any]
                grade_started = time.monotonic()
                with tempfile.TemporaryDirectory(prefix="vare-qwen-grade-") as grade_name:
                    grade = _apply_and_grade(task_id, spec, source_root, raw, Path(grade_name))
                grading_wall = time.monotonic() - grade_started
                record = {
                    "task_id": task_id,
                    "kind": kind,
                    "seed": seed,
                    "prompt_sha256": prompt_sha,
                    "prompt_tokens": getattr(last, "prompt_tokens", len(encoded)),
                    "generated_tokens": getattr(last, "generation_tokens", None),
                    "prompt_tokens_per_second": getattr(last, "prompt_tps", None),
                    "decode_tokens_per_second": getattr(last, "generation_tps", None),
                    "mlx_peak_memory_gib": getattr(last, "peak_memory", None),
                    "process_peak_rss_bytes": _peak_rss_bytes(),
                    "generation_wall_seconds": generation_wall,
                    "grading_wall_seconds": grading_wall,
                    "candidate_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "candidate_output": raw,
                    **grade,
                }
                records.append(record)
                with raw_records_path.open("a", encoding="utf-8") as raw_stream:
                    raw_stream.write(json.dumps(record, sort_keys=True) + "\n")
                    raw_stream.flush()
                    os.fsync(raw_stream.fileno())
                (output / "SHA256SUMS.partial").write_text(
                    f"{sha256(raw_records_path)}  raw-records.jsonl\n", encoding="utf-8"
                )
                if _peak_rss_bytes() > protocol["limits"]["max_process_rss_bytes"]:
                    raise RuntimeError("process RSS exceeded frozen stop limit")
                if _memory_snapshot().get("available_bytes", 0) < protocol["limits"]["min_available_memory_bytes"]:
                    raise RuntimeError("available memory fell below frozen stop limit")
                if _memory_snapshot().get("swap_free_bytes", 0) < protocol["limits"]["min_swap_free_bytes"]:
                    raise RuntimeError("free swap fell below frozen stop limit")
                del pieces, response, last
                gc.collect()
                mx.clear_cache()

    result = {
        "protocol_id": protocol["protocol_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=10).stdout.strip(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "mlx_version": mx.__version__,
        "mlx_lm_version": __import__("mlx_lm").__version__,
        "metal_available": mx.metal.is_available(),
        "default_device": str(mx.default_device()),
        "model_load_wall_seconds": load_wall,
        "memory_start": memory_start,
        "memory_end": _memory_snapshot(),
        "total_wall_seconds": time.monotonic() - run_started,
        "records": records,
        "task_summary": {
            task_id: {
                "greedy_success": next((r["passed"] for r in records if r["task_id"] == task_id and r["kind"] == "greedy"), False),
                "sample_successes": [r["passed"] for r in records if r["task_id"] == task_id and r["kind"] == "sampled"],
                "pass_at_k": any(r["passed"] for r in records if r["task_id"] == task_id and r["kind"] == "sampled"),
                "reward_variance": _binary_variance([r["passed"] for r in records if r["task_id"] == task_id and r["kind"] == "sampled"]),
            }
            for task_id in TASKS
        },
        "policy_updates_run": 0,
        "limitations": [
            "Single-turn patch generation only; no iterative tool use or edit-repair loop.",
            "Three Development tasks only; no frozen Held-out task was accessed.",
            "This feasibility run does not update policy weights or establish capability improvement.",
        ],
    }
    result_path = output / "feasibility.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        f"{sha256(result_path)}  feasibility.json\n{sha256(raw_records_path)}  raw-records.jsonl\n",
        encoding="utf-8",
    )
    print(json.dumps({"protocol_id": result["protocol_id"], "total_wall_seconds": result["total_wall_seconds"], "task_summary": result["task_summary"], "memory_end": result["memory_end"]}, indent=2, sort_keys=True))
    return 0


def _binary_variance(values: list[bool]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((float(value) - mean) ** 2 for value in values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
