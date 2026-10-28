#!/usr/bin/env python3
"""Fresh, no-weights preflight for the frozen Transformers #49752 trial."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def host_memory() -> tuple[int, str]:
    raw = subprocess.run(["/usr/bin/vm_stat"], check=True, capture_output=True, text=True).stdout
    import re
    page = int(re.search(r"page size of (\d+) bytes", raw).group(1))
    def pages(label: str) -> int:
        match = re.search(rf"^{re.escape(label)}:\s*(\d+)\.", raw, re.MULTILINE)
        return int(match.group(1)) if match else 0
    available = sum(pages(name) for name in ("Pages free", "Pages inactive", "Pages speculative")) * page
    return available, raw


def swap() -> tuple[int, int, str]:
    raw = subprocess.run(["/usr/sbin/sysctl", "vm.swapusage"], check=True, capture_output=True, text=True).stdout.strip()
    import re
    def size(label: str) -> int:
        match = re.search(rf"{label}\s*=\s*([0-9.]+)([KMG])", raw)
        units = {"K": 1024, "M": 1024**2, "G": 1024**3}
        return int(float(match.group(1)) * units[match.group(2)])
    used = size("used")
    return used, max(size("total") - used, 0), raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, workspace, out = args.repo.resolve(), args.workspace.resolve(), args.out.resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path[:0] = [str(repo / "scripts"), str(repo / "src")]
    from mlx_lm.utils import load_tokenizer
    from run_local_agent import tools_for
    from vare.agent_task_contract import load_agent_task

    protocol_path = repo / "protocols/sprint_xiv_transformers_49752_minicpm_development_v3.json"
    protocol = json.loads(protocol_path.read_text())
    model = protocol["model_trial"]
    gates = protocol["resource_gates"]
    task_root = repo / "benchmarks/external_development/transformers_49752_minicpm_stacked_visual_units"
    source_rel = "src/transformers/models/minicpmv4_6/video_processing_minicpmv4_6.py"
    source = workspace / source_rel
    baseline_preflight = None
    snapshot = Path(model["snapshot_path"]).resolve(strict=True)
    expected_manifest = model["snapshot_manifest"]
    observed_manifest = {}
    for name, previous in sorted(expected_manifest.items()):
        file_path = snapshot / name
        observed_manifest[name] = {
            "path_is_symlink": file_path.is_symlink(),
            "sha256": sha256_file(file_path),
            "size_bytes": file_path.stat().st_size,
        }
    manifest_match = observed_manifest == expected_manifest

    used_swap, free_swap, swap_text = swap()
    available, vm_text = host_memory()
    disk_free = shutil.disk_usage(repo).free
    contract = load_agent_task(task_root)
    allowed = list(contract.allowed_files)
    editable = [name for name in allowed if Path(name).name != "test_public.py"]
    tools = tools_for("unified_diff", public_tests=True)
    task_text = contract.prompt_text
    system = (
        "You are an offline code repair agent. Solve only the task below. The workspace contains only the task source files. "
        "Read the source path named in the user instruction, then use tools to inspect and edit. Use apply_patch for small source diffs; never rewrite a whole file. "
        "The patch must use exact headers --- a/<allowed source path> and +++ b/<allowed source path>; do not wrap it in a code fence. "
        + "Allowed source paths: " + ", ".join(sorted(editable)) + ". First call list_files, then read the exact source path named in the task. "
        + "Do not invent paths. The workspace is already prepared; never ask for a workspace path, more information, or permission. "
        + "The external grader is hidden. Do not describe a fix without applying it. Call finish only after changing a source file. "
        + "After changing source, call run_public_tests; read its output and repair failures. The test file is immutable. "
        + "Call finish only after the public smoke test passes. You have a limited tool budget; each assistant turn before finish must call a tool. "
        + "TASK:\n" + task_text.split("## Reproduce", 1)[0].rstrip()
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": model["instruction"]},
    ]
    tokenizer = load_tokenizer(str(snapshot))
    initial_tokens = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True)
    projected = [dict(item) for item in messages]
    projected.append({"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "list_files", "arguments": {}}}]})
    projected.append({"role": "tool", "name": "list_files", "content": json.dumps(sorted(allowed), ensure_ascii=False)})
    projected.append({"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"path": source_rel}}}]})
    projected.append({"role": "tool", "name": "read_file", "content": json.dumps({"resolved_path": source_rel, "content": source.read_text()}, ensure_ascii=False)})
    projected_tokens = tokenizer.apply_chat_template(projected, tools=tools, tokenize=True, add_generation_prompt=True)

    def git(*args_: str) -> str:
        return subprocess.run(["git", "-C", str(workspace), *args_], check=True, capture_output=True, text=True).stdout.strip()
    remotes = subprocess.run(["git", "-C", str(workspace), "remote", "-v"], capture_output=True, text=True).stdout.strip()
    task_hashes = {name: sha256_file(task_root / name) for name in ("TASK.md", "task.json", "test_public.py", "evaluator/grade.py")}
    expected_source = protocol["task"]["baseline_source_sha256"]
    runner = repo / model["runner_path"]
    grader = task_root / "evaluator/grade.py"
    checks = {
        "host_arch": platform.machine() in {"arm64", "aarch64"},
        "available_memory": available >= gates["minimum_available_memory_bytes"],
        "free_swap": free_swap >= gates["minimum_free_swap_bytes"],
        "disk": disk_free >= gates["minimum_disk_free_bytes"],
        "model_revision": snapshot.name == model["revision"],
        "model_snapshot_manifest": manifest_match,
        "runner_hash": sha256_file(runner) == model["runner_sha256"],
        "grader_hash": sha256_file(grader) == protocol["task"]["grader_sha256"],
        "task_hashes": task_hashes["TASK.md"] == protocol["task"]["task_md_sha256"] and task_hashes["task.json"] == protocol["task"]["task_json_sha256"] and task_hashes["test_public.py"] == protocol["task"]["public_test_sha256"] and task_hashes["evaluator/grade.py"] == protocol["task"]["grader_sha256"],
        "workspace_commit": git("rev-parse", "HEAD") == protocol["task"]["candidate_workspace_commit"],
        "workspace_tree": git("rev-parse", "HEAD^{tree}") == protocol["task"]["candidate_workspace_tree"],
        "workspace_no_remote": not remotes,
        "workspace_clean": not git("status", "--porcelain=v1", "--untracked-files=all"),
        "candidate_baseline_hash": sha256_file(source) == expected_source,
        "hidden_grader_absent": not (workspace / "evaluator").exists(),
        "context_budget": len(projected_tokens) + model["max_new_tokens_per_turn"] <= model["max_context_tokens"],
    }
    record = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "grader_sha256": sha256_file(grader),
        "status": "PASS_PRELOAD_GATE" if all(checks.values()) else "NO_GO_PREFLIGHT",
        "classification": "Fresh host, exact model snapshot, task workspace, runner, grader, public test, and tokenizer-context preflight; no model weights loaded or generation.",
        "checks": checks,
        "task_id": protocol["task"]["task_id"],
        "task_split": "development",
        "task_file_hashes": task_hashes,
        "candidate_workspace_commit": git("rev-parse", "HEAD"),
        "candidate_workspace_tree": git("rev-parse", "HEAD^{tree}"),
        "candidate_source_sha256": sha256_file(source),
        "workspace_has_upstream_remotes": bool(remotes),
        "model_repository": model["checkpoint"],
        "model_revision": model["revision"],
        "model_snapshot_path": str(snapshot),
        "model_snapshot_manifest_sha256_pinned": model["snapshot_manifest_sha256"],
        "model_snapshot_manifest_match": manifest_match,
        "model_snapshot_file_count": len(observed_manifest),
        "model_snapshot_total_size_bytes": sum(item["size_bytes"] for item in observed_manifest.values()),
        "runner_sha256": sha256_file(runner),
        "grader_sha256": sha256_file(grader),
        "tokenizer_class": type(tokenizer).__name__,
        "prompt_tokens_initial": len(initial_tokens),
        "prompt_tokens_after_list_and_source_read": len(projected_tokens),
        "projected_source_characters": len(source.read_text()),
        "max_context_tokens": model["max_context_tokens"],
        "host_resources": {
            "available_free_inactive_speculative_bytes": available,
            "minimum_available_memory_bytes": gates["minimum_available_memory_bytes"],
            "swap_used_bytes": used_swap,
            "free_swap_bytes": free_swap,
            "minimum_free_swap_bytes": gates["minimum_free_swap_bytes"],
            "swap_usage": swap_text,
            "disk_free_bytes": disk_free,
            "minimum_disk_free_bytes": gates["minimum_disk_free_bytes"],
            "vm_stat": vm_text,
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "mlx_lm": version("mlx-lm"),
            "mlx": version("mlx"),
        },
        "model_weights_loaded": False,
        "generated_tokens": 0,
        "optimizer_updates": 0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": record["status"], "checks": checks, "prompt_tokens_after_read": len(projected_tokens), "free_swap_bytes": free_swap, "available_memory_bytes": available, "disk_free_bytes": disk_free}, indent=2))
    return 0 if record["status"] == "PASS_PRELOAD_GATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
