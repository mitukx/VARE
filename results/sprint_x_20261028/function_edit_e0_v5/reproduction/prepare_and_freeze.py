#!/usr/bin/env python3
"""Qualify and freeze one post-edit-gate synthetic task before inference."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "function_edit_e0_v5"
REPO = Path("/Users/user/Documents/Codex/2026-10-26/vare-flagship-research-sprint-real-coding/work/VARE")
BASE = Path("/tmp/vare-sprint-x-function-e0-005")
PYTHON = Path("/tmp/vare-sprint-vii-mlx/bin/python")
MODEL = Path("/Users/user/.cache/huggingface/hub/models--mlx-community--Qwen2.5-Coder-7B-Instruct-4bit/snapshots/019cc73c45c770444708a6dd8690c66243cc5c80")
MODEL_MANIFEST_SOURCE = ROOT / "function_edit_e0_v3/model-snapshot-manifest.json"
TASK_ID = "sprint-x-synthetic-function-e0-005"
PROMPT = ("Repair normalize_batch_size in src/quant.py so any requested value below one returns one, "
          "and values of one or greater are returned unchanged. Edit only this function, then finish. "
          "The independent grader will check additional inputs.")
LIMITS = {
    "max_agent_wall_seconds": 600, "max_outer_wall_seconds": 750,
    "max_process_tree_rss_bytes": 10 * 1024**3, "max_swap_growth_bytes": 32 * 1024**2,
    "minimum_available_memory_bytes": 2 * 1024**3, "minimum_disk_free_bytes": 16 * 1024**3,
    "minimum_free_swap_bytes": 512 * 1024**2, "sample_interval_seconds": 1,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main() -> int:
    if (EVIDENCE / "protocol.json").exists() or BASE.exists():
        raise SystemExit("refusing to overwrite v5 protocol or workspace")
    grader = EVIDENCE / "grader/independent_grader_v1.py"
    if not grader.is_file() or not MODEL_MANIFEST_SOURCE.is_file():
        raise SystemExit("grader or pinned model manifest is missing")
    candidate = BASE / "candidate-workspace"
    taskroot = BASE / "task-root"
    candidate.mkdir(parents=True)
    (candidate / "src").mkdir()
    (candidate / "src/quant.py").write_text(
        'def normalize_batch_size(requested: int) -> int:\n'
        '    """Return a batch size that is at least one."""\n'
        '    return requested\n', encoding="utf-8")
    taskroot.mkdir(parents=True)
    (taskroot / "TASK.md").write_text(PROMPT + "\n", encoding="utf-8")
    (taskroot / "task.json").write_text(json.dumps({
        "task_id": TASK_ID, "source": {"base_revision": "PENDING"}, "prompt": PROMPT,
        "metadata": {"editable_files": ["src/quant.py"], "split": None},
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(candidate)], check=True)
    git(candidate, "config", "user.name", "VARE Synthetic Task Freezer")
    git(candidate, "config", "user.email", "vare-task-freezer.invalid")
    git(candidate, "add", "src/quant.py")
    git(candidate, "commit", "-q", "-m", "synthetic function E0 v5 baseline")
    base_revision = git(candidate, "rev-parse", "HEAD")
    task_spec = json.loads((taskroot / "task.json").read_text(encoding="utf-8"))
    task_spec["source"]["base_revision"] = base_revision
    save(taskroot / "task.json", task_spec)

    baseline = EVIDENCE / "qualification-baseline.py"
    fix = EVIDENCE / "qualification-known-fix.py"
    mutant = EVIDENCE / "qualification-wrong-mutant.py"
    shutil.copy2(candidate / "src/quant.py", baseline)
    fix.write_text(
        'def normalize_batch_size(requested: int) -> int:\n'
        '    """Return a batch size that is at least one."""\n'
        '    return max(1, requested)\n', encoding="utf-8")
    mutant.write_text(
        'def normalize_batch_size(requested: int) -> int:\n'
        '    """Return a batch size that is at least one."""\n'
        '    return min(1, requested)\n', encoding="utf-8")
    controls = {}
    for label, path in (("baseline", baseline), ("known_fix", fix), ("wrong_mutant", mutant)):
        result = subprocess.run([str(PYTHON), "-I", str(grader), "--candidate", str(path)],
                                capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(f"grader crashed for {label}: {result.stderr}")
        parsed = json.loads(result.stdout)
        expected = {"baseline": "FAIL", "known_fix": "PASS", "wrong_mutant": "FAIL"}[label]
        if parsed.get("status") != expected:
            raise SystemExit(f"control {label} expected {expected}: {parsed}")
        controls[label] = {"status": parsed["status"], "grader_result": parsed,
                           "source_sha256": sha(path)}
    save(EVIDENCE / "qualification-controls.json", controls)
    shutil.copy2(MODEL_MANIFEST_SOURCE, EVIDENCE / "model-snapshot-manifest.json")

    controller_paths = [
        "scripts/run_local_agent.py", "src/vare/agent_edit_format.py", "src/vare/agent_edit_v2.py",
        "src/vare/agent_function_edit_v1.py", "src/vare/agent_json_eos_v1.py", "src/vare/agent_json_eos_v2.py",
        "src/vare/agent_json_eos_v3.py", "src/vare/agent_runtime_snapshot_v1.py",
        "src/vare/agent_task_contract.py", "src/vare/schema_slot_protocol.py",
        "src/vare/source_navigation_v1.py", "src/vare/source_navigation_v2.py",
    ]
    controller_hashes = {name: sha(REPO / name) for name in controller_paths}
    source_sha = sha(candidate / "src/quant.py")
    grader_sha = sha(grader)
    task_hashes = {name: sha(taskroot / name) for name in ("TASK.md", "task.json")}
    vm, sw, disk = psutil.virtual_memory(), psutil.swap_memory(), psutil.disk_usage(str(EVIDENCE))
    preflight = {
        "decision": "GO" if (vm.available >= LIMITS["minimum_available_memory_bytes"]
                              and sw.free >= LIMITS["minimum_free_swap_bytes"]
                              and disk.free >= LIMITS["minimum_disk_free_bytes"]) else "NO_GO",
        "available_memory_bytes": vm.available, "free_swap_bytes": sw.free,
        "swap_used_bytes": sw.used, "disk_free_bytes": disk.free, "ram_total_bytes": vm.total,
        "limits": LIMITS, "task_id": TASK_ID,
        "candidate_baseline_commit": base_revision,
        "grader_controls": {key: value["status"] for key, value in controls.items()},
        "runner_commit": "5a0dc8d996d444bd4aa355cede988007c1bbb284",
        "model_repository": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "model_snapshot": MODEL.name,
        "model_weights_sha256": "56a3d94706833f753e6c6b47ea57af8ef638cd8cc1d74eca1142ba640b26060e",
        "mlx_version": "0.32.4", "mlx_lm_version": "0.32.2",
    }
    save(EVIDENCE / "preflight.json", preflight)
    if preflight["decision"] != "GO":
        raise SystemExit("preflight NO_GO; inference not started")

    protocol = {
        "protocol_id": "vare-sprint-x-synthetic-function-edit-e0-v5",
        "status": "FROZEN_BEFORE_MODEL_INFERENCE",
        "hypothesis": "A post-edit tool policy that preserves completion tools while suppressing stale inventories will let the same Qwen2.5-Coder-7B agent finish after a successful function edit.",
        "cheapest_falsification": "One greedy attempt on a fresh synthetic minimum-batch task with baseline/fix/mutant-calibrated AST grader.",
        "primary_outcome": "A real model edit is applied, the agent emits finish, and the independent AST grader passes.",
        "intervention": "Use the prospective source-read-gate fix plus existing post-edit-navigation and post-edit-completion controller flags. No output repair or grader feedback during inference.",
        "task": {
            "task_id": TASK_ID, "classification": "synthetic controller feasibility only",
            "baseline_revision": base_revision, "baseline_sha256": source_sha,
            "known_fix_sha256": sha(fix), "wrong_mutant_sha256": sha(mutant),
            "grader_sha256": grader_sha, "task_input_sha256": task_hashes,
            "candidate_visible_files": ["src/quant.py", "TASK.md"],
            "prior_vare_exposure": "new task instance; model has prior exposure to different synthetic edit schemas",
            "pretraining_exposure": "UNKNOWN", "grader_hidden_during_inference": True,
        },
        "model": {
            "repository": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", "snapshot": MODEL.name,
            "weights_sha256": preflight["model_weights_sha256"],
            "manifest_sha256": sha(EVIDENCE / "model-snapshot-manifest.json"),
            "backend": "MLX Metal", "mlx_version": "0.32.4", "mlx_lm_version": "0.32.2",
            "sampling": "greedy; seed=20261036",
        },
        "controller": {
            "runner_commit": preflight["runner_commit"], "controller_source_hashes": controller_hashes,
            "generation_protocol": "schema_slots_v9", "tool_interface": "lease_bound_function_edit_v2",
            "require_source_read_before_edit": True, "source_read_gate_post_edit_terminal_tools": True,
            "post_edit_navigation_gate": True, "post_edit_completion_guidance": True,
            "no_progress_guard": False,
        },
        "budget": {"attempts": 1, "max_turns": 8, "max_new_tokens_per_turn": 512,
                   "max_total_new_tokens": 1536, "max_context_tokens": 8192,
                   "temperature": 0.0, "wall_seconds": 600, "public_tests": "not exposed"},
        "resources": LIMITS,
        "stopping_rule": "One attempt. Stop at the frozen resource or wall limits; retain incomplete attempts; do not reuse task after inference.",
        "decision_rule": {
            "PASS": "synthetic end-to-end feasibility; proceed only to fresh external Development-task qualification, not capability claims",
            "FAIL": "do not repeat this model/task family; inspect allowed actions and preserve no-go for external tasks",
        },
        "claim_boundary": "Not an external Development repair, independent-family estimate, held-out result, coding improvement, policy update, GRPO/SFT, or long-horizon RL result.",
        "qualification_controls": {key: value["status"] for key, value in controls.items()},
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    protocol_hash = hashlib.sha256(canonical).hexdigest()
    save(EVIDENCE / "protocol.json", protocol)
    (EVIDENCE / "protocol.sha256").write_text(protocol_hash + "  protocol.json (canonical JSON)\n", encoding="utf-8")
    save(EVIDENCE / "execution-freeze.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "model_inference_started": False,
        "protocol_sha256": protocol_hash, "grader_sha256": grader_sha,
        "candidate_baseline_commit": base_revision, "candidate_baseline_sha256": source_sha,
        "preflight_sha256": sha(EVIDENCE / "preflight.json"), "task_input_sha256": task_hashes,
    })
    print(json.dumps({"task_id": TASK_ID, "baseline": base_revision,
                      "protocol_sha256": protocol_hash, "grader_sha256": grader_sha,
                      "preflight": preflight}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
