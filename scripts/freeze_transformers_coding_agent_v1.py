"""Freeze a bounded local-model coding-agent feasibility study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "benchmarks/mlsys_repo_eval_v5"
STUDY = ROOT / "benchmarks/transformers_coding_agent_feasibility_v1"
TASKS = (
    "transformers-49481-bloom-static-cache-v1",
    "transformers-49361-phi3-longrope-v1",
)
MODEL_MANIFEST = ROOT / "results/transformers-fresh-dev-cohort-v1/run-11/model-preflight.json"
ENVIRONMENT_LOCK = ROOT / "results/transformers-fresh-dev-cohort-v1/run-11/mlx-requirements.freeze.txt"
IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze() -> tuple[dict, dict]:
    old = json.loads((TASK_ROOT / "protocol-v9.json").read_text(encoding="utf-8"))
    model = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if model.get("inference_performed") is not False:
        raise ValueError("model preflight must precede inference")
    task_specs = {}
    for task_id in TASKS:
        spec = old["tasks"].get(task_id)
        if not isinstance(spec, dict):
            raise ValueError(f"task is absent from the v9 Development split: {task_id}")
        task_dir = TASK_ROOT / spec["directory"]
        task_path = task_dir / "task.json"
        lock_path = task_dir / "protocol.lock.json"
        manifest_path = task_dir / "baseline-source-manifest-v1.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        prompt = task.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"task has no frozen user-facing issue prompt: {task_id}")
        task_specs[task_id] = {
            "directory": spec["directory"],
            "source_revision": spec["source_revision"],
            "fixed_revision": spec["fixed_revision"],
            "target_path": spec["target_path"],
            "task_json_sha256": sha(task_path.read_bytes()),
            "task_lock_sha256": sha(lock_path.read_bytes()),
            "baseline_manifest_sha256": sha(manifest_path.read_bytes()),
            "prompt_sha256": sha(prompt.encode("utf-8")),
            "prompt": prompt,
        }
    protocol = {
        "protocol_id": "vare-transformers-real-model-coding-agent-feasibility-v1",
        "source_head_at_freeze": "e3adb197e05d566536f826c343970d1c35dbb47e",
        "split": "development_only",
        "tasks": task_specs,
        "model": {
            "repository": model["repository"],
            "snapshot_revision": model["snapshot_revision"],
            "file_manifest_sha256": sha(MODEL_MANIFEST.read_bytes()),
            "files": model["files"],
            "backend": "mlx_metal",
            "mlx_lm": "0.32.2",
            "mlx": "0.32.4",
        },
        "conditions": {
            "per_task": [
                {"label": "greedy", "count": 1, "seed": 2101, "do_sample": False, "temperature": 0.0},
                {"label": "sampled", "count": 3, "seeds": [2102, 2103, 2104], "do_sample": True,
                 "temperature": 0.7, "top_p": 0.9, "top_k": 40},
            ],
            "order": [
                "transformers-49481-bloom-static-cache-v1/greedy",
                "transformers-49361-phi3-longrope-v1/greedy",
                "transformers-49481-bloom-static-cache-v1/sample-1",
                "transformers-49361-phi3-longrope-v1/sample-1",
                "transformers-49481-bloom-static-cache-v1/sample-2",
                "transformers-49361-phi3-longrope-v1/sample-2",
                "transformers-49481-bloom-static-cache-v1/sample-3",
                "transformers-49361-phi3-longrope-v1/sample-3",
            ],
            "agent": {
                "edit_format": "structured",
                "max_turns": 6,
                "max_no_tool_retries": 1,
                "max_new_tokens_per_turn": 768,
                "max_total_new_tokens": 3072,
                "max_context_tokens": 8192,
                "wall_seconds": 360,
                "public_test_image": IMAGE,
                "public_test_memory_mb": 1900,
                "network": "none for candidate and public-test containers",
            },
        },
        "primary_metrics": [
            "independent_verified_success_per_rollout",
            "task_level_pass_at_4",
            "greedy_verified_success",
            "per_task_binary_reward_variance",
        ],
        "secondary_metrics": [
            "tool_call_validity", "edit_applied", "public_test_pass", "total_generated_tokens",
            "wall_clock_seconds", "tokens_per_second", "peak_mlx_memory", "host_max_rss",
            "failure_category", "cost_per_verified_success",
        ],
        "grader": {
            "implementation": "scripts/grade_transformers_fresh_dev_v4.py",
            "protocol": "benchmarks/mlsys_repo_eval_v5/protocol-v9.json",
            "candidate_gets": "task prompt, source allowlist, and public test only",
            "candidate_does_not_get": "grader implementation, evaluator, hidden assertions, expected outputs, Git history, held-out tasks",
        },
        "stopping_rules": [
            "Stop an individual rollout at the first configured turn, token, wall-time, or memory failure.",
            "Stop the study immediately if baseline source hashes, frozen grading image, or model file hashes fail validation.",
            "Do not change task membership, prompt, model revision, sampling, or budgets after the first inference.",
            "This two-task Development pilot cannot establish broad ability improvement or held-out generalization.",
            "Do not start SFT or GRPO under this protocol; reward variance here is feasibility evidence only.",
        ],
        "limitations": [
            "Two tasks from one repository, with two bug families; estimates are descriptive only.",
            "The tasks are public upstream issues and may overlap model pretraining data.",
            "This is a base-policy agent feasibility comparison, not a policy-update experiment.",
        ],
    }
    protocol_path = STUDY / "protocol-v1.json"
    lock_path = STUDY / "protocol-v1.lock.json"
    if protocol_path.exists() or lock_path.exists():
        raise FileExistsError("refusing to overwrite frozen coding-agent protocol")
    protocol_bytes = (json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode()
    code_paths = [
        "scripts/run_local_agent.py",
        "scripts/grade_transformers_fresh_dev_v4.py",
        "scripts/transformers_fresh_dev_worker_v1.py",
        "src/vare/agent_task_contract.py",
        "src/vare/agent_structured_edit.py",
        "src/vare/isolated_candidate.py",
        "src/vare/container_isolation.py",
    ]
    lock = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha(protocol_bytes),
        "parent_grader_protocol_sha256": sha((TASK_ROOT / "protocol-v9.json").read_bytes()),
        "environment_lock_sha256": sha(ENVIRONMENT_LOCK.read_bytes()),
        "implementation_sha256": {name: sha((ROOT / name).read_bytes()) for name in code_paths},
        "task_json_sha256": {name: row["task_json_sha256"] for name, row in task_specs.items()},
        "baseline_manifest_sha256": {name: row["baseline_manifest_sha256"] for name, row in task_specs.items()},
    }
    STUDY.mkdir(parents=True, exist_ok=True)
    protocol_path.write_bytes(protocol_bytes)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return protocol, lock


if __name__ == "__main__":
    protocol, lock = freeze()
    print(json.dumps({"protocol_id": protocol["protocol_id"], "protocol_sha256": lock["protocol_sha256"]}))
