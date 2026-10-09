#!/usr/bin/env python3
"""Read-only provenance/split audit for the ToolHazard feasibility candidate.

This does not execute benchmark-supplied Python, load a model, or score tasks.
It downloads only the pinned JSON inputs, checks their digests and ID disjointness,
and prints a deterministic JSON summary.
"""

from __future__ import annotations

import hashlib
import json
from urllib.request import urlopen


SOURCE_COMMIT = "544b73b12a25431cb0be3eb43df41b4aacce5335"
BASE = (
    "https://raw.githubusercontent.com/MurrayTom/ToolHazard/"
    f"{SOURCE_COMMIT}/toolhazard_bench/"
)
FILES = {
    "train_envs": (
        "train_set/40_rl_train_env_with_attack_points.json",
        "16671a1f593c8f7bd1ca709bfc9b27d1b194d99033805a001444827b0b1be039",
    ),
    "train_tasks": (
        "train_set/rl_tasks_IPI_all_new.json",
        "315d40f1a97d33afd5dc8c7d9c0cee8892dd045771e4d78cfe7f83d6a805fdd9",
    ),
    "train_trajectories": (
        "train_set/rl_traj.json",
        "0994cbd11dcecbbfffed2e3bf5d420833e2f5df16d62d4c3db28dc3d481ce4bc",
    ),
    "test_envs": (
        "test_set/env_with_attack_points.json",
        "3277ea321f994792e01975f996c663e796cbc2c144dc2b7e113d8ab2079d640e",
    ),
    "test_tasks": (
        "test_set/rl_task_top_5_IPI_toolselection.json",
        "c97d179e10d8ff75b1b83b30faf923b15b492e831749da224a32d1093b3792ac",
    ),
}


def load(name: str) -> tuple[bytes, object]:
    path, expected_sha256 = FILES[name]
    with urlopen(BASE + path, timeout=60) as response:
        raw = response.read()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(f"digest mismatch for {path}: {actual_sha256}")
    return raw, json.loads(raw)


def main() -> None:
    raw_and_data = {name: load(name) for name in FILES}
    train_tasks = raw_and_data["train_tasks"][1]
    test_tasks = raw_and_data["test_tasks"][1]
    train_envs = raw_and_data["train_envs"][1]
    test_envs = raw_and_data["test_envs"][1]

    if not all(isinstance(rows, list) for rows in (train_tasks, test_tasks)):
        raise SystemExit("task JSON roots must be lists")
    if not all(isinstance(rows, dict) for rows in (train_envs, test_envs)):
        raise SystemExit("environment JSON roots must be mappings")

    train_task_ids = [row["task_id"] for row in train_tasks]
    test_task_ids = [row["task_id"] for row in test_tasks]
    train_task_envs = {row["env_id"] for row in train_tasks}
    test_task_envs = {row["env_id"] for row in test_tasks}
    train_env_ids = set(train_envs)
    test_env_ids = set(test_envs)

    result = {
        "source_commit": SOURCE_COMMIT,
        "inputs": {
            name: {
                "path": FILES[name][0],
                "sha256": FILES[name][1],
                "bytes": len(raw_and_data[name][0]),
            }
            for name in FILES
        },
        "train": {
            "environment_metadata_ids": len(train_env_ids),
            "task_environment_ids": len(train_task_envs),
            "task_rows": len(train_tasks),
            "unique_task_ids": len(set(train_task_ids)),
            "released_rl_trajectories": len(raw_and_data["train_trajectories"][1]),
        },
        "test": {
            "environment_metadata_ids": len(test_env_ids),
            "task_environment_ids": len(test_task_envs),
            "task_rows": len(test_tasks),
            "unique_task_ids": len(set(test_task_ids)),
            "first_task_checker_count": len(test_tasks[0]["checklist_with_func"]),
        },
        "overlap": {
            "environment_metadata_ids": sorted(train_env_ids & test_env_ids),
            "task_environment_ids": sorted(train_task_envs & test_task_envs),
            "task_ids": sorted(set(train_task_ids) & set(test_task_ids)),
        },
        "scope_limit": (
            "Provenance and declared split disjointness only. No benchmark-supplied "
            "checker code was executed or independently validated; no model outputs, "
            "base success, parser rate, reward variance, or runtime were measured."
        ),
    }
    if len(set(train_task_ids)) != len(train_task_ids):
        raise SystemExit("duplicate training task IDs")
    if len(set(test_task_ids)) != len(test_task_ids):
        raise SystemExit("duplicate test task IDs")
    if any(result["overlap"].values()):
        raise SystemExit("train/test ID overlap")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
