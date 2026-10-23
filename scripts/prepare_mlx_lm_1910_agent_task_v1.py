#!/usr/bin/env python3
"""Recreate the deterministic local Git base required by the agent runner."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

EXPECTED = "d151a804ca04296e758e77503bc552515f723399"
FILES = ("qwen3_coder.py", "test_public.py")


def run(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True,
                            capture_output=True, text=True, env=env)
    return result.stdout.strip()


def prepare(task_root: Path) -> str:
    workspace = task_root / "workspace"
    for name in FILES:
        path = workspace / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"expected regular task input: {name}")
    if (workspace / ".git").exists():
        head = run(workspace, "rev-parse", "HEAD")
        if head != EXPECTED:
            raise ValueError(f"existing task workspace has unexpected HEAD: {head}")
        return head
    run(workspace, "init", "-q")
    run(workspace, "config", "user.name", "VARE Experiment")
    run(workspace, "config", "user.email", "vare-experiment@example.invalid")
    run(workspace, "add", *FILES)
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = "@1792778316 +0900"
    env["GIT_COMMITTER_DATE"] = "@1792778316 +0900"
    run(workspace, "commit", "-qm", "Freeze MLX-LM #1910 baseline source", env=env)
    head = run(workspace, "rev-parse", "HEAD")
    if head != EXPECTED:
        raise ValueError(f"recreated task base SHA differs: {head}")
    return head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.task_root.resolve(strict=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
