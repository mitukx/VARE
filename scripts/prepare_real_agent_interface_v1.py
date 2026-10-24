#!/usr/bin/env python3
"""Materialize one pinned upstream baseline without its Git history."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.agent_task_contract import load_agent_task  # noqa: E402

TASKS = ROOT / "benchmarks/mlsys_real_agent_interface_v1/tasks"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_git(repo: Path, *args: str, text: bool = True):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True,
        capture_output=True, text=text,
    )
    return result.stdout.strip() if text else result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    task_root = TASKS / args.task_id
    lock = json.loads((task_root / "protocol.lock.json").read_text(encoding="utf-8"))
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    if lock["task_id"] != args.task_id or task["task_id"] != args.task_id:
        parser.error("task identity mismatch")
    source = args.source_repo.resolve()
    base = lock["upstream"]["base_revision"]
    resolved = run_git(source, "rev-parse", f"{base}^{{commit}}")
    if resolved != base:
        parser.error(f"upstream base mismatch: expected {base}, found {resolved}")

    workspace = args.workspace.resolve()
    if workspace.exists():
        parser.error(f"refusing to overwrite workspace: {workspace}")
    workspace.mkdir(parents=True)
    archive = run_git(source, "archive", "--format=tar", base, "mlx_lm", text=False)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        members = bundle.getmembers()
        for member in members:
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                parser.error("upstream archive contains an unsafe path")
        bundle.extractall(workspace, members=members, filter="data")

    # setuptools-scm creates this generated package version at installation time.
    # Keep it deterministic and outside the model's allowlist.
    version_file = workspace / "mlx_lm/_version.py"
    version_file.write_text('__version__ = "0.32.2"\n', encoding="utf-8")
    public_test = task_root / "test_public.py"
    (workspace / "test_public.py").write_bytes(public_test.read_bytes())

    subprocess.run(["git", "init", "-q", str(workspace)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(workspace), "add", "-A"], check=True, capture_output=True)
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "VARE frozen task materializer",
        "GIT_AUTHOR_EMAIL": "vare@local.invalid",
        "GIT_COMMITTER_NAME": "VARE frozen task materializer",
        "GIT_COMMITTER_EMAIL": "vare@local.invalid",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
    })
    subprocess.run(
        ["git", "-C", str(workspace), "commit", "-q", "-m", f"baseline {args.task_id}"],
        check=True, capture_output=True, env=env,
    )

    contract = load_agent_task(task_root)
    missing = sorted(name for name in contract.allowed_files if not (workspace / name).is_file())
    if missing:
        parser.error(f"materialized workspace is missing allowlisted paths: {missing}")
    if (workspace / "evaluator").exists() or (workspace / "protocol.lock.json").exists():
        parser.error("trusted evaluator metadata leaked into candidate workspace")
    target = workspace / lock["upstream"]["target_file"]
    target_hash = sha256(target.read_bytes())
    if target_hash != lock["upstream"]["base_target_sha256"]:
        parser.error("materialized target bytes do not match pinned upstream base")

    manifest = {
        "task_id": args.task_id,
        "upstream_base_revision": base,
        "candidate_workspace_commit": run_git(workspace, "rev-parse", "HEAD"),
        "candidate_workspace_has_upstream_git_history": False,
        "generated_version_file_sha256": sha256(version_file.read_bytes()),
        "target_file": lock["upstream"]["target_file"],
        "target_sha256": target_hash,
        "test_public_sha256": sha256((workspace / "test_public.py").read_bytes()),
        "allowed_files": list(contract.allowed_files),
    }
    manifest_path = workspace.parent / "materialization.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
