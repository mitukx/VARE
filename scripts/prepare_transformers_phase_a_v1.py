"""Create Git-history-free Development workspaces from pinned source commits."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "transformers-49116-tokenizer-routing-v2": (
        "transformers_49116_tokenizer_routing_v2",
        ("src/transformers",),
    ),
    "transformers-49412-fusion-registration-idempotence-v2": (
        "transformers_49412_fusion_idempotence_v2",
        ("src/transformers", "tests/utils/test_fusion_mapping.py"),
    ),
}


def prepare(task_id: str, upstream_repo: Path, output_root: Path) -> Path:
    task_dir_name, archive_paths = TASKS[task_id]
    task_root = ROOT / "benchmarks/mlsys_repo_eval_v4" / task_dir_name
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    revision = task["source"]["base_revision"]
    resolved = subprocess.run(
        ["git", "-C", str(upstream_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != revision:
        resolved = subprocess.run(
            ["git", "-C", str(upstream_repo), "rev-parse", f"{revision}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if resolved != revision:
        raise ValueError("upstream repository cannot resolve the pinned baseline commit")
    output = output_root / task_id / "workspace"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite workspace: {output}")
    output.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(upstream_repo), "archive", revision, *archive_paths],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        for member in stream.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError("baseline archive contains an unsafe member")
        stream.extractall(output, filter="data")
    (output / "test_public.py").write_bytes((task_root / "test_public.py").read_bytes())
    if (output / ".git").exists():
        raise AssertionError("Git history must not be exported to the candidate workspace")

    sys.path.insert(0, str(ROOT / "src"))
    from vare.agent_task_contract import load_agent_task, verify_agent_workspace

    contract = load_agent_task(task_root)
    verify_agent_workspace(output, contract)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=sorted(TASKS), required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.task_id, args.upstream_repo.resolve(), args.output_root.resolve())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
