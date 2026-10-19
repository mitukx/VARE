#!/usr/bin/env python3
"""Run the pinned historical grader in a resource-limited, offline container."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/isolated_historical_grader_v1.json"
TASK_ROOT = ROOT / "benchmarks/historical/rvl_behavior_policy_parity"
_MAX_LOG_BYTES = 4 * 1024 * 1024


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_checked(command: list[str], *, timeout: int = 10) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {command[1:]}")
    return result.stdout.strip()


def verify_runtime(context: str, image: str) -> dict[str, str]:
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker CLI is unavailable")
    version = _run_checked([docker, "--context", context, "info", "--format", "{{.ServerVersion}}"])
    repo_digests = json.loads(_run_checked([
        docker, "--context", context, "image", "inspect", image, "--format", "{{json .RepoDigests}}"
    ]))
    if image not in repo_digests:
        raise RuntimeError("pinned container image digest is unavailable locally")
    return {"docker_server_version": version, "image_digest": image}


def _git(workspace: Path, *args: str, binary: bool = False) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(workspace), *args], check=False, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git command failed: {args[0]}")
    return result.stdout if binary else result.stdout.strip()


def _verify_inputs(workspace: Path, task_root: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    lock_path = task_root / "protocol.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if task["source"]["base_revision"] != protocol["task_revision"]:
        raise ValueError("task revision differs from sandbox protocol")
    locked = lock["locked_hashes"]
    if sha256_file(task_root / "evaluator/grade.py") != locked["evaluator_sha256"]:
        raise ValueError("grader source hash differs from locked task protocol")
    if sha256_file(task_root / "task.json") != locked["task_json_sha256"]:
        raise ValueError("task descriptor hash differs from locked task protocol")
    if sha256_file(task_root / "TASK.md") != locked["task_brief_sha256"]:
        raise ValueError("task brief hash differs from locked task protocol")

    head = _git(workspace, "rev-parse", "HEAD").decode()
    if head != protocol["task_revision"]:
        raise ValueError("candidate workspace HEAD differs from pinned task revision")
    _git(workspace, "cat-file", "-e", f"{head}^{{commit}}")
    status = _git(workspace, "status", "--porcelain", "--untracked-files=all").decode()
    allowed = set(task["source"]["files"])
    changed = _git(workspace, "diff", "--name-only", "--", *sorted(allowed)).decode().splitlines()
    all_changed = _git(workspace, "diff", "--name-only").decode().splitlines()
    staged = _git(workspace, "diff", "--cached", "--name-only").decode().splitlines()
    untracked = _git(workspace, "ls-files", "--others", "--exclude-standard").decode().splitlines()
    if set(all_changed) != set(changed) or staged or untracked:
        raise ValueError("workspace contains staged, untracked, or out-of-scope changes")
    for name in allowed:
        path = workspace / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"candidate source is not a regular file: {name}")
    source_hashes = {name: sha256_file(workspace / name) for name in sorted(allowed)}
    diff = _git(workspace, "diff", "--binary", head, "--", *sorted(allowed), binary=True)
    return {
        "task_id": task["task_id"],
        "workspace_revision": head,
        "workspace_status_before": status,
        "source_hashes_before": source_hashes,
        "candidate_diff_sha256_before": sha256_bytes(diff),
        "candidate_diff_base64": __import__("base64").b64encode(diff).decode("ascii"),
        "task_inputs_sha256": {
            "TASK.md": sha256_file(task_root / "TASK.md"),
            "task.json": sha256_file(task_root / "task.json"),
            "protocol.lock.json": sha256_file(lock_path),
            "grader.py": sha256_file(task_root / "evaluator/grade.py"),
        },
        "source_files": sorted(allowed),
    }


def _write_git_shim(harness: Path, identity: dict[str, Any]) -> None:
    binary_dir = harness / "bin"
    binary_dir.mkdir()
    (binary_dir / "git").write_text(
        "#!/usr/local/bin/python3\n"
        "import base64, json, sys\n"
        "data=json.load(open('/harness/identity.json', encoding='utf-8'))\n"
        "args=sys.argv[1:]\n"
        "if args == ['-C','/candidate','rev-parse','HEAD']:\n"
        " print(data['workspace_revision']); raise SystemExit(0)\n"
        "if args == ['-C','/candidate','cat-file','-e',data['workspace_revision']+'^{commit}']:\n"
        " raise SystemExit(0)\n"
        "prefix=['-C','/candidate','diff','--binary',data['workspace_revision'],'--']\n"
        "if args[:len(prefix)] == prefix and args[len(prefix):] == data['source_files']:\n"
        " sys.stdout.buffer.write(base64.b64decode(data['candidate_diff_base64'])); raise SystemExit(0)\n"
        " print('unsupported git invocation', file=sys.stderr); raise SystemExit(2)\n",
        encoding="utf-8",
    )
    (binary_dir / "git").chmod(0o755)
    (harness / "identity.json").write_text(
        json.dumps(identity, sort_keys=True), encoding="utf-8"
    )
    (harness / "identity.json").chmod(0o444)
    (harness / "candidate.diff").write_bytes(__import__("base64").b64decode(identity["candidate_diff_base64"]))
    (harness / "candidate.diff").chmod(0o444)


def build_command(
    *, docker: str, context: str, image: str, container_name: str,
    workspace: Path, task_root: Path, harness: Path,
) -> list[str]:
    return [
        docker, "--context", context, "run", "--rm", "--init", "--name", container_name,
        "--network", "none", "--read-only", "--cap-drop=ALL",
        "--security-opt", "no-new-privileges:true", "--pids-limit", "64",
        "--cpus", "1", "--memory", "536870912", "--memory-swap", "536870912",
        "--user", "65532:65532",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=33554432,mode=1777",
        "--mount", f"type=bind,src={workspace},dst=/candidate,readonly,bind-propagation=rprivate",
        "--mount", f"type=bind,src={task_root},dst=/task,readonly,bind-propagation=rprivate",
        "--mount", f"type=bind,src={harness},dst=/harness,readonly,bind-propagation=rprivate",
        "--env", "PATH=/harness/bin:/usr/local/bin:/usr/bin:/bin",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "PYTHONHASHSEED=0",
        image, "/usr/local/bin/python3", "/task/evaluator/grade.py", "--workspace", "/candidate",
        "--task-root", "/task", "--allow-fail",
    ]


def _parse_grade(stdout: str, task_id: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    found = None
    for offset, character in enumerate(stdout):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stdout[offset:])
        except json.JSONDecodeError:
            continue
        if (type(value) is dict and value.get("task_id") == task_id
                and type(value.get("passed")) is bool and type(value.get("failures")) is list):
            found = value
    return found


def _execute(command: list[str], *, timeout: int) -> tuple[int | None, str, str, float]:
    start = time.monotonic()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        def limit_log_file() -> None:
            resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_LOG_BYTES, _MAX_LOG_BYTES))

        try:
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, start_new_session=True,
                preexec_fn=limit_log_file,
            )
        except OSError as exc:
            return None, "", f"{type(exc).__name__}: {exc}", time.monotonic() - start
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return None, "", "container grader exceeded wall timeout", time.monotonic() - start
        stdout.seek(0)
        stderr.seek(0)
        out = stdout.read(_MAX_LOG_BYTES).decode("utf-8", errors="replace")
        err = stderr.read(_MAX_LOG_BYTES).decode("utf-8", errors="replace")
        return code, out, err, time.monotonic() - start


def run_isolated_grade(
    *, workspace: Path, task_root: Path = TASK_ROOT, output_dir: Path,
    context: str = "colima", timeout_seconds: int | None = None,
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    limits = protocol["security"]
    timeout = timeout_seconds or limits["wall_seconds"]
    if type(timeout) is not int or not 1 <= timeout <= limits["wall_seconds"]:
        raise ValueError("wall timeout must be within the protocol budget")
    workspace = workspace.resolve(strict=True)
    task_root = task_root.resolve(strict=True)
    output_dir = output_dir.resolve()
    if workspace == task_root or workspace in task_root.parents or task_root in workspace.parents:
        raise ValueError("candidate workspace and task/evaluator assets must be disjoint")
    if output_dir == workspace or output_dir in workspace.parents or workspace in output_dir.parents:
        raise ValueError("result directory and candidate workspace must be disjoint")
    if output_dir == task_root or output_dir in task_root.parents or task_root in output_dir.parents:
        raise ValueError("result directory and task/evaluator assets must be disjoint")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output_dir}")

    identity = _verify_inputs(workspace, task_root)
    runtime = verify_runtime(context, protocol["image"])
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker CLI is unavailable")
    output_dir.mkdir(parents=True, exist_ok=False)
    harness = Path(tempfile.mkdtemp(prefix="vare-grader-harness-", dir=ROOT.parent))
    harness.chmod(0o755)
    try:
        _write_git_shim(harness, identity)
        name = f"vare-grade-{uuid4().hex[:16]}"
        command = build_command(
            docker=docker, context=context, image=protocol["image"], container_name=name,
            workspace=workspace, task_root=task_root, harness=harness,
        )
        code, stdout, stderr, elapsed = _execute(command, timeout=timeout)
        if code is None and stderr == "container grader exceeded wall timeout":
            subprocess.run(
                [docker, "--context", context, "rm", "--force", name],
                check=False, capture_output=True, timeout=10,
            )
        result = _parse_grade(stdout, identity["task_id"])
        after_hashes = {name: sha256_file(workspace / name) for name in identity["source_files"]}
        stable = after_hashes == identity["source_hashes_before"]
        evidence: dict[str, Any] = {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "status": "graded" if result is not None and stable else "invalid_evidence",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task_id": identity["task_id"],
            "workspace_revision": identity["workspace_revision"],
            "candidate_diff_sha256": identity["candidate_diff_sha256_before"],
            "task_inputs_sha256": identity["task_inputs_sha256"],
            "source_hashes_before": identity["source_hashes_before"],
            "source_hashes_after": after_hashes,
            "workspace_stable_after_grade": stable,
            "runtime": {
                **runtime,
                "host_python": platform.python_version(),
                "guest_python": result.get("runtime", {}).get("python") if result else None,
                "host_platform": platform.platform(),
                "context": context,
            },
            "isolation": {
                **limits,
                "mounts": ["candidate:read-only", "task-and-evaluator:read-only", "harness:read-only"],
                "candidate_code_executed_inside_container": True,
            },
            "elapsed_seconds": round(elapsed, 3),
            "grader_exit_code": code,
            "grader_result": result,
            "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
            "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
            "peak_memory_bytes": None,
            "memory_limit_bytes": limits["memory_limit_bytes"],
            "claim_limit": protocol["claim_limit"],
        }
        (output_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
        (output_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
        if result is not None:
            (output_dir / "grader-result.json").write_text(
                json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        (output_dir / "grade.json").write_text(
            json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return evidence
    finally:
        shutil.rmtree(harness, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=TASK_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--docker-context", default="colima")
    parser.add_argument("--timeout-seconds", type=int)
    args = parser.parse_args(argv)
    try:
        report = run_isolated_grade(
            workspace=args.workspace,
            task_root=args.task_root,
            output_dir=args.output_dir,
            context=args.docker_context,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"isolated grade refused: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 3
    print(json.dumps({
        "status": report["status"],
        "passed": report["grader_result"].get("passed") if report["grader_result"] else None,
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    return 0 if report["status"] == "graded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
