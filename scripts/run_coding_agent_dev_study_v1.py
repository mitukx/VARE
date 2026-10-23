#!/usr/bin/env python3
"""Run the frozen, Development-only Qwen3-4B Phase A v6 study once."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/coding_agent_phase_a_v6"
PROTOCOL_PATH = STUDY / "protocol-v1.json"
LOCK_PATH = STUDY / "protocol-v1.lock.json"
TASK_PACK = ROOT / "benchmarks/mlsys_repo_eval_v6"
IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)
MODEL_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
MODEL_REPOSITORY = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
INSTRUCTION = (
    "Please implement the requested repository change using the available tools. "
    "Run the public tests, inspect the final diff, and finish when the repair is ready."
)
TASK_IDS = (
    "transformers-49315-watermark-repeat-v1",
    "transformers-49515-rope-setter-v1",
    "transformers-49731-generation-device-type-v1",
    "transformers-49767-janus-background-v1",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def create_docker_shared_workspace(task_id: str, *, home: Path | None = None) -> Path:
    """Create candidate files below Colima's shared home mount, not macOS /var/folders."""
    if not task_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in task_id):
        raise ValueError("unsafe task id for workspace directory")
    home_root = Path.home() if home is None else home.resolve(strict=True)
    root = home_root / ".cache" / "vare" / "candidate-workspaces"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("Docker-shared candidate workspace root must be a real directory")
    root.chmod(0o700)
    parent = Path(tempfile.mkdtemp(prefix=f"{task_id}-", dir=root))
    parent.chmod(0o700)
    workspace = parent / "workspace"
    # materialize_workspace owns creation of the destination and rejects any
    # pre-existing path. Keep this leaf absent until pinned files are written.
    if workspace.exists() or workspace.is_symlink():
        raise ValueError("candidate workspace destination must not already exist")
    return workspace


def classify_public_test_result(result: Any) -> str:
    """Classify the public runner result without turning launch errors into reward."""
    if result is None:
        # The agent may exhaust its action budget before invoking the public
        # evaluator. Preserve that as an ordinary unsuccessful rollout.
        return "not_run"
    if not isinstance(result, dict):
        return "evaluator_unavailable"
    explicit_contract = {
        "passed": (True, True),
        "assertion_failure": (False, True),
        "test_execution_error": (False, True),
        "test_failure": (False, True),
        "candidate_timeout": (False, True),
        "candidate_output_limit": (False, True),
        "container_resource_failure": (False, True),
        "container_launch_failure": (None, False),
        "runner_integrity_error": (None, False),
        "evaluator_unavailable": (None, False),
    }
    if "classification" in result:
        classification = result.get("classification")
        if not isinstance(classification, str) or classification not in explicit_contract:
            return "runner_integrity_error"
        expected_passed, expected_gradeable = explicit_contract[classification]
        if (
            result.get("passed") is not expected_passed
            or result.get("gradeable") is not expected_gradeable
        ):
            return "runner_integrity_error"
        # A claimed pass must also agree with the process outcome.
        if classification == "passed" and (
            ("returncode" in result and result["returncode"] != 0)
            or result.get("error")
        ):
            return "runner_integrity_error"
        return classification
    # Historical records predate the explicit class field. Exit 125 is Docker's
    # launch-failure code; it means no test assertion ran.
    code = result.get("returncode")
    # Legacy records have no explicit class. Reject contradictory pass
    # claims before classifying nonzero exits as ordinary test failures.
    if result.get("passed") is True and (
        code != 0 or result.get("error") or result.get("gradeable") is False
    ):
        return "runner_integrity_error"
    if code in (125, 126, 127):
        return "container_launch_failure"
    error = result.get("error")
    if isinstance(error, str) and "container exceeded wall-time limit" in error:
        return "candidate_timeout"
    if isinstance(error, str) and "container exceeded output byte limit" in error:
        return "candidate_output_limit"
    if isinstance(error, str) and "public test file is missing or changed" in error:
        return "runner_integrity_error"
    if code == 0 and result.get("passed") is True:
        return "passed"
    if error:
        return "evaluator_unavailable"
    if isinstance(code, int):
        return "container_resource_failure" if code in (137, 139) else "test_failure"
    return "evaluator_unavailable"


def is_unavailable_public_evaluator(result: Any) -> bool:
    return classify_public_test_result(result) in {
        "container_launch_failure", "runner_integrity_error", "evaluator_unavailable",
    }


def run_checked(
    command: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()


def load_frozen() -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    lock_bytes = LOCK_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    lock = json.loads(lock_bytes)
    if sha256_bytes(protocol_bytes) != lock.get("protocol_sha256"):
        raise ValueError("frozen protocol hash differs from its lock")
    if protocol.get("protocol_id") != lock.get("protocol_id"):
        raise ValueError("frozen protocol identity differs from its lock")
    if protocol.get("model", {}).get("revision") != MODEL_REVISION:
        raise ValueError("frozen model revision differs")
    if protocol.get("software", {}).get("grading_image") != IMAGE:
        raise ValueError("frozen grading image differs")
    if protocol.get("task_split", {}).get("development") != list(TASK_IDS):
        raise ValueError("frozen Development task membership or order differs")
    for relative, expected in lock["implementation_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"locked implementation changed: {relative}")
    for task_id in TASK_IDS:
        descriptor = STUDY / "tasks" / task_id / "task.json"
        if sha256_file(descriptor) != lock["task_descriptor_sha256"][task_id]:
            raise ValueError(f"frozen task descriptor changed: {task_id}")
    return protocol, lock, protocol_bytes, lock_bytes


def verify_sha256_manifest(directory: Path, filename: str) -> None:
    for line in (directory / filename).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = directory / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"evidence manifest mismatch: {path}")


def verify_calibrations() -> dict[str, Any]:
    hidden_root = ROOT / "results/phase-a-v6/protocol-v2"
    verify_sha256_manifest(hidden_root, "SHA256SUMS")
    expected = {
        "watermark": ("transformers-49315-watermark-repeat-v1",),
        "rope": ("transformers-49515-rope-setter-v1",),
        "move": ("transformers-49731-generation-device-type-v1",),
        "janus": ("transformers-49767-janus-background-v1",),
    }
    hidden = []
    for prefix, (task_id,) in expected.items():
        baseline = json.loads(
            (hidden_root / "calibration" / f"{prefix}-baseline.json").read_text()
        )
        fixed = json.loads(
            (hidden_root / "calibration" / f"{prefix}-known-fix.json").read_text()
        )
        if (
            baseline.get("task_id") != task_id
            or baseline.get("passed") is not False
            or fixed.get("task_id") != task_id
            or fixed.get("passed") is not True
            or baseline.get("grader_files_in_candidate_subject") is not False
            or fixed.get("grader_files_in_candidate_subject") is not False
            or baseline.get("grading_image") != IMAGE
            or fixed.get("grading_image") != IMAGE
        ):
            raise ValueError(
                f"pinned hidden grader calibration is invalid for {task_id}"
            )
        hidden.extend([baseline, fixed])
    public_root = hidden_root / "public-calibration-v3"
    verify_sha256_manifest(public_root, "SHA256SUMS")
    report = json.loads((public_root / "REPORT.json").read_text())
    calibration_script = ROOT / "scripts/calibrate_public_tests_v6_v2.py"
    if (
        report.get("calibration_passed") is not True
        or len(report.get("results", [])) != 8
        or any(
            record.get("calibration_script_sha256") != sha256_file(calibration_script)
            for record in report["results"]
        )
    ):
        raise ValueError("provenance-complete public calibration v3 is invalid")
    return {
        "hidden_calibration_sha256_manifest_verified": True,
        "hidden_baseline_fail_known_fix_pass": True,
        "hidden_calibration_records": len(hidden),
        "public_calibration_sha256_manifest_verified": True,
        "public_calibration_passed": True,
        "public_calibration_records": len(report["results"]),
        "public_calibration_report_sha256": sha256_file(public_root / "REPORT.json"),
    }


def materialize_workspace(
    source_repo: Path,
    task_id: str,
    destination: Path,
    manifest: dict[str, Any],
) -> None:
    """Create only manifest-listed regular files from the pinned Git commit."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to reuse candidate workspace: {destination}")
    if destination.parent.exists() and destination.parent.is_symlink():
        raise ValueError("workspace parent must not be a symlink")
    # The container runs as uid 65534; make the materialized tree traversable
    # while keeping candidate writes constrained by the read-only mount.
    destination.mkdir(parents=True, mode=0o755)
    os.chmod(destination, 0o755)
    expected_files: dict[str, dict[str, Any]] = manifest["files"]
    expected_dirs: dict[str, int] = manifest["directories"]
    revision = manifest["source_revision"]
    archive_paths = [
        name for name in sorted(expected_files) if name != "test_public.py"
    ]
    for name in [*archive_paths, *expected_dirs]:
        parsed = PurePosixPath(name)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in name:
            raise ValueError(f"unsafe path in pinned manifest: {name}")
    with tempfile.NamedTemporaryFile(
        prefix="vare-v6-source-", suffix=".tar", delete=False
    ) as handle:
        archive_path = Path(handle.name)
    try:
        with archive_path.open("wb") as output:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_repo),
                    "archive",
                    "--format=tar",
                    revision,
                    "--",
                    *archive_paths,
                ],
                stdout=output,
                check=True,
                timeout=180,
            )
        found: set[str] = set()
        for directory, mode in sorted(
            expected_dirs.items(), key=lambda item: item[0].count("/")
        ):
            path = destination / directory
            path.mkdir(parents=True, exist_ok=True)
            os.chmod(path, mode)
        with tarfile.open(archive_path, "r:") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                if not name or name == "." or name in expected_dirs:
                    continue
                if name not in expected_files or name == "test_public.py":
                    raise ValueError(
                        f"source archive contains a path outside the manifest: {name}"
                    )
                if not member.isfile():
                    raise ValueError(f"manifest source is not a regular file: {name}")
                body = archive.extractfile(member)
                if body is None:
                    raise ValueError(f"source archive omitted file bytes: {name}")
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body.read())
                os.chmod(target, expected_files[name]["mode"])
                if sha256_file(target) != expected_files[name]["sha256"]:
                    raise ValueError(f"pinned source blob hash differs: {name}")
                found.add(name)
        if found != set(archive_paths):
            raise ValueError(f"source archive inventory differs for {task_id}")
        task_test = TASK_PACK / task_id / "test_public.py"
        public = destination / "test_public.py"
        public.write_bytes(task_test.read_bytes())
        os.chmod(public, expected_files["test_public.py"]["mode"])
        if sha256_file(public) != expected_files["test_public.py"]["sha256"]:
            raise ValueError("pinned public test differs from the baseline manifest")
        for directory, mode in sorted(
            expected_dirs.items(), key=lambda item: item[0].count("/"), reverse=True
        ):
            os.chmod(destination / directory, mode)
    finally:
        archive_path.unlink(missing_ok=True)


def workspace_snapshot_hash(workspace: Path) -> str:
    entries: list[bytes] = []
    for path in sorted(workspace.rglob("*")):
        if path.is_symlink():
            raise ValueError("candidate workspace contains a symlink")
        if path.is_file():
            relative = path.relative_to(workspace).as_posix()
            entries.extend(
                (relative.encode(), b"\0", bytes.fromhex(sha256_file(path)), b"\n")
            )
    return sha256_bytes(b"".join(entries))


def candidate_diff(original: bytes, candidate: bytes, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.decode("utf-8").splitlines(keepends=True),
            candidate.decode("utf-8").splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def command_env(protocol: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "4",
            "PYTHONPATH": str(ROOT / "src"),
            "DOCKER_CONTEXT": protocol["software"]["docker_context"],
        }
    )
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/phase-a-v6/protocol-v1/run-1"
    )
    args = parser.parse_args()
    protocol, lock, protocol_bytes, lock_bytes = load_frozen()
    source_repo = args.source_repo.resolve(strict=True)
    model = args.model_snapshot.resolve(strict=True)
    output = args.output_dir
    if output.exists() or output.is_symlink():
        parser.error("refusing to overwrite a prior experiment run")
    if model.name != MODEL_REVISION:
        parser.error("model snapshot does not match the frozen revision")
    current_head = run_checked(["git", "rev-parse", "HEAD"], cwd=ROOT)
    if current_head != protocol["implementation_commit"]:
        parser.error("working HEAD differs from the frozen implementation commit")
    status = run_checked(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT
    )
    status_paths = {line[3:] for line in status.splitlines()}
    allowed_frozen_artifacts = {
        "experiments/coding_agent_phase_a_v6/protocol-v1.json",
        "experiments/coding_agent_phase_a_v6/protocol-v1.lock.json",
        str(Path(__file__).relative_to(ROOT)),
    }
    non_output_paths = {
        path
        for path in status_paths
        if not path.startswith("results/phase-a-v6/protocol-v1/run-1/")
    }
    if (
        any(not line.startswith("?? ") for line in status.splitlines())
        or non_output_paths != allowed_frozen_artifacts
    ):
        parser.error(
            "only the frozen protocol, its lock, and this hash-recorded runner may be untracked"
        )
    failed_setup = (
        ROOT / "results/phase-a-v6/protocol-v1/run-1/run-manifest.partial.json"
    )
    if failed_setup.is_file():
        previous = json.loads(failed_setup.read_text(encoding="utf-8"))
        if previous.get("rollouts_completed") != 0 or previous.get("rollouts"):
            parser.error(
                "run-1 contains model rollout evidence and cannot be resumed as a fresh run"
            )
        if any((ROOT / "results/phase-a-v6/protocol-v1/run-1").rglob("agent.json")):
            parser.error(
                "run-1 contains an agent record and cannot be treated as pre-inference setup"
            )
    if (
        run_checked(["docker", "context", "show"], cwd=ROOT, env=command_env(protocol))
        != protocol["software"]["docker_context"]
    ):
        parser.error("Docker context differs from the frozen protocol")
    image_id = run_checked(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        cwd=ROOT,
        env=command_env(protocol),
    )
    if image_id != "sha256:" + IMAGE.rsplit("sha256:", 1)[1]:
        parser.error("grading image identity differs from the frozen digest")
    for task_id in TASK_IDS:
        source_revision = protocol["tasks"][task_id]["base_revision"]
        run_checked(
            ["git", "cat-file", "-e", f"{source_revision}^{{commit}}"], cwd=source_repo
        )
    model_files = protocol["model"]["snapshot_files_sha256"]
    for name, expected in model_files.items():
        path = model / name
        if not path.is_file() or sha256_file(path) != expected:
            parser.error(f"frozen model snapshot file differs: {name}")

    sys.path.insert(0, str(ROOT / "src"))
    from vare.agent_task_contract import load_agent_task, verify_agent_workspace
    from scripts.audit_coding_agent_dev_resource_gate_v1 import audit as resource_audit

    preflight = resource_audit(model)
    if preflight["decision"] != "GO":
        parser.error("current exact resource gate returned NO-GO")
    limits = protocol["limits"]
    if preflight["hardware"]["swap"]["used_bytes"] > limits["maximum_swap_used_bytes"]:
        parser.error("current swap use exceeds the frozen maximum")
    calibration_audit = verify_calibrations()
    output.mkdir(parents=True)
    (output / "protocol-v1.json").write_bytes(protocol_bytes)
    (output / "protocol-v1.lock.json").write_bytes(lock_bytes)
    write_json(output / "preflight.json", preflight)
    write_json(output / "calibration-audit.json", calibration_audit)
    start_epoch = time.time()
    deadline_epoch = start_epoch + limits["maximum_total_wall_seconds"]
    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": lock["protocol_sha256"],
        "implementation_commit": current_head,
        "pre_run_git_status": status.splitlines(),
        "runner_path": str(Path(__file__).relative_to(ROOT)),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_snapshot_files_sha256": model_files,
        "grading_image": IMAGE,
        "container_image_id": image_id,
        "source_repository": run_checked(
            ["git", "remote", "get-url", "origin"], cwd=source_repo
        ),
        "source_commits": {
            task: protocol["tasks"][task]["base_revision"] for task in TASK_IDS
        },
        "source_repo_head": run_checked(["git", "rev-parse", "HEAD"], cwd=source_repo),
        "source_repo_dirty": bool(
            run_checked(["git", "status", "--porcelain"], cwd=source_repo)
        ),
        "python": sys.version,
        "mlx_version": protocol["software"]["mlx"],
        "mlx_lm_version": protocol["software"]["mlx_lm"],
        "docker_context": protocol["software"]["docker_context"],
        "task_ids": list(TASK_IDS),
        "instruction": INSTRUCTION,
        "instruction_sha256": sha256_bytes(INSTRUCTION.encode()),
        "started_at_epoch": start_epoch,
        "shared_deadline_epoch": deadline_epoch,
        "maximum_total_wall_seconds": limits["maximum_total_wall_seconds"],
        "rollouts_planned": protocol["sampling"]["total_planned_rollouts"],
        "rollouts_completed": 0,
        "rollouts": [],
        "stop_reason": None,
    }
    execution_config = {
        "schema_version": 1,
        "supplements_protocol": lock["protocol_id"],
        "protocol_sha256": lock["protocol_sha256"],
        "runner_sha256": run_manifest["runner_sha256"],
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "backend": "mlx",
        "grading_image": IMAGE,
        "container_image_id": image_id,
        "instruction": INSTRUCTION,
        "instruction_sha256": sha256_bytes(INSTRUCTION.encode()),
        "task_ids_in_order": list(TASK_IDS),
        "rollout_order": [
            {"condition": "greedy", "seeds": [0], "do_sample": False},
            {
                "condition": "sampled",
                "seeds": [101, 202],
                "do_sample": True,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 20,
            },
        ],
        "agent_limits": protocol["agent"],
        "resource_limits": protocol["limits"],
        "environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "4",
            "DOCKER_CONTEXT": protocol["software"]["docker_context"],
        },
        "network_policy": "candidate tests run in frozen Docker image with network disabled",
        "grader_policy": "independent v7 host grader invoked after the agent process exits",
    }
    execution_config_path = output / "execution-supplement-v1.json"
    write_json(execution_config_path, execution_config)
    run_manifest["execution_supplement_sha256"] = sha256_file(execution_config_path)
    write_json(output / "run-manifest.partial.json", run_manifest)

    for task_id in TASK_IDS:
        task_dir = STUDY / "tasks" / task_id
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (TASK_PACK / task_id / "baseline-source-manifest-v1.json").read_text(
                encoding="utf-8"
            )
        )
        task_output = output / task_id
        task_output.mkdir()
        workspace = create_docker_shared_workspace(task_id)
        materialize_workspace(source_repo, task_id, workspace, manifest)
        contract = load_agent_task(task_dir)
        baseline_hashes = verify_agent_workspace(workspace, contract)
        if workspace_snapshot_hash(workspace) != task["source"]["snapshot_sha256"]:
            raise RuntimeError(f"materialized workspace hash differs: {task_id}")
        write_json(
            task_output / "workspace-baseline.json",
            {
                "task_id": task_id,
                "base_revision": task["source"]["base_revision"],
                "workspace_sha256": workspace_snapshot_hash(workspace),
                "file_count": len(manifest["files"]),
                "directory_count": len(manifest["directories"]),
                "editable_source_hashes": baseline_hashes,
                "public_test_sha256": task["source"]["readonly_file_sha256"][
                    "test_public.py"
                ],
            },
        )
        baseline_out = task_output / "baseline-grade.json"
        baseline_command = [
            sys.executable,
            "-m",
            "scripts.grade_transformers_fresh_dev_v7",
            "--task-id",
            task_id,
            "--workspace",
            str(workspace),
            "--image",
            IMAGE,
            "--output",
            str(baseline_out),
        ]
        baseline_proc = subprocess.run(
            baseline_command,
            cwd=ROOT,
            env=command_env(protocol),
            capture_output=True,
            text=True,
            timeout=180,
        )
        (task_output / "baseline-grade.stdout.txt").write_text(
            baseline_proc.stdout, encoding="utf-8"
        )
        (task_output / "baseline-grade.stderr.txt").write_text(
            baseline_proc.stderr, encoding="utf-8"
        )
        if (
            baseline_proc.returncode != 1
            or not baseline_out.exists()
            or json.loads(baseline_out.read_text())["passed"] is not False
        ):
            run_manifest["stop_reason"] = f"baseline_calibration_not_failed:{task_id}"
            write_json(output / "run-manifest.partial.json", run_manifest)
            return 2

        target = task["metadata"]["editable_files"][0]
        baseline_target = (workspace / target).read_bytes()
        for sampled in (False, True):
            sampling = protocol["sampling"]["sampled" if sampled else "greedy"]
            for seed in sampling["seeds"]:
                if time.time() >= deadline_epoch:
                    run_manifest["stop_reason"] = "shared_study_deadline_reached"
                    write_json(output / "run-manifest.partial.json", run_manifest)
                    return 2
                (workspace / target).write_bytes(baseline_target)
                os.chmod(workspace / target, manifest["files"][target]["mode"])
                verify_agent_workspace(workspace, contract)
                rollout_id = f"{'sampled' if sampled else 'greedy'}-{seed}"
                rollout_dir = task_output / rollout_id
                rollout_dir.mkdir()
                agent_path = rollout_dir / "agent.json"
                agent_cmd = [
                    sys.executable,
                    str(ROOT / "scripts/run_local_agent.py"),
                    "--workspace",
                    str(workspace),
                    "--model-path",
                    str(model),
                    "--model-repository",
                    MODEL_REPOSITORY,
                    "--backend",
                    "mlx",
                    "--public-test-image",
                    IMAGE,
                    "--public-test-memory-mb",
                    str(protocol["agent"]["public_test_memory_mb"]),
                    "--task-root",
                    str(task_dir),
                    "--output",
                    str(agent_path),
                    "--instruction",
                    INSTRUCTION,
                    "--seed",
                    str(seed),
                    "--edit-format",
                    protocol["agent"]["edit_format"],
                    "--max-turns",
                    str(protocol["agent"]["max_turns"]),
                    "--max-no-tool-retries",
                    str(protocol["agent"]["max_no_tool_retries"]),
                    "--max-new-tokens-per-turn",
                    str(protocol["agent"]["max_new_tokens_per_turn"]),
                    "--max-total-new-tokens",
                    str(protocol["agent"]["max_total_new_tokens"]),
                    "--max-context-tokens",
                    str(protocol["agent"]["max_context_tokens"]),
                    "--wall-seconds",
                    str(protocol["agent"]["wall_seconds_per_rollout"]),
                    "--study-deadline-epoch",
                    str(deadline_epoch),
                    "--minimum-available-memory-bytes",
                    str(limits["minimum_available_memory_bytes"]),
                    "--minimum-free-swap-bytes",
                    str(limits["minimum_free_swap_bytes"]),
                    "--maximum-swap-used-bytes",
                    str(limits["maximum_swap_used_bytes"]),
                    "--maximum-mlx-peak-memory-bytes",
                    str(limits["maximum_mlx_peak_bytes_per_turn"]),
                    "--temperature",
                    str(sampling.get("temperature", 0.3)),
                    "--top-p",
                    str(sampling.get("top_p", 0.9)),
                    "--top-k",
                    str(sampling.get("top_k", 20)),
                    "--do-sample" if sampling["do_sample"] else "--no-do-sample",
                    "--require-split",
                    "development",
                ]
                rollout_record: dict[str, Any] = {
                    "task_id": task_id,
                    "rollout_id": rollout_id,
                    "seed": seed,
                    "sampling": sampling,
                    "agent_command": agent_cmd,
                    "workspace_before_sha256": workspace_snapshot_hash(workspace),
                    "status": "started",
                }
                write_json(rollout_dir / "launch.json", rollout_record)
                print(
                    json.dumps(
                        {
                            "event": "rollout_start",
                            "task_id": task_id,
                            "rollout_id": rollout_id,
                            "completed": run_manifest["rollouts_completed"],
                            "planned": protocol["sampling"]["total_planned_rollouts"],
                        }
                    ),
                    flush=True,
                )
                try:
                    proc = subprocess.run(
                        agent_cmd,
                        cwd=ROOT,
                        env=command_env(protocol),
                        capture_output=True,
                        text=True,
                        timeout=min(
                            protocol["agent"]["wall_seconds_per_rollout"] + 90,
                            max(1, int(deadline_epoch - time.time())),
                        ),
                    )
                    (rollout_dir / "agent.stdout.txt").write_text(
                        proc.stdout, encoding="utf-8"
                    )
                    (rollout_dir / "agent.stderr.txt").write_text(
                        proc.stderr, encoding="utf-8"
                    )
                    rollout_record["agent_exit_code"] = proc.returncode
                except subprocess.TimeoutExpired as exc:
                    (rollout_dir / "agent.stdout.txt").write_bytes(exc.stdout or b"")
                    (rollout_dir / "agent.stderr.txt").write_bytes(exc.stderr or b"")
                    rollout_record["status"] = "runner_timeout_no_agent_record"
                    run_manifest["rollouts"].append(rollout_record)
                    run_manifest["stop_reason"] = "runner_timeout"
                    write_json(output / "run-manifest.partial.json", run_manifest)
                    return 2
                rollout_record["agent_record_sha256"] = (
                    sha256_file(agent_path) if agent_path.is_file() else None
                )
                if not agent_path.is_file():
                    rollout_record["status"] = "agent_record_missing"
                    run_manifest["rollouts"].append(rollout_record)
                    run_manifest["stop_reason"] = "agent_record_missing"
                    write_json(output / "run-manifest.partial.json", run_manifest)
                    return 2
                agent = json.loads(agent_path.read_text(encoding="utf-8"))
                candidate_target = (workspace / target).read_bytes()
                (rollout_dir / "candidate-target.py").write_bytes(candidate_target)
                (rollout_dir / "candidate.patch").write_text(
                    candidate_diff(baseline_target, candidate_target, target),
                    encoding="utf-8",
                )
                rollout_record["workspace_after_sha256"] = workspace_snapshot_hash(
                    workspace
                )
                public_test_status = classify_public_test_result(
                    agent.get("public_test_result")
                )
                rollout_record["public_test_classification"] = public_test_status
                if is_unavailable_public_evaluator(agent.get("public_test_result")):
                    rollout_record.update(
                        {
                            "status": "public_test_infrastructure_error",
                            "agent_stop_reason": agent.get("stop_reason"),
                            "verified_reward": None,
                            "grading_skipped_reason": public_test_status,
                        }
                    )
                    run_manifest["rollouts"].append(rollout_record)
                    run_manifest["stop_reason"] = "public_test_infrastructure_error"
                    write_json(output / "run-manifest.partial.json", run_manifest)
                    return 2
                grade_path = rollout_dir / "independent-grade.json"
                grade_cmd = [
                    sys.executable,
                    "-m",
                    "scripts.grade_transformers_fresh_dev_v7",
                    "--task-id",
                    task_id,
                    "--workspace",
                    str(workspace),
                    "--image",
                    IMAGE,
                    "--output",
                    str(grade_path),
                ]
                grade_proc = subprocess.run(
                    grade_cmd,
                    cwd=ROOT,
                    env=command_env(protocol),
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                (rollout_dir / "grader.stdout.txt").write_text(
                    grade_proc.stdout, encoding="utf-8"
                )
                (rollout_dir / "grader.stderr.txt").write_text(
                    grade_proc.stderr, encoding="utf-8"
                )
                if grade_proc.returncode not in (0, 1) or not grade_path.is_file():
                    rollout_record["status"] = "grader_integrity_error"
                    run_manifest["rollouts"].append(rollout_record)
                    run_manifest["stop_reason"] = "grader_integrity_error"
                    write_json(output / "run-manifest.partial.json", run_manifest)
                    return 2
                grading = json.loads(grade_path.read_text(encoding="utf-8"))
                # Never award reward from unvalidated candidate-supplied flags.
                public_passed = public_test_status == "passed"
                changed = agent.get("source_hashes_before") != agent.get(
                    "source_hashes_after"
                )
                reward = int(
                    bool(changed and public_passed and grading.get("passed") is True)
                )
                rollout_record.update(
                    {
                        "status": "completed",
                        "agent_stop_reason": agent.get("stop_reason"),
                        "agent_new_tokens": agent.get("new_tokens"),
                        "agent_elapsed_seconds": agent.get("elapsed_seconds"),
                        "agent_runtime": agent.get("runtime"),
                        "agent_candidate_diff_sha256": agent.get(
                            "candidate_diff_sha256"
                        ),
                        "public_test_passed": public_passed,
                        "public_test_classification": public_test_status,
                        "independent_grader_passed": grading.get("passed") is True,
                        "candidate_target_sha256": grading.get(
                            "candidate_target_sha256"
                        ),
                        "verified_reward": reward,
                        "tool_turns": agent.get("tool_turns"),
                        "edit_attempted": agent.get("edit_attempted"),
                        "edit_applied": agent.get("edit_applied"),
                        "format_parse_failure_count": agent.get(
                            "format_parse_failure_count"
                        ),
                        "no_tool_retries": agent.get("no_tool_retries"),
                        "valid_tool_calls": sum(
                            1
                            for turn in agent.get("transcript", [])
                            for call in turn.get("tool_calls", [])
                            if isinstance(call, dict)
                            and isinstance(call.get("name"), str)
                            and call.get("name") != "recovered_edit"
                        ),
                        "invalid_tool_calls": sum(
                            1
                            for turn in agent.get("transcript", [])
                            for call in turn.get("tool_calls", [])
                            if isinstance(call, dict)
                            and isinstance(call.get("error"), str)
                        ),
                        "public_test_attempts": sum(
                            len(turn.get("automatic_public_tests", []))
                            for turn in agent.get("transcript", [])
                            if isinstance(turn, dict)
                        )
                        + int(isinstance(agent.get("public_test_result"), dict)),
                        "public_test_passes": sum(
                            int(item.get("passed") is True)
                            for turn in agent.get("transcript", [])
                            if isinstance(turn, dict)
                            for item in turn.get("automatic_public_tests", [])
                            if isinstance(item, dict)
                        )
                        + int(
                            isinstance(agent.get("public_test_result"), dict)
                            and agent["public_test_result"].get("passed") is True
                        ),
                        "prompt_tokens_per_second": [
                            turn.get("prompt_tokens_per_second")
                            for turn in agent.get("transcript", [])
                            if isinstance(
                                turn.get("prompt_tokens_per_second"), (int, float)
                            )
                        ],
                        "decode_tokens_per_second": [
                            turn.get("decode_tokens_per_second")
                            for turn in agent.get("transcript", [])
                            if isinstance(
                                turn.get("decode_tokens_per_second"), (int, float)
                            )
                        ],
                        "mlx_peak_memory_bytes_per_turn": [
                            turn.get("mlx_peak_memory_bytes")
                            for turn in agent.get("transcript", [])
                            if isinstance(turn.get("mlx_peak_memory_bytes"), int)
                        ],
                        "max_rss_bytes": agent.get("max_rss_bytes"),
                    }
                )
                run_manifest["rollouts"].append(rollout_record)
                run_manifest["rollouts_completed"] += 1
                print(
                    json.dumps(
                        {
                            "event": "rollout_complete",
                            "task_id": task_id,
                            "rollout_id": rollout_id,
                            "reward": reward,
                            "elapsed_seconds": rollout_record["agent_elapsed_seconds"],
                            "stop_reason": rollout_record["agent_stop_reason"],
                        }
                    ),
                    flush=True,
                )
                safety_stops = {
                    "minimum_available_memory_breached",
                    "minimum_free_swap_breached",
                    "maximum_swap_used_breached",
                    "maximum_mlx_peak_memory_breached",
                    "resource_probe_unavailable",
                    "study_wall_budget_exhausted",
                }
                if agent.get("stop_reason") in safety_stops:
                    run_manifest["stop_reason"] = agent["stop_reason"]
                    write_json(output / "run-manifest.partial.json", run_manifest)
                    return 2
                write_json(output / "run-manifest.partial.json", run_manifest)

    rewards = [record.get("verified_reward", 0) for record in run_manifest["rollouts"]]
    per_task: dict[str, Any] = {}
    for task_id in TASK_IDS:
        rows = [row for row in run_manifest["rollouts"] if row["task_id"] == task_id]
        sample_rewards = [
            row["verified_reward"]
            for row in rows
            if row["rollout_id"].startswith("sampled-")
        ]
        per_task[task_id] = {
            "rollouts": len(rows),
            "verified_successes": sum(row["verified_reward"] for row in rows),
            "pass_at_3": int(any(row["verified_reward"] for row in rows)),
            "sample_rewards": sample_rewards,
            "sample_reward_variance_population": (
                sum(
                    (value - sum(sample_rewards) / len(sample_rewards)) ** 2
                    for value in sample_rewards
                )
                / len(sample_rewards)
                if sample_rewards
                else None
            ),
        }
    run_manifest["overall"] = {
        "verified_successes": sum(rewards),
        "binary_reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "binary_reward_variance_population": (
            sum((value - sum(rewards) / len(rewards)) ** 2 for value in rewards)
            / len(rewards)
            if rewards
            else None
        ),
        "distinct_tasks_with_success": sum(
            int(item["pass_at_3"] > 0) for item in per_task.values()
        ),
        "per_task": per_task,
    }
    nonconstant_sample_group = any(
        len(set(item["sample_rewards"])) > 1 for item in per_task.values()
    )
    go = (
        run_manifest["overall"]["distinct_tasks_with_success"] >= 2
        and nonconstant_sample_group
    )
    run_manifest["decision"] = "GO" if go else "NO-GO"
    run_manifest["nonconstant_sample_group"] = nonconstant_sample_group
    run_manifest["stop_reason"] = "completed_frozen_rollouts"
    run_manifest.pop("partial", None)
    write_json(output / "run-manifest.json", run_manifest)
    (output / "run-manifest.partial.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
