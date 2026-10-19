from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_isolated_grader", ROOT / "scripts/run_isolated_grader.py"
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


TASK_ID = "rvl-hf-behavior-policy-parity-v1"


def _identity(diff: bytes = b"") -> dict:
    return {
        "task_id": TASK_ID,
        "workspace_revision": "a" * 40,
        "candidate_diff_sha256_before": hashlib.sha256(diff).hexdigest(),
        "candidate_diff_base64": base64.b64encode(diff).decode("ascii"),
        "source_files": ["src/allowed.py"],
        "source_hashes_before": {"src/allowed.py": hashlib.sha256(b"x=1\n").hexdigest()},
        "task_inputs_sha256": {},
        "grader_sha256": "b" * 64,
        "task_descriptor_sha256": "c" * 64,
    }


def test_container_command_is_offline_read_only_and_resource_limited(tmp_path: Path) -> None:
    command = module.build_command(
        docker="/usr/bin/docker",
        context="colima",
        image="python@sha256:" + "a" * 64,
        container_name="vare-grade-test",
        workspace=tmp_path / "candidate",
        task_root=tmp_path / "task",
        harness=tmp_path / "harness",
    )

    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert command[command.index("--user") + 1] == "65532:65532"
    assert command[command.index("--pids-limit") + 1] == "64"
    assert command[command.index("--memory") + 1] == "536870912"
    assert all("readonly" in command[index + 1] for index, value in enumerate(command) if value == "--mount")
    assert not any("socket" in value.lower() or "credential" in value.lower() for value in command)


def test_grader_refuses_to_reuse_an_existing_evidence_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "candidate"
    task_root = tmp_path / "task"
    output_dir = tmp_path / "evidence"
    workspace.mkdir()
    task_root.mkdir()
    output_dir.mkdir()

    try:
        module.run_isolated_grade(
            workspace=workspace, task_root=task_root, output_dir=output_dir,
        )
    except FileExistsError as exc:
        assert "refusing to overwrite evidence directory" in str(exc)
    else:
        raise AssertionError("existing evidence directory was accepted")


def test_generated_git_shim_supports_only_pinned_queries(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    harness.mkdir()
    identity = _identity(b"diff --git a/src/allowed.py b/src/allowed.py\n")
    module._write_git_shim(harness, identity)
    shim = harness / "bin/git"

    unsupported = subprocess.run(
        [sys.executable, str(shim), "status"], check=False, capture_output=True, text=True,
    )
    assert unsupported.returncode == 2
    assert "unsupported git invocation" in unsupported.stderr

    revision = subprocess.run(
        [sys.executable, str(shim), "-C", "/candidate", "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    assert revision.returncode == 0
    assert revision.stdout.strip() == identity["workspace_revision"]

    commit_check = subprocess.run(
        [sys.executable, str(shim), "-C", "/candidate", "cat-file", "-e",
         identity["workspace_revision"] + "^{commit}"],
        check=False, capture_output=True, text=True,
    )
    assert commit_check.returncode == 0


def test_minimal_snapshot_omits_ignored_workspace_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    task_root = tmp_path / "task"
    workspace.mkdir()
    task_root.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src/allowed.py").write_text("x=1\n", encoding="utf-8")
    (workspace / "credential.txt").write_text("must not be mounted", encoding="utf-8")
    (task_root / "evaluator").mkdir()
    task_files = {
        "TASK.md": "brief",
        "task.json": "{}",
        "protocol.lock.json": "{}",
        "evaluator/grade.py": "print('grade')\n",
    }
    for name, value in task_files.items():
        path = task_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    identity = _identity()
    identity["source_hashes_before"] = {"src/allowed.py": digest("x=1\n")}
    identity["task_inputs_sha256"] = {
        name: digest(value) for name, value in task_files.items()
    }

    candidate_snapshot, task_snapshot, hashes = module._make_snapshots(
        workspace, task_root, tmp_path / "snapshots", identity
    )

    assert hashes == identity["source_hashes_before"]
    assert sorted(path.relative_to(candidate_snapshot).as_posix()
                  for path in candidate_snapshot.rglob("*") if path.is_file()) == ["src/allowed.py"]
    assert not (candidate_snapshot / "credential.txt").exists()
    assert sorted(path.relative_to(task_snapshot).as_posix()
                  for path in task_snapshot.rglob("*") if path.is_file()) == sorted(task_files)


def test_task_lock_is_pinned_outside_caller_selected_directory(tmp_path: Path) -> None:
    task_root = tmp_path / "task-copy"
    shutil.copytree(ROOT / "benchmarks/historical/rvl_behavior_policy_parity", task_root)
    with (task_root / "evaluator/grade.py").open("a", encoding="utf-8") as stream:
        stream.write("# tampered\n")

    try:
        module._verify_inputs(tmp_path / "not-a-workspace", task_root)
    except ValueError as exc:
        assert "trusted sandbox manifest" in str(exc)
    else:
        raise AssertionError("modified evaluator passed task manifest verification")


def test_grader_parser_rejects_fragments_duplicates_and_trailing_output() -> None:
    record = {"task_id": TASK_ID, "passed": False, "failures": ["known regression"]}
    encoded = json.dumps(record)
    assert module._parse_grade("\n" + encoded + "\n", TASK_ID) == record
    assert module._parse_grade("noise\n" + encoded, TASK_ID) is None
    assert module._parse_grade(encoded + "\ntrailing", TASK_ID) is None
    assert module._parse_grade(encoded + "\n" + encoded, TASK_ID) is None
    assert module._parse_grade(
        '{"task_id":"other","passed":true,"failures":[]}', TASK_ID
    ) is None
    assert module._parse_grade(
        '{"task_id":"rvl-hf-behavior-policy-parity-v1","passed":1,"failures":[]}', TASK_ID
    ) is None


def test_grader_result_requires_zero_exit_and_matching_provenance() -> None:
    identity = _identity()
    identity.update({
        "source_files_sha256": identity["source_hashes_before"],
    })
    result = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "workspace_revision": identity["workspace_revision"],
        "candidate_diff_sha256": identity["candidate_diff_sha256_before"],
        "source_file_sha256": identity["source_hashes_before"],
        "evaluator_sha256": identity["grader_sha256"],
        "task_descriptor_sha256": identity["task_descriptor_sha256"],
        "passed": False,
        "failures": ["expected negative control"],
    }

    assert module._validate_grade_result(
        result, exit_code=0, identity=identity, source_hashes=identity["source_hashes_before"]
    )
    assert not module._validate_grade_result(
        result, exit_code=7, identity=identity, source_hashes=identity["source_hashes_before"]
    )
    forged = dict(result, evaluator_sha256="0" * 64)
    assert not module._validate_grade_result(
        forged, exit_code=0, identity=identity, source_hashes=identity["source_hashes_before"]
    )
