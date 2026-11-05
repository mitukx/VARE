import json
import resource
import sys
from pathlib import Path

import pytest

from vare.environments.spec import ResourceLimits
from vare.research_lab.controller import create_run
from vare.research_lab.edits import apply_edits
from vare.research_lab.evaluation import grade
from vare.research_lab.execute import (
    HARD_ADDRESS_SPACE_LIMIT_APPLIED,
    IS_TRUSTED_OS_SANDBOX,
    NETWORK_NAMESPACE_APPLIED,
    parse_worker_stdout,
    run_bounded,
    run_worker,
)
from vare.research_lab.paths import PathViolation
from vare.research_lab.spec import load_catalog, suite_manifest
from vare.research_lab.workspace import IntegrityError, materialize, stage_subject


def _limits(**kwargs):
    values = {"wall_time_s": 5, "cpu_time_s": 5, "memory_mb": None, "max_output_bytes": 100_000}
    values.update(kwargs)
    return ResourceLimits(**values)


def test_edit_rejects_traversal_and_non_editable(tmp_path: Path):
    task = load_catalog()["grad_accum_scale_v1"]
    workspace = tmp_path / "workspace"
    materialize(task, workspace)
    with pytest.raises(PathViolation):
        apply_edits(workspace, task.editable_files, [{
            "path": "../secret.py",
            "old": "a",
            "new": "b",
        }])
    with pytest.raises(PathViolation):
        apply_edits(workspace, task.editable_files, [{
            "path": "/etc/passwd",
            "old": "a",
            "new": "b",
        }])
    with pytest.raises(PathViolation):
        apply_edits(workspace, task.editable_files, [{
            "path": "accum.py/../../evaluation.py",
            "old": "a",
            "new": "b",
        }])


def test_symlink_subject_is_rejected(tmp_path: Path):
    task = load_catalog()["grad_accum_scale_v1"]
    workspace = tmp_path / "workspace"
    materialize(task, workspace)
    target = workspace / "accum.py"
    target.unlink()
    target.symlink_to(tmp_path / "outside.py")
    with pytest.raises((PathViolation, IntegrityError)):
        stage_subject(workspace, task.editable_files, tmp_path / "subject")


def test_forged_workspace_report_does_not_pass(tmp_path: Path):
    task = load_catalog()["grad_accum_scale_v1"]
    workspace = tmp_path / "workspace"
    materialize(task, workspace)
    (workspace / "result.json").write_text('{"passed": true, "max_abs_error": 0}\n', encoding="utf-8")
    result = grade(task, workspace)
    assert result["passed"] is False
    assert result["classification"] == "fail_correctness"


def test_timeout_is_not_a_correctness_failure(tmp_path: Path):
    subject = tmp_path / "subject"
    subject.mkdir()
    (subject / "spin.py").write_text("def spin():\n    while True:\n        pass\n", encoding="utf-8")
    execution = run_worker(
        subject,
        [{"id": "spin", "op": "call", "module": "spin", "function": "spin", "args": []}],
        _limits(wall_time_s=0.4, cpu_time_s=30),
    )
    assert execution.classification == "timeout"
    assert execution.timed_out is True


def test_malformed_worker_stdout_fails_closed(tmp_path: Path):
    with pytest.raises(ValueError):
        parse_worker_stdout("not-json\n")
    execution = run_bounded(
        [sys.executable, "-c", "print('forged-pass')"],
        cwd=tmp_path,
        stdin=b"",
        limits=_limits(),
    )
    assert execution.classification == "ok"
    with pytest.raises(ValueError):
        parse_worker_stdout(execution.stdout)


def test_clock_monkeypatch_does_not_zero_the_trusted_timer(tmp_path: Path):
    subject = tmp_path / "subject"
    subject.mkdir()
    (subject / "project.py").write_text(
        "def score_matrix(rows, cols, passes):\n"
        "    import time\n"
        "    time.perf_counter = lambda: 0.0\n"
        "    total = 0.0\n"
        "    for _ in range(30000):\n"
        "        total += 1.0\n"
        "    return [[total]]\n",
        encoding="utf-8",
    )
    execution = run_worker(
        subject,
        [{
            "id": "bench",
            "op": "bench",
            "module": "project",
            "function": "score_matrix",
            "args": [[[1.0]], [[1.0]], 1],
            "warmup": 0,
            "repeats": 2,
        }],
        _limits(),
    )
    assert execution.payload is not None
    samples = execution.payload["results"][0]["elapsed_s"]
    assert min(samples) > 0.0


def test_candidate_cannot_import_grader_package(tmp_path: Path):
    task = load_catalog()["grad_accum_scale_v1"]
    workspace = tmp_path / "workspace"
    materialize(task, workspace)
    (workspace / "accum.py").write_text(
        "def accumulated_gradient(batches, params):\n"
        "    import vare\n"
        "    return params\n",
        encoding="utf-8",
    )
    result = grade(task, workspace)
    assert result["passed"] is False
    assert result["correctness_passed"] is False


def test_shell_action_is_rejected_without_success(tmp_path: Path):
    task = load_catalog()["grad_accum_scale_v1"]
    run_dir = tmp_path / "run"
    controller = create_run(
        run_dir,
        task,
        run_id="sec",
        proposer_name="external",
        proposer_kind="external_agent",
        suite_sha256=suite_manifest()["manifest_sha256"],
    )
    record = controller.apply({"type": "shell", "argv": ["curl", "https://example.invalid"]})
    assert record["status"] == "stopped"
    assert record["failure_reason"] == "disallowed_action"
    assert record.get("verification") is None
    text = (run_dir / "journal.jsonl").read_text(encoding="utf-8")
    assert "https://example.invalid" not in text
    blocked = controller.apply({"type": "network", "hypothesis_id": "dial"})
    assert blocked["status"] == "stopped"
    assert blocked["failure_reason"] == "disallowed_action"


def test_runner_does_not_apply_an_address_space_cap(tmp_path: Path):
    assert IS_TRUSTED_OS_SANDBOX is False
    assert HARD_ADDRESS_SPACE_LIMIT_APPLIED is False
    parent = resource.getrlimit(resource.RLIMIT_AS)
    execution = run_bounded(
        [sys.executable, "-c", "import json,resource; print(json.dumps(resource.getrlimit(resource.RLIMIT_AS)))"],
        cwd=tmp_path,
        stdin=b"",
        limits=_limits(memory_mb=1024),
    )
    assert execution.memory_limit_applied is False
    assert execution.classification == "ok"
    assert tuple(json.loads(execution.stdout)) == parent


def test_worker_can_open_a_socket_and_a_wrong_answer_still_fails(tmp_path: Path):
    """Residual: no network namespace. Socket creation is not a passing grade."""
    assert NETWORK_NAMESPACE_APPLIED is False
    subject = tmp_path / "subject"
    subject.mkdir()
    (subject / "probe.py").write_text(
        "import socket\n"
        "def probe():\n"
        "    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "    sock.close()\n"
        "    return 1\n",
        encoding="utf-8",
    )
    execution = run_worker(
        subject,
        [{"id": "net", "op": "call", "module": "probe", "function": "probe", "args": []}],
        _limits(),
    )
    assert execution.classification == "ok"
    assert execution.payload is not None
    assert execution.payload["results"][0]["value"] == 1
    task = load_catalog()["grad_accum_scale_v1"]
    workspace = tmp_path / "workspace"
    materialize(task, workspace)
    (workspace / "accum.py").write_text(
        "import socket\n"
        "def accumulated_gradient(batches, params):\n"
        "    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "    sock.close()\n"
        "    return [0.0 for _ in params]\n",
        encoding="utf-8",
    )
    result = grade(task, workspace)
    assert result["passed"] is False
    assert result["classification"] == "fail_correctness"
