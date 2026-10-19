"""Contract tests for candidate/grader separation. Docker is optional."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vare.agent_task_contract import assert_train_only, load_agent_task
from vare.container_isolation import (
    DockerIsolationRunner,
    IsolationError,
    IsolationLimits,
    IsolationResult,
)
from vare.environments.spec import (
    CommandSpec,
    EngineeringTaskSpec,
    RepositorySource,
    ResourceLimits,
)
from vare.evidence import sha256_json
from vare.sandboxed_repository import DockerRepositoryEvaluator
from vare.separated_grading import (
    CANARY_PROBE_SOURCE,
    SeparatedCandidateGrader,
    assert_candidate_namespace_closed,
    canary_visible_under_mounts,
    repository_grader_argv,
    source_only_snapshot,
    stage_hidden_grader,
)
from vare.trusted_scoring import JSONIOCase, _encode

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "local/test@sha256:" + "a" * 64
DOCKER_IMAGE = "python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1"
PROTOCOL = ROOT / "protocols/separated_coding_eval_cpu_v1.lock.json"


def _limits(image: str = IMAGE) -> IsolationLimits:
    return IsolationLimits(image, timeout_s=15, memory_mb=256, cpus=0.5, pids_limit=32)


def _cases() -> tuple[JSONIOCase, ...]:
    raw = json.loads(PROTOCOL.read_text())["hidden_cases"]
    return tuple(JSONIOCase(item["case_id"], item["request"], item["expected"]) for item in raw)


def _load_probe_module():
    path = ROOT / "scripts/verify_docker_isolation_boundary_v1.py"
    spec = importlib.util.spec_from_file_location("verify_docker_isolation_boundary_v1", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_audit_module():
    path = ROOT / "scripts/audit_docker_isolation_boundary_v1.py"
    spec = importlib.util.spec_from_file_location("audit_docker_isolation_boundary_v1", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalRunner:
    """Workspace-local subprocess stand-in. Mount proofs use the real argv builder."""

    def __init__(self) -> None:
        self.inner = DockerIsolationRunner()
        self.calls: list[tuple] = []

    def candidate_argv(self, *args, **kwargs):
        return self.inner.candidate_argv(*args, **kwargs)

    async def run_candidate(self, limits, *, allowed_workspace_root, workspace, command):
        self.calls.append(("candidate", Path(workspace), tuple(command)))
        completed = subprocess.run(  # noqa: ASYNC221
            command, cwd=workspace, input=b"", capture_output=True,
            env={"PATH": os.environ["PATH"]}, check=False,
        )
        return IsolationResult(completed.returncode, completed.stdout.decode(), completed.stderr.decode())

    async def run_candidate_io(
        self, limits, *, allowed_workspace_root, workspace, command, stdin_data,
    ):
        self.calls.append(("io", Path(workspace), bytes(stdin_data), tuple(command)))
        completed = subprocess.run(  # noqa: ASYNC221
            command, cwd=workspace, input=stdin_data, capture_output=True,
            env={"PATH": os.environ["PATH"]}, check=False,
        )
        return IsolationResult(
            completed.returncode, completed.stdout.decode(), completed.stderr.decode(),
        )


def _grader(tmp_path: Path, runner) -> tuple[SeparatedCandidateGrader, Path, Path]:
    root = tmp_path / "workspaces"
    workspace = root / "trial"
    workspace.mkdir(parents=True)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    return SeparatedCandidateGrader(
        runner=runner, limits=_limits(), allowed_workspace_root=root, snapshot_root=snapshots,
    ), root, workspace


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


BROKEN = (ROOT / "benchmarks/separated_json_square_v1/fixture/solve.py").read_text(encoding="utf-8")
FIXED = (
    "import json\nimport sys\n\n"
    "def main():\n"
    "    request = json.loads(sys.stdin.read() or '{}')\n"
    "    number = request['n']\n"
    "    sys.stdout.write(json.dumps({'value': number * number}))\n\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)


def test_historical_probe_argv_reads_canary_zero_of_three(tmp_path):
    module = _load_probe_module()
    root = tmp_path / "candidate-root"
    workspace = root / "workspace"
    grader_root = tmp_path / "grader-root"
    workspace.mkdir(parents=True)
    grader_root.mkdir()
    runner = LocalRunner()

    async def fake_execute(argv, limits):
        runner.calls.append(argv)
        return IsolationResult(0, "CANARY_NOT_VISIBLE\n", "")

    runner.execute = fake_execute  # type: ignore[method-assign]
    visible = []
    for _ in range(3):
        canary = "vare-synthetic-grader-canary-" + os.urandom(8).hex()
        canary_file = grader_root / "synthetic-hidden-canary.txt"
        canary_file.write_text(canary, encoding="utf-8")
        observed = asyncio.run(module._grader_probe(
            runner, root, workspace, grader_root, canary, image=IMAGE,
        ))
        argv = runner.calls[-1]
        visible.append(observed["canary_visible"])
        assert canary not in "\n".join(argv)
        assert canary not in CANARY_PROBE_SOURCE
        assert [module_target(flag) for flag in observed["mount_flags"]] == ["/workspace"]
        assert observed["network_mode_none"] is True
        assert canary_visible_under_mounts(argv, canary_file=canary_file, canary=canary) is False
    assert visible == [False, False, False]
    assert sum(visible) == 0


def module_target(flag: str) -> str:
    return flag.split(",target=", 1)[1].split(",", 1)[0]


def test_legacy_public_grader_mount_still_exposes_the_canary(tmp_path):
    root = tmp_path / "candidate-root"
    workspace = root / "workspace"
    grader_root = tmp_path / "grader-root"
    workspace.mkdir(parents=True)
    grader_root.mkdir()
    canary = "vare-synthetic-grader-canary-" + os.urandom(8).hex()
    canary_file = grader_root / "synthetic-hidden-canary.txt"
    canary_file.write_text(canary, encoding="utf-8")
    spec = EngineeringTaskSpec(
        id="synthetic-grader-visibility-v1",
        title="Synthetic grader namespace visibility probe",
        prompt="Read the synthetic canary from the grader namespace.",
        family="isolation-probe",
        source=RepositorySource(local_path="."),
        tests=(CommandSpec(("true",), name="unused-control"),),
        limits=ResourceLimits(wall_time_s=3, cpu_time_s=2, memory_mb=64),
        metadata={"purpose": "synthetic isolation probe only"},
    )
    spec_path = grader_root / "task.json"
    spec.write(spec_path)
    backend = DockerRepositoryEvaluator(
        runner=DockerIsolationRunner(), limits=_limits(), allowed_workspace_root=root,
    )
    argv = backend._argv(
        spec_path=spec_path, workspace=workspace, spec_digest=sha256_json(spec.to_dict()),
        policy_id="synthetic-probe", policy_version=1, baseline_metrics=None,
        name="vare-repo-" + "a" * 24,
    )
    assert canary_visible_under_mounts(argv, canary_file=canary_file, canary=canary) is True


def test_auditor_keeps_historical_failure_and_accepts_closed_mounts():
    audit = _load_audit_module()
    legacy = [{
        "canary_visible": True,
        "returncode": 0,
        "network_mode_none": True,
        "mount_flags": [
            "type=bind,source=/tmp/workspace,target=/candidate,readonly",
            "type=bind,source=/tmp/grader-root,target=/grader,readonly",
        ],
    } for _ in range(3)]
    assert audit.classify_grader_confidentiality(legacy) == "FAIL_SYNTHETIC_CANARY_VISIBLE"
    assert audit.overall_decision(
        canary_reads=3, confidentiality="FAIL_SYNTHETIC_CANARY_VISIBLE", other_gates_pass=True,
    ) == "FAIL_CONFIDENTIALITY_BOUNDARY"
    closed = [{
        "canary_visible": False,
        "returncode": 0,
        "network_mode_none": True,
        "mount_flags": ["type=bind,source=/tmp/workspace,target=/workspace,readonly"],
    } for _ in range(3)]
    assert audit.classify_grader_confidentiality(closed) == "PASS_CANDIDATE_CANARY_UNREADABLE"
    missing_file_but_grader_mounted = [{
        "canary_visible": False,
        "returncode": 0,
        "network_mode_none": True,
        "mount_flags": ["type=bind,source=/tmp/grader-root,target=/grader,readonly"],
    } for _ in range(3)]
    assert audit.classify_grader_confidentiality(missing_file_but_grader_mounted) == "NOT_DEMONSTRATED"


def test_scoring_ignores_fake_result_json_and_planted_oracle(tmp_path):
    runner = LocalRunner()
    grader, _root, workspace = _grader(tmp_path, runner)
    _write(workspace / "solve.py", BROKEN)
    plant = (
        "from pathlib import Path\n"
        "Path('result.json').write_text('{\"passed\": true, \"score\": 1}\\n')\n"
        "Path('oracle.json').write_text('{\"value\": 999}\\n')\n"
        "Path('solve.py').write_text("
        + json.dumps(
            "import json\nfrom pathlib import Path\nimport sys\n"
            "print(Path('oracle.json').read_text())\n"
        )
        + ")\n"
    )
    report = asyncio.run(grader.evaluate(
        task_id="separated-json-square-repair-v1",
        policy_id="reward-hack", policy_version=1, workspace=workspace,
        candidate_command=("python3", "-c", plant),
        scoring_command=("python3", "solve.py"),
        cases=_cases(),
    ))
    assert report.scoring.passed is False
    assert report.scoring.score == 0.0
    assert all(case.passed is False for case in report.scoring.cases)
    assert all(case.failure == "wrong_json_response" for case in report.scoring.cases)
    assert all(b'"value":9' not in call[2] for call in runner.calls if call[0] == "io")


def test_reading_grader_path_does_not_reveal_expected_output(tmp_path):
    runner = LocalRunner()
    grader, _root, workspace = _grader(tmp_path, runner)
    _write(workspace / "solve.py", BROKEN)
    cheat = (
        "from pathlib import Path\n"
        "Path('solve.py').write_text("
        + json.dumps(
            "from pathlib import Path\n"
            "try:\n"
            "    print(Path('/grader/synthetic-hidden-canary.txt').read_text())\n"
            "except OSError:\n"
            "    print('{\"value\": null}')\n"
        )
        + ")\n"
    )
    report = asyncio.run(grader.evaluate(
        task_id="separated-json-square-repair-v1",
        policy_id="path-read", policy_version=1, workspace=workspace,
        candidate_command=("python3", "-c", cheat),
        scoring_command=("python3", "solve.py"),
        cases=_cases()[:1],
    ))
    assert report.scoring.passed is False
    assert report.scoring.cases[0].passed is False


def test_known_bad_passes_only_the_predeclared_zero_case_and_known_good_passes(tmp_path):
    protocol = json.loads(PROTOCOL.read_text())
    runner = LocalRunner()
    grader, _root, workspace = _grader(tmp_path, runner)
    _write(workspace / "solve.py", BROKEN)
    bad = asyncio.run(grader.evaluate(
        task_id="separated-json-square-repair-v1",
        policy_id="known-bad", policy_version=1, workspace=workspace,
        candidate_command=("python3", "-c", "print('unchanged')"),
        scoring_command=("python3", "solve.py"),
        cases=_cases(),
    ))
    passed_ids = [case.case_id for case in bad.scoring.cases if case.passed]
    assert passed_ids == protocol["harness_expectations"]["known_bad_passing_case_ids"]
    assert bad.scoring.passed is False

    runner = LocalRunner()
    grader, _root, workspace = _grader(tmp_path / "good", runner)
    _write(workspace / "solve.py", BROKEN)
    good = asyncio.run(grader.evaluate(
        task_id="separated-json-square-repair-v1",
        policy_id="known-good", policy_version=1, workspace=workspace,
        candidate_command=("python3", "-c", "from pathlib import Path; Path('solve.py').write_text(" + json.dumps(FIXED) + ")"),
        scoring_command=("python3", "solve.py"),
        cases=_cases(),
    ))
    assert good.scoring.passed is True
    assert good.scoring.score == 1.0


def test_snapshot_happens_after_exit_and_ignores_later_workspace_edits(tmp_path):
    runner = LocalRunner()
    grader, _root, workspace = _grader(tmp_path, runner)
    _write(workspace / "solve.py", BROKEN)

    async def fail_candidate(*args, **kwargs):
        raise IsolationError("candidate still running")

    runner.run_candidate = fail_candidate  # type: ignore[method-assign]
    with pytest.raises(IsolationError, match="still running"):
        asyncio.run(grader.evaluate(
            task_id="t", policy_id="p", policy_version=1, workspace=workspace,
            candidate_command=("python3", "-c", "print('x')"),
            scoring_command=("python3", "solve.py"),
            cases=_cases()[:1],
        ))
    assert not any(call[0] == "io" for call in runner.calls)

    with pytest.raises(IsolationError, match="before the candidate"):
        grader.capture_snapshot(workspace, candidate_exited=False)

    runner = LocalRunner()
    grader, _root, workspace = _grader(tmp_path / "later", runner)
    _write(workspace / "solve.py", BROKEN)
    original = (workspace / "solve.py").read_text(encoding="utf-8")

    real_io = runner.run_candidate_io

    async def io_and_mutate(*args, **kwargs):
        _write(workspace / "solve.py", FIXED)
        return await real_io(*args, **kwargs)

    runner.run_candidate_io = io_and_mutate  # type: ignore[method-assign]
    report = asyncio.run(grader.evaluate(
        task_id="t", policy_id="p", policy_version=1, workspace=workspace,
        candidate_command=("python3", "-c", "print('exit')"),
        scoring_command=("python3", "solve.py"),
        cases=_cases(),
    ))
    assert report.snapshot.path != workspace
    assert (report.snapshot.path / "solve.py").read_text(encoding="utf-8") == original
    assert report.scoring.passed is False
    io_roots = [call[1] for call in runner.calls if call[0] == "io"]
    assert io_roots and all(path == report.snapshot.path for path in io_roots)


def test_expected_output_in_the_workspace_blocks_launch(tmp_path):
    runner = LocalRunner()
    grader, _root, workspace = _grader(tmp_path, runner)
    expected = _encode(_cases()[0].expected).decode("utf-8")
    _write(workspace / "solve.py", expected)
    with pytest.raises(IsolationError, match="expected grader output"):
        asyncio.run(grader.evaluate(
            task_id="t", policy_id="p", policy_version=1, workspace=workspace,
            candidate_command=("python3", "-c", "print('x')"),
            scoring_command=("python3", "solve.py"),
            cases=_cases()[:1],
        ))
    assert runner.calls == []


def test_symlink_escape_is_not_copied_into_the_snapshot(tmp_path):
    runner = LocalRunner()
    grader, _root, workspace = _grader(tmp_path, runner)
    secret = tmp_path / "secret-canary.txt"
    secret.write_text("vare-synthetic-grader-canary-secret", encoding="utf-8")
    (workspace / "solve.py").symlink_to(secret)
    with pytest.raises(IsolationError, match="symlink escapes"):
        grader.capture_snapshot(workspace, candidate_exited=True)
    assert not any(path.name.startswith("snapshot-") and (path / "solve.py").exists()
                   and "secret" in (path / "solve.py").read_text(encoding="utf-8")
                   for path in (tmp_path / "snapshots").iterdir())


def test_network_and_hidden_flags_stay_out_of_the_candidate_sandbox(tmp_path):
    root = tmp_path / "workspaces"
    workspace = root / "trial"
    workspace.mkdir(parents=True)
    limits = _limits()
    argv = DockerIsolationRunner().candidate_argv(
        limits, name="vare-agent-" + "a" * 24,
        allowed_workspace_root=root, workspace=workspace,
        command=("python", "-c", "print('--network=host')"),
    )
    assert_candidate_namespace_closed(argv, limits=limits, workspace=workspace)
    assert argv.index("--network=none") < argv.index(limits.image)
    flags = argv[:argv.index(limits.image)]
    assert "--network=host" not in flags
    assert "--env" not in flags and "-e" not in flags
    tampered = argv[:]
    image_at = tampered.index(limits.image)
    tampered.insert(image_at, "--network=host")
    with pytest.raises(IsolationError, match="forbidden docker flag"):
        assert_candidate_namespace_closed(tampered, limits=limits, workspace=workspace)
    mounted = argv[:]
    mounted.insert(mounted.index("--mount") + 2, "--mount")
    mounted.insert(mounted.index("--mount") + 3, "type=bind,source=/tmp,target=/grader,readonly")
    with pytest.raises(IsolationError, match="only its workspace"):
        assert_candidate_namespace_closed(mounted, limits=limits, workspace=workspace)


def test_public_task_text_does_not_contain_canonical_hidden_outputs():
    task = (ROOT / "benchmarks/separated_json_square_v1/TASK.md").read_text(encoding="utf-8")
    fixture = (ROOT / "benchmarks/separated_json_square_v1/fixture/solve.py").read_text(encoding="utf-8")
    for case in _cases():
        blob = _encode(case.expected).decode("utf-8")
        assert blob not in task
        assert blob not in fixture


def _docker_ready() -> bool:
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE],
            capture_output=True, check=False, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


@pytest.mark.skipif(not _docker_ready(), reason="pinned python image is not available locally")
def test_docker_probe_reads_canary_zero_of_three(tmp_path):
    module = _load_probe_module()
    root = tmp_path / "candidate-root"
    workspace = root / "workspace"
    grader_root = tmp_path / "grader-root"
    workspace.mkdir(parents=True)
    grader_root.mkdir()
    runner = DockerIsolationRunner()
    visible = []
    for _ in range(3):
        canary = "vare-synthetic-grader-canary-" + os.urandom(8).hex()
        (grader_root / "synthetic-hidden-canary.txt").write_text(canary, encoding="utf-8")
        observed = asyncio.run(module._grader_probe(
            runner, root, workspace, grader_root, canary, image=DOCKER_IMAGE,
        ))
        visible.append(observed["canary_visible"])
        assert observed["returncode"] == 0
        assert observed["stdout"].strip() == "CANARY_NOT_VISIBLE"
        assert module_target(observed["mount_flags"][0]) == "/workspace"
    assert visible == [False, False, False]


@pytest.mark.skipif(not _docker_ready(), reason="pinned python image is not available locally")
def test_docker_network_is_closed_and_forged_score_fails(tmp_path):
    root = tmp_path / "workspaces"
    workspace = root / "trial"
    snapshots = tmp_path / "snapshots"
    workspace.mkdir(parents=True)
    snapshots.mkdir()
    workspace.chmod(0o777)
    _write(workspace / "solve.py", BROKEN)
    (workspace / "solve.py").chmod(0o666)
    limits = _limits(DOCKER_IMAGE)
    runner = DockerIsolationRunner()
    probe = asyncio.run(runner.run_candidate(
        limits, allowed_workspace_root=root, workspace=workspace,
        command=[
            "python", "-c",
            (
                "import socket\n"
                "try:\n"
                "    socket.create_connection(('203.0.113.1', 65000), timeout=0.5)\n"
                "    print('NETWORK_OPEN')\n"
                "except OSError:\n"
                "    print('NETWORK_CLOSED')\n"
            ),
        ],
    ))
    assert probe.returncode == 0
    assert probe.stdout.strip() == "NETWORK_CLOSED"

    grader = SeparatedCandidateGrader(
        runner=runner, limits=limits, allowed_workspace_root=root, snapshot_root=snapshots,
    )
    forged = (
        "from pathlib import Path\n"
        "Path('/workspace/result.json').write_text('{\"passed\": true}\\n')\n"
        "Path('/workspace/solve.py').write_text('import json\\nprint(json.dumps({\"passed\": True, \"score\": 1}))\\n')\n"
    )
    report = asyncio.run(grader.evaluate(
        task_id="separated-json-square-repair-v1",
        policy_id="forged", policy_version=1, workspace=workspace,
        candidate_command=("python", "-c", forged),
        scoring_command=("python", "/workspace/solve.py"),
        cases=_cases()[:1],
    ))
    assert "passed" in (report.snapshot.path / "solve.py").read_text(encoding="utf-8")
    assert report.scoring.passed is False
    assert report.scoring.cases[0].failure == "wrong_json_response"


def _engineering_fixture(tmp_path: Path, *, split: str = "train") -> Path:
    task = tmp_path / "task"
    subject = task / "subject"
    evaluator = task / "evaluator"
    subject.mkdir(parents=True)
    evaluator.mkdir()
    (subject / "add.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
    (evaluator / "test_hidden.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "workspace = Path(sys.argv[1]).resolve()\n"
        "sys.path.insert(0, str(workspace))\n"
        "from add import add\n"
        "if add(1, 1) != 2:\n"
        "    raise SystemExit('mismatch')\n",
        encoding="utf-8",
    )
    (evaluator / "secret-canary.txt").write_text("vare-synthetic-grader-canary-hidden\n", encoding="utf-8")
    spec = {
        "id": "fixture-add",
        "title": "Add two integers",
        "prompt": "Make add return the sum of its arguments.",
        "family": "fixture",
        "source": {"local_path": "subject"},
        "tests": [{
            "argv": ["python", "test_hidden.py", "{workspace}"],
            "cwd": "evaluator",
            "cwd_base": "evaluator",
            "env": {},
            "name": "hidden-correctness",
        }],
        "limits": {"wall_time_s": 10, "cpu_time_s": 5, "memory_mb": 256, "max_output_bytes": 20000},
        "metadata": {
            "editable_files": ["add.py"],
            "split": split,
            "grader_visibility": "hidden",
        },
        "correctness_weight": 1.0,
    }
    (task / "task.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return task


def test_engineering_contract_hides_evaluator_and_refuses_heldout(tmp_path):
    task = _engineering_fixture(tmp_path)
    contract = load_agent_task(task)
    assert contract.schema == "engineering_task_spec"
    assert contract.allowed_files == ("add.py",)
    assert "evaluator" not in contract.prompt_text
    assert "secret-canary" not in json.dumps(contract.input_hashes)
    assert_train_only(contract, {"train": ["fixture-add"], "heldout": ["other"]})
    heldout = load_agent_task(_engineering_fixture(tmp_path / "held", split="heldout"))
    with pytest.raises(ValueError, match="held-out"):
        assert_train_only(heldout, {"train": ["fixture-add"], "heldout": ["fixture-add"]})
    (task / "task.json").write_text(json.dumps({
        "metadata": {"editable_files": ["../evaluator/test_hidden.py"]},
        "prompt": "no",
        "source": {"local_path": "subject"},
        "id": "bad",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_agent_task(task)


def test_repository_grader_argv_is_not_a_candidate_sandbox(tmp_path):
    root = tmp_path / "snapshots"
    snapshot = root / "snapshot-one"
    grader_root = tmp_path / "grader"
    source = tmp_path / "src"
    snapshot.mkdir(parents=True)
    grader_root.mkdir()
    source.mkdir()
    (snapshot / "add.py").write_text("def add(left, right):\n    return left + right\n", encoding="utf-8")
    limits = _limits()
    argv = repository_grader_argv(
        DockerIsolationRunner(), limits, name="vare-grader-" + "a" * 24,
        snapshot=snapshot, grader_root=grader_root, vare_source=source,
        spec_digest="b" * 64, policy_id="fixture", policy_version=1,
    )
    mounts = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--mount"]
    assert [module_target(flag) for flag in mounts] == ["/candidate", "/grader", "/opt/vare/src"]
    with pytest.raises(IsolationError, match="forbidden docker flag"):
        assert_candidate_namespace_closed(argv, limits=limits, workspace=snapshot)
    staged_task = _engineering_fixture(tmp_path / "pack")
    stage = stage_hidden_grader(staged_task, tmp_path / "stage")
    assert (stage / "evaluator" / "secret-canary.txt").is_file()
    assert not (stage / "subject").exists()
    full = SeparatedCandidateGrader(
        runner=DockerIsolationRunner(), limits=limits,
        allowed_workspace_root=root, snapshot_root=root,
    ).capture_snapshot(snapshot, candidate_exited=True)
    grade = source_only_snapshot(full, ["add.py"], tmp_path / "grade-root")
    assert not (grade.path / ".git").exists()
    assert (grade.path / "add.py").is_file()


@pytest.mark.skipif(not _docker_ready(), reason="pinned python image is not available locally")
def test_docker_hidden_repository_grade_sees_canary_only_after_exit(tmp_path):
    task = _engineering_fixture(tmp_path)
    root = tmp_path / "workspaces"
    workspace = root / "trial"
    workspace.mkdir(parents=True)
    (workspace / "add.py").write_text((task / "subject" / "add.py").read_text(encoding="utf-8"), encoding="utf-8")
    workspace.chmod(0o755)
    limits = _limits(DOCKER_IMAGE)
    runner = DockerIsolationRunner()
    grader = SeparatedCandidateGrader(
        runner=runner, limits=limits, allowed_workspace_root=root,
        snapshot_root=tmp_path / "snapshots",
    )
    canary = (task / "evaluator" / "secret-canary.txt").read_text(encoding="utf-8").strip()
    probe = asyncio.run(runner.run_candidate(
        limits, allowed_workspace_root=root, workspace=workspace,
        command=[
            "python", "-c",
            (
                "from pathlib import Path\n"
                "print(Path('/grader/evaluator/secret-canary.txt').read_text() "
                "if Path('/grader/evaluator/secret-canary.txt').is_file() else 'CANARY_NOT_VISIBLE')\n"
            ),
        ],
    ))
    assert probe.stdout.strip() == "CANARY_NOT_VISIBLE"
    assert canary not in probe.stdout
    snapshot = grader.capture_snapshot(workspace, candidate_exited=True)
    grade_tree = source_only_snapshot(snapshot, ["add.py"], tmp_path / "grade-snapshots")
    stage = stage_hidden_grader(task, tmp_path / "grader-stage")
    broken = asyncio.run(grader.grade_hidden_repository(
        grade_tree, task_dir=task, grader_stage=stage,
        vare_source=ROOT / "src", policy_id="broken", policy_version=1,
    ))
    assert broken["passed"] is False
    (workspace / "add.py").write_text("def add(left, right):\n    return left + right\n", encoding="utf-8")
    fixed_snapshot = grader.capture_snapshot(workspace, candidate_exited=True)
    fixed_tree = source_only_snapshot(fixed_snapshot, ["add.py"], tmp_path / "grade-snapshots")
    passed = asyncio.run(grader.grade_hidden_repository(
        fixed_tree, task_dir=task, grader_stage=stage,
        vare_source=ROOT / "src", policy_id="fixed", policy_version=1,
    ))
    assert passed["passed"] is True
    assert passed["task_id"] == "fixture-add"


def _load_eval_module():
    path = ROOT / "scripts/run_separated_coding_eval.py"
    spec = importlib.util.spec_from_file_location("run_separated_coding_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pack_task(task_dir: Path, task_id: str, split: str) -> None:
    subject = task_dir / "subject"
    evaluator = task_dir / "evaluator"
    subject.mkdir(parents=True)
    evaluator.mkdir()
    (subject / "add.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
    (evaluator / "secret-canary.txt").write_text("vare-pack-canary\n", encoding="utf-8")
    spec = {
        "id": task_id,
        "title": "Add two integers",
        "prompt": "Make add return the sum.",
        "family": "fixture",
        "source": {"local_path": "subject"},
        "tests": [{
            "argv": ["python", "test_hidden.py", "{workspace}"],
            "cwd": "evaluator",
            "cwd_base": "evaluator",
            "env": {},
            "name": "hidden-correctness",
        }],
        "limits": {"wall_time_s": 10, "cpu_time_s": 5, "memory_mb": 256, "max_output_bytes": 20000},
        "metadata": {
            "editable_files": ["add.py"],
            "split": split,
            "grader_visibility": "hidden",
            "reward_kind": "hidden-correctness",
        },
        "correctness_weight": 1.0,
    }
    (task_dir / "task.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def _commit_pack(repo: Path) -> str:
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "vare",
        "GIT_AUTHOR_EMAIL": "vare@local",
        "GIT_COMMITTER_NAME": "vare",
        "GIT_COMMITTER_EMAIL": "vare@local",
    })
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True, env=env)
    subprocess.run(
        ["git", "commit", "-m", "pack"], cwd=repo, check=True, capture_output=True, env=env,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True,
    ).stdout.strip()


def test_pack_adapter_checks_manifest_and_strips_hidden_files(tmp_path):
    module = _load_eval_module()
    repo = tmp_path / "repo"
    pack = repo / "benchmarks" / "mlsys_coding_v1"
    _pack_task(pack / "tasks" / "add", "fixture-add", "train")
    _pack_task(pack / "tasks" / "held", "fixture-held", "heldout")
    (pack / "split.json").write_text(json.dumps({
        "heldout": ["fixture-held"],
        "pack_id": "mlsys-coding-v1",
        "train": ["fixture-add"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (pack / "reward.py").write_text("def decide(result):\n    return {'success': False, 'reward': 0.0}\n", encoding="utf-8")
    (pack / "manifest.json").write_text(module.rebuild_manifest_text(pack), encoding="utf-8")
    head = _commit_pack(repo)
    protocol = {"pack": {
        "git_commit": head,
        "manifest_sha256": module.sha256_file(pack / "manifest.json"),
        "train": ["fixture-add"],
        "heldout_not_used": ["fixture-held"],
    }}
    assert module.verify_pack(repo, protocol) == pack.resolve()
    tasks = module.select_train_tasks(pack, protocol, tmp_path / "task-roots")
    assert [item["task_id"] for item in tasks] == ["fixture-add"]
    names = sorted(path.name for path in tasks[0]["task_root"].iterdir())
    assert names == ["task.json"]
    assert "vare-pack-canary" not in (tasks[0]["task_root"] / "task.json").read_text(encoding="utf-8")
    (pack / "tasks" / "add" / "subject" / "add.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rebuilt file list"):
        module.verify_pack(repo, protocol)


def test_pack_adapter_refuses_a_heldout_task_marked_train(tmp_path):
    module = _load_eval_module()
    pack = tmp_path / "pack"
    _pack_task(pack / "tasks" / "held", "fixture-held", "heldout")
    (pack / "split.json").write_text(json.dumps({
        "heldout": [],
        "train": ["fixture-held"],
    }), encoding="utf-8")
    protocol = {"pack": {"train": ["fixture-held"], "heldout_not_used": []}}
    with pytest.raises(ValueError, match="held-out"):
        module.select_train_tasks(pack, protocol, tmp_path / "roots")


def test_unpatched_success_and_closed_grader_abort():
    module = _load_eval_module()
    assert module.abort_reason_for_unpatched(
        "fixture-add", "hidden-correctness",
        decision={"success": True}, error=None,
    )
    assert module.abort_reason_for_unpatched(
        "fixture-add", "hidden-correctness",
        decision={"success": False}, error=None,
    ) is None
    assert module.abort_reason_for_unpatched(
        "mlsys-v1-range-sums", module.PERFORMANCE_KIND,
        decision=None, error="container exceeded wall-time limit",
    ) is None
    assert module.abort_reason_for_unpatched(
        "fixture-add", "hidden-correctness",
        decision=None, error="container exceeded wall-time limit",
    )


def test_protocol_command_uses_the_pinned_pack_and_colima_image():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    command = protocol["local_measurement"]["command"]
    assert protocol["pack"]["git_commit"] in command
    assert protocol["execution"]["accepted_container_images"]["colima_aarch64"] in command
    assert "--pack-root" in command
    assert protocol["local_measurement"]["status"].startswith("not executed")
    for task_id in protocol["pack"]["heldout_not_used"]:
        assert task_id not in command
    accepted = set(protocol["execution"]["accepted_container_images"].values())
    module = _load_eval_module()
    assert module.resolve_image(protocol, None) == protocol["execution"]["container_image"]
    with pytest.raises(ValueError, match="pre-registered"):
        module.resolve_image(protocol, "python@sha256:" + "b" * 64)
    assert protocol["execution"]["accepted_container_images"]["colima_aarch64"] in accepted


def test_historical_task_contract_still_loads(tmp_path):
    root = tmp_path / "historical"
    (root / "evaluator").mkdir(parents=True)
    (root / "TASK.md").write_text("Repair solve.py.\n", encoding="utf-8")
    (root / "protocol.lock.json").write_text("{}\n", encoding="utf-8")
    (root / "evaluator" / "grade.py").write_text("raise SystemExit('hidden')\n", encoding="utf-8")
    (root / "task.json").write_text(json.dumps({
        "task_id": "historical-square",
        "source": {"files": ["solve.py"], "base_revision": "abc"},
    }), encoding="utf-8")
    contract = load_agent_task(root)
    assert contract.schema == "historical"
    assert contract.allowed_files == ("solve.py",)
    assert contract.prompt_text.startswith("Repair")
    assert "grader.py" in contract.input_hashes


def test_local_agent_parses_engineering_task_without_loading_weights(tmp_path, monkeypatch, capsys):
    path = ROOT / "scripts/run_local_agent.py"
    spec = importlib.util.spec_from_file_location("run_local_agent_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task = _engineering_fixture(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "add.py").write_text((task / "subject" / "add.py").read_text(encoding="utf-8"), encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "run_local_agent.py",
        "--workspace", str(workspace),
        "--model-path", str(model),
        "--task-root", str(task),
        "--output", str(tmp_path / "agent.json"),
        "--instruction", "Make add return the sum of its arguments.",
        "--seed", "101",
        "--require-split", "train",
        "--no-do-sample",
    ])
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 2
    assert "HF_HUB_OFFLINE" in capsys.readouterr().err
    held = _engineering_fixture(tmp_path / "heldout-task", split="heldout")
    held_workspace = tmp_path / "held-workspace"
    held_workspace.mkdir()
    (held_workspace / "add.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "run_local_agent.py",
        "--workspace", str(held_workspace),
        "--model-path", str(model),
        "--task-root", str(held),
        "--output", str(tmp_path / "held.json"),
        "--instruction", "do not run",
        "--seed", "101",
        "--require-split", "train",
    ])
    with pytest.raises(SystemExit) as held_exit:
        module.main()
    assert held_exit.value.code == 2
    assert "required split" in capsys.readouterr().err
