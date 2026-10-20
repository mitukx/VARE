"""Temporal separation between a candidate sandbox and hidden tests.

The candidate container mounts only its workspace. Expected JSON stays in the
operator process. After that container has exited, the workspace is copied to
a snapshot. JSON cases are then scored by ``TrustedJSONScorer``, whose
``evaluate`` signature is unchanged. Repository tasks are scored by a second
container that mounts the snapshot and the hidden evaluator. That grader argv
is not a candidate sandbox, and the candidate container never receives it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import secrets
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .container_isolation import (
    DockerIsolationRunner,
    IsolationError,
    IsolationLimits,
    IsolationResult,
    _mount_source,
)
from .environments.spec import EngineeringTaskSpec
from .evidence import sha256_json
from .isolated_candidate import (
    CandidateHarnessError,
    FunctionCall,
    grade_calls,
    paired_timing,
)
from .trusted_scoring import (
    JSONIOCase,
    JSONIOEvaluationReport,
    TrustedJSONScorer,
    _encode,
)

CANARY_CONTAINER_PATH = "/grader/synthetic-hidden-canary.txt"
CANARY_MISS = "CANARY_NOT_VISIBLE"
CANARY_PROBE_SOURCE = (
    "from pathlib import Path\n"
    "p = Path('/grader/synthetic-hidden-canary.txt')\n"
    "try:\n"
    "    text = p.read_text() if p.is_file() else 'CANARY_NOT_VISIBLE'\n"
    "except OSError:\n"
    "    text = 'CANARY_NOT_VISIBLE'\n"
    "print(text)\n"
)
_FORBIDDEN_DOCKER_FLAGS = (
    "--env", "-e", "--env-file", "--network=host", "--network=bridge",
    "--privileged", "--publish", "-p", "--pid=host", "--ipc=host",
)


def mount_flags(argv: Sequence[str]) -> list[str]:
    return [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--mount"]


def mount_target(flag: str) -> str:
    if ",target=" not in flag:
        return ""
    return flag.split(",target=", 1)[1].split(",", 1)[0]


def mount_source(flag: str) -> str:
    prefix = "type=bind,source="
    if not flag.startswith(prefix) or ",target=" not in flag:
        return ""
    return flag[len(prefix):].split(",target=", 1)[0]


def canary_visible_under_mounts(
    argv: Sequence[str], *, canary_file: Path, canary: str,
    container_path: str = CANARY_CONTAINER_PATH,
) -> bool:
    """Return whether a bind-mount argv can read ``canary`` at ``container_path``.

    This is a mount-table model of the probe. It does not start a container.
    """
    if not canary or not canary_file.is_file():
        return False
    if canary_file.read_text(encoding="utf-8") != canary:
        return False
    host = canary_file.resolve()
    for flag in mount_flags(argv):
        source_text = mount_source(flag)
        target = mount_target(flag)
        if not source_text or not target:
            continue
        source = Path(source_text).resolve()
        try:
            relative = host.relative_to(source)
        except ValueError:
            continue
        mounted = "/" + str(Path(target.lstrip("/")) / relative)
        if mounted == container_path or str(Path(mounted)) == str(Path(container_path)):
            return True
    return False


def assert_candidate_namespace_closed(
    argv: Sequence[str], *, limits: IsolationLimits, workspace: Path,
    forbidden_texts: Sequence[str] = (), readonly: bool = False,
) -> None:
    """Reject a candidate argv that mounts or embeds hidden grader material."""
    if limits.image not in argv:
        raise IsolationError("candidate argv is missing its pinned image")
    image_at = argv.index(limits.image)
    flags = list(argv[:image_at])
    if flags.count("--network=none") != 1:
        raise IsolationError("candidate sandbox requires exactly one --network=none docker flag")
    if any(flag in flags for flag in _FORBIDDEN_DOCKER_FLAGS):
        raise IsolationError("candidate sandbox enables a forbidden docker flag")
    if any("/var/run/docker.sock" in flag for flag in flags):
        raise IsolationError("candidate sandbox must not receive the Docker socket")
    mounts = [flags[index + 1] for index, value in enumerate(flags[:-1]) if value == "--mount"]
    if mounts != mount_flags(argv):
        raise IsolationError("candidate command must not add mounts")
    if len(mounts) != 1:
        raise IsolationError("candidate sandbox must mount only its workspace")
    source = mount_source(mounts[0])
    target = mount_target(mounts[0])
    if target != "/workspace":
        raise IsolationError("candidate mount target must be /workspace")
    if Path(source).resolve() != Path(workspace).resolve():
        raise IsolationError("candidate mount source is not the candidate workspace")
    if readonly and not mounts[0].endswith(",readonly"):
        raise IsolationError("scoring mount must be read-only")
    if not readonly and mounts[0].endswith(",readonly"):
        raise IsolationError("candidate workspace mount must be writable")
    leaked = [text for text in forbidden_texts if text and text in "\n".join(argv)]
    if leaked:
        raise IsolationError("hidden grader material leaked into the candidate argv")


def assert_expected_outputs_absent(root: Path, cases: Sequence[JSONIOCase]) -> None:
    """Fail closed when operator-held expected outputs are already in the workspace."""
    forbidden = [_encode(case.expected) for case in cases]
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            if name == "synthetic-hidden-canary.txt":
                raise IsolationError("hidden canary file is inside the candidate workspace")
            size = path.stat().st_size
            if size > 2_000_000:
                raise IsolationError("candidate file exceeds the hidden-output scan bound")
            blob = path.read_bytes()
            if any(item in blob for item in forbidden):
                raise IsolationError(
                    "expected grader output is present in the candidate workspace"
                )


def _hash_tree(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    source: Path
    path: Path
    file_sha256: dict[str, str]
    captured_after_exit: bool


@dataclass(frozen=True, slots=True)
class SeparatedEvaluation:
    candidate_returncode: int
    candidate_mounts: tuple[str, ...]
    snapshot: WorkspaceSnapshot
    changed_files: tuple[str, ...]
    scoring: JSONIOEvaluationReport


class SeparatedCandidateGrader:
    """Run a candidate container, snapshot it after exit, then score the copy.

    ``candidate_command`` executes in the writable sandbox. ``scoring_command``
    is the operator-owned entrypoint passed to ``TrustedJSONScorer`` and is not
    read from the workspace.
    """

    def __init__(
        self, *, runner: DockerIsolationRunner, limits: IsolationLimits,
        allowed_workspace_root: Path, snapshot_root: Path,
    ) -> None:
        self.runner = runner
        self.limits = limits
        self.allowed_workspace_root = allowed_workspace_root
        self.snapshot_root = snapshot_root
        self._paths = DockerIsolationRunner()

    def candidate_argv(
        self, *, name: str, workspace: Path, command: Sequence[str],
        readonly_workspace: bool = False, interactive: bool = False,
        forbidden_texts: Sequence[str] = (),
    ) -> list[str]:
        argv = self.runner.candidate_argv(
            self.limits, name=name, allowed_workspace_root=self.allowed_workspace_root,
            workspace=workspace, command=command,
            readonly_workspace=readonly_workspace, interactive=interactive,
        )
        assert_candidate_namespace_closed(
            argv, limits=self.limits, workspace=workspace,
            forbidden_texts=forbidden_texts, readonly=readonly_workspace,
        )
        return argv

    def capture_snapshot(self, workspace: Path, *, candidate_exited: bool) -> WorkspaceSnapshot:
        if candidate_exited is not True:
            raise IsolationError("refusing to snapshot before the candidate container exits")
        source = self._paths._workspace(self.allowed_workspace_root, workspace)
        destination_root = Path(self.snapshot_root)
        if not destination_root.is_absolute() or destination_root.is_symlink():
            raise IsolationError("snapshot root must be an absolute non-symlink path")
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / ("snapshot-" + secrets.token_hex(8))
        destination.mkdir()
        hashes: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
            current = Path(dirpath)
            relative_dir = current.relative_to(source)
            dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
            (destination / relative_dir).mkdir(parents=True, exist_ok=True)
            for name in filenames:
                path = current / name
                target = destination / relative_dir / name
                if path.is_symlink():
                    link = Path(os.readlink(path))
                    resolved = (path.parent / link).resolve(strict=False) if not link.is_absolute() else link
                    try:
                        resolved.resolve().relative_to(source)
                    except ValueError as exc:
                        raise IsolationError(
                            "candidate snapshot symlink escapes the workspace"
                        ) from exc
                    target.symlink_to(link)
                    continue
                if not path.is_file():
                    continue
                data = path.read_bytes()
                target.write_bytes(data)
                relative = (relative_dir / name).as_posix()
                hashes[relative] = hashlib.sha256(data).hexdigest()
        return WorkspaceSnapshot(
            source=source, path=destination, file_sha256=hashes, captured_after_exit=True,
        )

    async def grade_snapshot(
        self, snapshot: WorkspaceSnapshot, *, task_id: str, policy_id: str,
        policy_version: int, command: Sequence[str], cases: Sequence[JSONIOCase],
        total_timeout_s: float = 600.0,
    ) -> JSONIOEvaluationReport:
        if snapshot.captured_after_exit is not True:
            raise IsolationError("refusing to grade a snapshot taken before candidate exit")
        forbidden = [_encode(case.expected).decode("utf-8") for case in cases]
        guarded = _OracleGuardedRunner(self.runner, forbidden_texts=tuple(forbidden))
        scorer = TrustedJSONScorer(
            runner=guarded, limits=self.limits,
            allowed_workspace_root=self.snapshot_root,
        )
        return await scorer.evaluate(
            task_id=task_id, policy_id=policy_id, policy_version=policy_version,
            workspace=snapshot.path, command=command, cases=cases,
            total_timeout_s=total_timeout_s,
        )

    async def grade_hidden_repository(
        self, snapshot: WorkspaceSnapshot, *, task_dir: Path, grader_stage: Path,
        vare_source: Path, policy_id: str, policy_version: int,
        grade_limits: IsolationLimits | None = None,
    ) -> dict:
        """Score a snapshot with hidden evaluator files in a new container.

        The candidate container must already have exited. This mount is not
        the candidate mount: it is the only place the evaluator directory is
        attached, and only after the snapshot exists.
        """
        if snapshot.captured_after_exit is not True:
            raise IsolationError("refusing to grade a snapshot taken before candidate exit")
        spec = EngineeringTaskSpec.load(task_dir / "task.json")
        digest = sha256_json(spec.to_dict())
        limits = self.limits if grade_limits is None else grade_limits
        name = "vare-grader-" + secrets.token_hex(12)
        argv = repository_grader_argv(
            self.runner, limits, name=name, snapshot=snapshot.path,
            grader_root=grader_stage, vare_source=vare_source, spec_digest=digest,
            policy_id=policy_id, policy_version=policy_version,
        )
        if any(mount_target(flag) == "/workspace" for flag in mount_flags(argv)):
            raise IsolationError("repository grader must not use the candidate workspace mount")
        result = await self.runner.execute(argv, limits)
        if result.returncode != 0:
            raise IsolationError("hidden repository grader failed closed")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise IsolationError("hidden repository grader returned invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("task_id") != spec.id:
            raise IsolationError("hidden repository grader identity mismatch")
        return payload

    async def evaluate(
        self, *, task_id: str, policy_id: str, policy_version: int,
        workspace: Path, candidate_command: Sequence[str], scoring_command: Sequence[str],
        cases: Sequence[JSONIOCase], total_timeout_s: float = 600.0,
        check_workspace_for_oracles: bool = True,
    ) -> SeparatedEvaluation:
        if not cases:
            raise IsolationError("hidden case set is empty")
        if check_workspace_for_oracles:
            assert_expected_outputs_absent(workspace, cases)
        forbidden = [_encode(case.expected).decode("utf-8") for case in cases]
        before = _hash_tree(workspace)
        name = "vare-agent-" + secrets.token_hex(12)
        argv = self.candidate_argv(
            name=name, workspace=workspace, command=candidate_command,
            forbidden_texts=forbidden,
        )
        result = await self.runner.run_candidate(
            self.limits, allowed_workspace_root=self.allowed_workspace_root,
            workspace=workspace, command=candidate_command,
        )
        snapshot = self.capture_snapshot(workspace, candidate_exited=True)
        changed = tuple(sorted(
            set(before) ^ set(snapshot.file_sha256)
            | {key for key in before.keys() & snapshot.file_sha256.keys()
               if before[key] != snapshot.file_sha256[key]}
        ))
        scoring = await self.grade_snapshot(
            snapshot, task_id=task_id, policy_id=policy_id, policy_version=policy_version,
            command=scoring_command, cases=cases, total_timeout_s=total_timeout_s,
        )
        return SeparatedEvaluation(
            candidate_returncode=result.returncode,
            candidate_mounts=tuple(mount_flags(argv)),
            snapshot=snapshot,
            changed_files=changed,
            scoring=scoring,
        )


class _OracleGuardedRunner:
    """Prove each scoring container's argv before forwarding the request."""

    def __init__(self, inner, *, forbidden_texts: tuple[str, ...]) -> None:
        self.inner = inner
        self.forbidden_texts = forbidden_texts

    async def run_candidate_io(
        self, limits: IsolationLimits, *, allowed_workspace_root: Path,
        workspace: Path, command: Sequence[str], stdin_data: bytes,
    ) -> IsolationResult:
        if not hasattr(self.inner, "candidate_argv"):
            raise IsolationError("scoring runner cannot prove its mounts")
        argv = self.inner.candidate_argv(
            limits, name="vare-agent-" + "a" * 24,
            allowed_workspace_root=allowed_workspace_root, workspace=workspace,
            command=command, readonly_workspace=True, interactive=True,
        )
        assert_candidate_namespace_closed(
            argv, limits=limits, workspace=workspace,
            forbidden_texts=self.forbidden_texts, readonly=True,
        )
        for text in self.forbidden_texts:
            if text and text.encode("utf-8") in stdin_data:
                raise IsolationError("expected output leaked into the scoring request")
        return await self.inner.run_candidate_io(
            limits, allowed_workspace_root=allowed_workspace_root,
            workspace=workspace, command=command, stdin_data=stdin_data,
        )


def repository_grader_argv(
    runner: DockerIsolationRunner, limits: IsolationLimits, *, name: str,
    snapshot: Path, grader_root: Path, vare_source: Path, spec_digest: str,
    policy_id: str, policy_version: int,
) -> list[str]:
    """Grader argv used only after the candidate container has exited.

    ``grader_root`` contains the task spec and hidden evaluator files.
    ``vare_source`` is the repository ``src`` tree, not the task pack.
    The candidate phase must not use this argv.
    """
    if not re.fullmatch(r"vare-grader-[0-9a-f]{24}", name):
        raise IsolationError("invalid repository grader container identity")
    if not re.fullmatch(r"[0-9a-f]{64}", spec_digest):
        raise IsolationError("invalid trusted task spec digest")
    checked_snapshot = runner._workspace(snapshot.parent, snapshot)
    checked_grader = _mount_source(grader_root, directory=True)
    checked_source = _mount_source(vare_source, directory=True)
    if checked_grader in checked_snapshot.parents or checked_snapshot in checked_grader.parents:
        raise IsolationError("grader root overlaps the candidate snapshot")
    if checked_source in checked_grader.parents or checked_grader in checked_source.parents:
        raise IsolationError("vare source overlaps the hidden grader root")
    return [
        *runner._common(limits, name),
        "--env", "PYTHONPATH=/opt/vare/src",
        "--mount", f"type=bind,source={checked_snapshot},target=/candidate,readonly",
        "--mount", f"type=bind,source={checked_grader},target=/grader,readonly",
        "--mount", f"type=bind,source={checked_source},target=/opt/vare/src,readonly",
        "--tmpfs", "/scratch:rw,nosuid,nodev,size=256m",
        "--workdir=/",
        limits.image, "python", "-m", "vare.sandboxed_repository", "--worker",
        "--spec", "/grader/task.json",
        "--candidate", "/candidate",
        "--scratch", "/scratch",
        "--expected-spec-sha256", spec_digest,
        "--policy-id", policy_id,
        "--policy-version", str(policy_version),
        "--baseline-json", "{}",
    ]


def chmod_world_readable(root: Path) -> None:
    """Let the unprivileged container user read a copied tree."""
    if root.is_symlink():
        raise IsolationError("refusing to chmod through a symlink")
    root.chmod(0o755)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        current.chmod(0o755)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            path.chmod(0o644)


def stage_hidden_grader(task_dir: Path, destination: Path) -> Path:
    """Copy the spec and evaluator tree without the candidate subject."""
    spec_path = task_dir / "task.json"
    if not spec_path.is_file():
        raise IsolationError("task spec is missing")
    evaluator = task_dir / "evaluator"
    if not evaluator.is_dir():
        raise IsolationError("hidden evaluator directory is missing")
    destination.mkdir(parents=True)
    (destination / "task.json").write_bytes(spec_path.read_bytes())
    _copy_tree_no_escape(evaluator, destination / "evaluator")
    subject = task_dir / "subject"
    if subject.exists() and destination in subject.resolve().parents:
        raise IsolationError("staged grader escaped into the subject tree")
    chmod_world_readable(destination)
    return destination


def _copy_tree_no_escape(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.mkdir(parents=True)
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        current = Path(dirpath)
        relative = current.relative_to(source)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        (destination / relative).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            path = current / name
            target = destination / relative / name
            if path.is_symlink():
                raise IsolationError("hidden evaluator symlink is not copied")
            if path.is_file():
                target.write_bytes(path.read_bytes())


def source_only_snapshot(
    snapshot: WorkspaceSnapshot, allowed_files: Sequence[str], destination_root: Path,
) -> WorkspaceSnapshot:
    """Copy allowlisted files into a grade tree that has no ``.git`` metadata."""
    if snapshot.captured_after_exit is not True:
        raise IsolationError("refusing to derive a grade tree before candidate exit")
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / ("grade-" + secrets.token_hex(8))
    destination.mkdir()
    hashes: dict[str, str] = {}
    for name in allowed_files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise IsolationError("grade file escapes the snapshot")
        source = snapshot.path / relative
        if not source.is_file() or source.is_symlink():
            raise IsolationError("allowlisted snapshot file is missing")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        target.write_bytes(data)
        hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    chmod_world_readable(destination)
    return WorkspaceSnapshot(
        source=snapshot.path, path=destination, file_sha256=hashes, captured_after_exit=True,
    )


def harness_required_record() -> dict[str, Any]:
    """A repository task with no host-side harness spec is not a pass.

    The pack reward's exit-0 rule is not consulted.
    """
    return {
        "decision": {
            "success": False,
            "reward": 0.0,
            "conservative_speedup": None,
            "oracle_subject_ratio": None,
            "workload_sufficient": False,
            "reason": "isolated_harness_required",
        },
    }


def _finite_number(value: Any, label: str, *, allow_zero: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        or (not allow_zero and float(value) == 0)
    ):
        raise IsolationError(f"isolated call spec {label} is invalid")
    return float(value)


def _load_isolated_calls(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IsolationError("isolated call spec is unreadable") from exc
    if type(payload) is not dict:
        raise IsolationError("isolated call spec must be an object")
    if payload.get("task_kind") not in {"correctness", "performance"}:
        raise IsolationError("isolated call spec task_kind is invalid")
    if not isinstance(payload.get("module"), str) or not isinstance(payload.get("function"), str):
        raise IsolationError("isolated call spec names are invalid")
    payload["rtol"] = _finite_number(payload.get("rtol", 0.0), "rtol", allow_zero=True)
    payload["atol"] = _finite_number(payload.get("atol", 0.0), "atol", allow_zero=True)
    payload["timeout_s"] = _finite_number(payload.get("timeout_s", 30), "timeout_s", allow_zero=False)
    return payload


def _call_rows(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = payload.get(key)
    if type(rows) is not list or not rows:
        raise IsolationError("isolated call spec has no cases")
    checked: list[dict[str, Any]] = []
    for row in rows:
        if type(row) is not dict or "expected" not in row:
            raise IsolationError("isolated call spec case is invalid")
        expected_exception = row.get("expected_exception")
        if expected_exception is not None and (
            type(expected_exception) is not str or not expected_exception.isidentifier()
        ):
            raise IsolationError("isolated call expected_exception is invalid")
        args = row.get("args", [])
        kwargs = row.get("kwargs", {})
        if type(args) is not list or type(kwargs) is not dict:
            raise IsolationError("isolated call spec case is invalid")
        checked.append(row)
    return checked


def _paired_ratio_lower_bound(
    baselines: Sequence[float], candidates: Sequence[float], t_critical: float,
) -> tuple[float, float]:
    """Return the pack's lower 95% bound and mean paired speedup."""
    if len(baselines) != len(candidates) or len(baselines) < 2:
        raise IsolationError("performance metric requires paired timing samples")
    ratios: list[float] = []
    for baseline, candidate in zip(baselines, candidates, strict=True):
        if (
            isinstance(baseline, bool) or isinstance(candidate, bool)
            or not isinstance(baseline, (int, float))
            or not isinstance(candidate, (int, float))
            or not math.isfinite(float(baseline))
            or not math.isfinite(float(candidate))
            or baseline < 0 or candidate <= 0
        ):
            raise IsolationError("performance timing samples are invalid")
        ratios.append(float(baseline) / float(candidate))
    mean = statistics.mean(ratios)
    lower = mean - t_critical * statistics.stdev(ratios) / math.sqrt(len(ratios))
    return lower, mean


def grade_isolated_subject(subject: Path, calls_path: Path, *, image: str) -> dict[str, Any]:
    """Score one snapshot from the host. A child exit status is not a pass."""
    spec = _load_isolated_calls(calls_path)
    module = str(spec["module"])
    function = str(spec["function"])
    rtol = float(spec["rtol"])
    atol = float(spec["atol"])
    timeout_s = float(spec["timeout_s"])
    tuple_paths = spec.get("tuple_paths", [])
    if type(tuple_paths) is not list or any(type(path) is not list for path in tuple_paths):
        raise IsolationError("isolated tuple paths are invalid")
    try:
        if spec["task_kind"] == "correctness":
            rows = _call_rows(spec, "calls")
            calls = [
                FunctionCall(
                    str(row.get("case_id", index)), module, function,
                    list(row.get("args", [])), row["expected"], dict(row.get("kwargs", {})),
                    expected_exception=row.get("expected_exception"),
                    tuple_paths=tuple(tuple(path) for path in tuple_paths),
                )
                for index, row in enumerate(rows)
            ]
            report = grade_calls(
                subject, calls, task_kind="correctness", image=image,
                timeout_s=timeout_s, rtol=rtol, atol=atol,
            )
            passed = report.passed
            return {
                "decision": {
                    "success": passed,
                    "reward": 1.0 if passed else 0.0,
                    "conservative_speedup": None,
                    "oracle_subject_ratio": None,
                    "workload_sufficient": False,
                    "reason": "correctness" if passed else "correctness_failed",
                },
                "grade_passed": passed,
                "grade": report.to_dict(),
                "test_timed_out": False,
            }
        rows = _call_rows(spec, "cases")
        repeats = spec.get("repeats")
        seed = spec.get("seed")
        source = spec.get("baseline_source")
        if type(repeats) is not int or isinstance(repeats, bool) or repeats != len(rows):
            raise IsolationError("isolated call spec repeats do not match its cases")
        if type(seed) is not int or isinstance(seed, bool):
            raise IsolationError("isolated call spec seed is invalid")
        if type(source) is not str or not source or len(source.encode("utf-8")) > 100_000:
            raise IsolationError("isolated call spec baseline is invalid")
        with tempfile.TemporaryDirectory(prefix="vare-oracle-") as folder:
            oracle = Path(folder)
            (oracle / f"{module}.py").write_text(source, encoding="utf-8")

            def make_case(index: int) -> tuple[list[Any], dict[str, Any], Any]:
                row = rows[index]
                return list(row.get("args", [])), dict(row.get("kwargs", {})), row["expected"]

            timing = paired_timing(
                subject, oracle, module=module, function=function, make_case=make_case,
                repeats=repeats, seed=seed, task_kind="performance", image=image,
                timeout_s=timeout_s, rtol=rtol, atol=atol,
                tuple_paths=tuple_paths,
            )
        metric = spec.get("performance_metric")
        if type(metric) is not dict or metric.get("name") != "paired_ratio_student_t_lower":
            raise IsolationError("isolated performance metric is missing or unsupported")
        try:
            t_critical = _finite_number(metric.get("t_critical"), "t_critical", allow_zero=False)
            threshold = _finite_number(metric.get("success_threshold"), "success_threshold", allow_zero=False)
            saturation = _finite_number(metric.get("reward_saturation"), "reward_saturation", allow_zero=False)
            if timing.repeats != repeats or len(timing.pairs) != repeats:
                raise IsolationError("performance timing count does not match the frozen metric")
            baselines = [pair.baseline_s for pair in timing.pairs]
            candidates = [pair.candidate_s for pair in timing.pairs]
            if any(value is None for value in (*baselines, *candidates)):
                raise IsolationError("performance timing sample is missing")
            conservative_speedup, mean_speedup = _paired_ratio_lower_bound(
                baselines, candidates, t_critical,  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, IsolationError):
                raise
            raise IsolationError("isolated performance metric is invalid") from exc
        passed = timing.output_pass
        reward = min(1.0, max(0.0, conservative_speedup / saturation)) if passed else 0.0
        success = passed and conservative_speedup >= threshold
        decision = {
            "success": success,
            "reward": reward,
            "conservative_speedup": conservative_speedup,
            "oracle_subject_ratio": mean_speedup,
            "workload_sufficient": True,
            "reason": (
                "performance" if success else
                "performance_below_threshold" if passed else "output_mismatch"
            ),
        }
        return {
            "decision": decision,
            "grade_passed": timing.output_pass,
            "grade": timing.to_dict(),
            "test_timed_out": False,
        }
    except CandidateHarnessError as exc:
        raise IsolationError(str(exc)) from exc


def build_confidentiality_probe_argv(
    runner: DockerIsolationRunner, limits: IsolationLimits, *, name: str,
    allowed_workspace_root: Path, workspace: Path,
) -> list[str]:
    """Argv for the historical grader-canary probe, without a grader mount."""
    grader = SeparatedCandidateGrader(
        runner=runner, limits=limits,
        allowed_workspace_root=allowed_workspace_root,
        snapshot_root=allowed_workspace_root,
    )
    return grader.candidate_argv(
        name=name, workspace=workspace,
        command=("python", "-c", CANARY_PROBE_SOURCE),
        readonly_workspace=True,
    )
