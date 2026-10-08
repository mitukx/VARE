from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from ..evidence import sha256_file, sha256_json
from .spec import CommandSpec, EngineeringTaskSpec, MetricSpec, ResourceLimits


@dataclass(frozen=True, slots=True)
class CommandResult:
    name: str
    argv: tuple[str, ...]
    returncode: int
    wall_s: float
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    value: float
    baseline: float | None
    relative_gain: float | None
    passed_regression_gate: bool


@dataclass(slots=True)
class EnvironmentResult:
    task_id: str
    passed: bool
    score: float
    correctness: float
    tests: list[CommandResult] = field(default_factory=list)
    metrics: list[MetricResult] = field(default_factory=list)
    integrity_ok: bool = True
    integrity_failures: list[str] = field(default_factory=list)
    workspace_diff: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "score": self.score,
            "correctness": self.correctness,
            "tests": [asdict(x) for x in self.tests],
            "metrics": [asdict(x) for x in self.metrics],
            "integrity_ok": self.integrity_ok,
            "integrity_failures": list(self.integrity_failures),
            "workspace_diff": self.workspace_diff,
            "provenance": dict(self.provenance),
        }


class Workspace:
    """Ephemeral checkout for one engineering attempt.

    The source repository is never mutated. Local sources are copied; remote
    sources are shallow-cloned when the caller explicitly provides a clone URL.
    """

    def __init__(self, spec: EngineeringTaskSpec, *, root: str | Path | None = None) -> None:
        self.spec = spec
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        if root is None:
            self._tmp = tempfile.TemporaryDirectory(prefix=f"vare-{spec.id}-")
            self.path = Path(self._tmp.name) / "workspace"
        else:
            self.path = Path(root)
        self._materialize()

    def _materialize(self) -> None:
        source = self.spec.source
        if source.local_path:
            raw_src = Path(source.local_path)
            if raw_src.is_absolute():
                src = raw_src.resolve()
            else:
                base = (
                    Path(self.spec._source_base)
                    if self.spec._source_base is not None
                    else Path.cwd()
                )
                src = (base / raw_src).resolve()
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copytree(src, self.path, dirs_exist_ok=False, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
            # Always create an ephemeral baseline commit. This gives every local
            # task immutable before/after diff evidence even when the source is
            # a plain directory rather than a Git checkout.
            subprocess.run(["git", "init", "-q"], cwd=self.path, check=True)
            subprocess.run(["git", "add", "-A"], cwd=self.path, check=True)
            subprocess.run(
                ["git", "-c", "user.name=VARE", "-c", "user.email=vare@localhost", "commit", "-qm", "workspace baseline"],
                cwd=self.path,
                check=True,
            )
        else:
            assert source.clone_url
            subprocess.run(["git", "clone", "--quiet", "--no-tags", source.clone_url, str(self.path)], check=True)
            if source.base_revision:
                subprocess.run(["git", "checkout", "--quiet", source.base_revision], cwd=self.path, check=True)
        if source.local_path and source.base_revision:
            raise ValueError("base_revision is only supported for clone_url sources")

    def diff(self) -> str:
        if not (self.path / ".git").exists():
            return ""
        proc = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary"],
            cwd=self.path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return proc.stdout

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _preexec(limits: ResourceLimits):
    def apply() -> None:
        try:
            import resource
        except ImportError:
            return
        if limits.cpu_time_s is not None:
            resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_time_s, limits.cpu_time_s + 1))
        if limits.memory_mb is not None:
            cap = limits.memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            except (ValueError, OSError):
                pass
    return apply


def run_command(
    workspace: Path,
    command: CommandSpec,
    limits: ResourceLimits,
    *,
    evaluator_root: Path | None = None,
) -> CommandResult:
    workspace = workspace.resolve()
    if command.cwd_base == "workspace":
        base = workspace
    else:
        if evaluator_root is None:
            raise ValueError("evaluator-root command requires evaluator_root")
        base = evaluator_root.resolve()
    cwd = (base / command.cwd).resolve()
    if base not in (cwd, *cwd.parents):
        raise ValueError("command cwd escapes declared base")
    if not cwd.exists():
        raise FileNotFoundError(cwd)
    substitutions = {
        "{workspace}": str(workspace),
        "{evaluator}": str(evaluator_root.resolve()) if evaluator_root is not None else "",
    }
    argv = []
    for item in command.argv:
        value = item
        for token, replacement in substitutions.items():
            value = value.replace(token, replacement)
        argv.append(value)
    env = os.environ.copy()
    env.update(command.env)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            timeout=limits.wall_time_s,
            check=False,
            preexec_fn=_preexec(limits) if os.name == "posix" else None,
        )
        elapsed = time.perf_counter() - start
        stdout = proc.stdout[: limits.max_output_bytes].decode("utf-8", errors="replace")
        stderr = proc.stderr[: limits.max_output_bytes].decode("utf-8", errors="replace")
        return CommandResult(command.name, tuple(argv), int(proc.returncode), elapsed, stdout, stderr, False)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        out = (exc.stdout or b"")[: limits.max_output_bytes]
        err = (exc.stderr or b"")[: limits.max_output_bytes]
        stdout = out if isinstance(out, str) else out.decode("utf-8", errors="replace")
        stderr = err if isinstance(err, str) else err.decode("utf-8", errors="replace")
        return CommandResult(command.name, tuple(argv), 124, elapsed, stdout, stderr, True)


def _metric_value(result: CommandResult, spec: MetricSpec) -> float:
    if not result.passed:
        raise ValueError(f"metric command failed: {spec.name}")
    payload = json.loads(result.stdout)
    value = float(payload[spec.json_key])
    if not math.isfinite(value):
        raise ValueError(f"metric {spec.name} is non-finite")
    return value


def _relative_gain(candidate: float, baseline: float, direction: str) -> float:
    denom = max(abs(baseline), 1e-12)
    signed = candidate - baseline if direction == "max" else baseline - candidate
    return signed / denom


class ExecutableEvaluator:
    """Trusted, fail-closed evaluator for repository-editing environments."""

    def __init__(self, spec: EngineeringTaskSpec, *, evaluator_root: str | Path | None = None) -> None:
        self.spec = spec
        self.evaluator_root = None if evaluator_root is None else Path(evaluator_root).resolve()

    def _check_integrity(self, workspace: Path) -> tuple[bool, list[str]]:
        failures: list[str] = []
        for protected in self.spec.protected_paths:
            path = (workspace / protected.path).resolve()
            if workspace.resolve() not in (path, *path.parents):
                failures.append(f"protected path escapes workspace: {protected.path}")
                continue
            if not path.is_file():
                failures.append(f"protected file missing: {protected.path}")
                continue
            if sha256_file(path) != protected.sha256:
                failures.append(f"protected file modified: {protected.path}")
        return not failures, failures

    def evaluate(
        self,
        workspace: str | Path,
        *,
        baseline_metrics: dict[str, float] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> EnvironmentResult:
        root = Path(workspace).resolve()
        integrity_ok, integrity_failures = self._check_integrity(root)
        tests: list[CommandResult] = []
        if integrity_ok:
            tests = [run_command(root, cmd, self.spec.limits, evaluator_root=self.evaluator_root) for cmd in self.spec.tests]
        correctness = 0.0 if not tests else sum(x.passed for x in tests) / len(tests)
        metric_results: list[MetricResult] = []
        metrics_ok = True
        if integrity_ok and correctness == 1.0:
            for metric in self.spec.metrics:
                cmd_result = run_command(root, metric.command, self.spec.limits, evaluator_root=self.evaluator_root)
                try:
                    value = _metric_value(cmd_result, metric)
                except Exception:
                    metrics_ok = False
                    metric_results.append(MetricResult(metric.name, math.nan, None, None, False))
                    continue
                baseline = None if baseline_metrics is None else baseline_metrics.get(metric.name)
                gain = None if baseline is None else _relative_gain(value, baseline, metric.direction)
                passed = True
                if gain is not None and metric.max_regression_fraction is not None:
                    passed = gain >= -metric.max_regression_fraction
                metrics_ok = metrics_ok and passed
                metric_results.append(MetricResult(metric.name, value, baseline, gain, passed))

        metric_weight = 1.0 - self.spec.correctness_weight
        if metric_results:
            weighted_total = sum(m.weight for m in self.spec.metrics)
            if weighted_total > 0:
                by_name = {m.name: m for m in self.spec.metrics}
                utilities = []
                weights = []
                for result in metric_results:
                    spec = by_name[result.name]
                    gain = 0.0 if result.relative_gain is None else result.relative_gain
                    utility = max(0.0, min(1.0, 0.5 + 0.5 * gain))
                    utilities.append(utility)
                    weights.append(spec.weight)
                metric_score = sum(u * w for u, w in zip(utilities, weights)) / weighted_total
            else:
                metric_score = 0.5
        else:
            metric_score = 0.5
        score = self.spec.correctness_weight * correctness + metric_weight * metric_score
        passed = integrity_ok and correctness == 1.0 and metrics_ok
        diff = ""
        if (root / ".git").exists():
            diff = subprocess.run(
                ["git", "diff", "--no-ext-diff"], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, check=False,
            ).stdout
        payload_provenance = dict(provenance or {})
        payload_provenance.update({
            "task_spec_sha256": sha256_json(self.spec.to_dict()),
            "workspace": str(root),
        })
        return EnvironmentResult(
            task_id=self.spec.id,
            passed=passed,
            score=max(0.0, min(1.0, score)),
            correctness=correctness,
            tests=tests,
            metrics=metric_results,
            integrity_ok=integrity_ok,
            integrity_failures=integrity_failures,
            workspace_diff=diff,
            provenance=payload_provenance,
        )


def metric_map(result: EnvironmentResult) -> dict[str, float]:
    return {x.name: x.value for x in result.metrics if math.isfinite(x.value)}
