"""Fail-closed Docker execution for repository graders.

This moves *executable* repository tests off the controller host and into a
separately administered container/VM. Candidate source is read-only in the
container and copied to private scratch before tests. It does not conceal
grader source files from hostile code running in the same grader container.
Confidential runs mount this worker only after the candidate container has
exited; see ``vare.separated_grading``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import secrets
import shutil
import sys
import tempfile
from typing import Any, Mapping

from .container_isolation import (
    DockerIsolationRunner, IsolationError, IsolationLimits, _mount_source,
)
from .environments.runner import (
    CommandResult, EnvironmentResult, ExecutableEvaluator, MetricResult,
)
from .environments.spec import EngineeringTaskSpec
from .evidence import sha256_json


def _check_finite(value: Any, *, fraction: bool = False) -> float:
    if type(value) not in (float, int) or not math.isfinite(float(value)):
        raise IsolationError("nonfinite or nonnumeric evaluator score")
    number = float(value)
    if fraction and not 0.0 <= number <= 1.0:
        raise IsolationError("evaluator score outside [0, 1]")
    return number


def _decode(raw_text: str, *, task_id: str, spec_digest: str,
            policy_id: str, policy_version: int) -> EnvironmentResult:
    try:
        raw = json.loads(raw_text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        required = {
            "task_id", "passed", "score", "correctness", "tests", "metrics",
            "integrity_ok", "integrity_failures", "workspace_diff", "provenance",
        }
        if type(raw) is not dict or set(raw) != required:
            raise ValueError("unexpected evaluator result schema")
        if (raw["task_id"] != task_id or type(raw["passed"]) is not bool
                or type(raw["integrity_ok"]) is not bool
                or type(raw["tests"]) is not list or type(raw["metrics"]) is not list
                or type(raw["integrity_failures"]) is not list
                or not all(type(item) is str for item in raw["integrity_failures"])
                or type(raw["workspace_diff"]) is not str
                or type(raw["provenance"]) is not dict):
            raise ValueError("invalid evaluator result fields")
        provenance = raw["provenance"]
        if (provenance.get("task_spec_sha256") != spec_digest
                or provenance.get("policy_id") != policy_id
                or provenance.get("policy_version") != policy_version):
            raise ValueError("evaluator provenance mismatch")
        tests = []
        for t in raw["tests"]:
            if (type(t) is not dict or set(t) != {
                "name", "argv", "returncode", "wall_s", "stdout", "stderr",
                "timed_out", "output_limited",
            } or type(t["name"]) is not str or type(t["argv"]) is not list
                    or not all(type(x) is str for x in t["argv"])
                    or type(t["returncode"]) is not int
                    or type(t["stdout"]) is not str or type(t["stderr"]) is not str
                    or type(t["timed_out"]) is not bool
                    or type(t["output_limited"]) is not bool):
                raise ValueError("invalid command result")
            _check_finite(t["wall_s"])
            tests.append(CommandResult(**{**t, "argv": tuple(t["argv"])}))
        metrics = []
        for metric in raw["metrics"]:
            if (type(metric) is not dict or set(metric) != {
                "name", "value", "baseline", "relative_gain", "passed_regression_gate",
            } or type(metric["name"]) is not str
                    or type(metric["passed_regression_gate"]) is not bool):
                raise ValueError("invalid metric result")
            _check_finite(metric["value"])
            for k in ("baseline", "relative_gain"):
                if metric[k] is not None:
                    _check_finite(metric[k])
            metrics.append(MetricResult(**metric))
        score = _check_finite(raw["score"], fraction=True)
        correctness = _check_finite(raw["correctness"], fraction=True)
        if raw["passed"] and (
            not raw["integrity_ok"] or correctness != 1.0
            or not tests or not all(test.passed for test in tests)
            or not all(metric.passed_regression_gate for metric in metrics)
        ):
            raise ValueError("internally inconsistent passing result")
        return EnvironmentResult(
            task_id=task_id, passed=raw["passed"], score=score,
            correctness=correctness, tests=tests, metrics=metrics,
            integrity_ok=raw["integrity_ok"],
            integrity_failures=raw["integrity_failures"],
            workspace_diff=raw["workspace_diff"],
            provenance=provenance,
        )
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise IsolationError("invalid or untrusted repository-evaluator output") from exc


class DockerRepositoryEvaluator:
    """Operator-controlled backend for CatalogEnvironmentVerifier.

    Candidate code shares this container namespace with grader files. Evaluation
    is disabled unless the operator opts in and the task is explicitly public.
    Do not pass task/candidate-controlled images, evaluator paths, or Docker
    credentials to this constructor. Confidential hidden tests use
    ``vare.separated_grading.SeparatedCandidateGrader``, which does not mount
    the grader directory into the candidate container.
    """

    def __init__(
        self, *,
        runner: DockerIsolationRunner,
        limits: IsolationLimits,
        allowed_workspace_root: Path,
        allow_public_grader_visible_execution: bool = False,
    ) -> None:
        if type(allow_public_grader_visible_execution) is not bool:
            raise ValueError("public grader-visible execution opt-in must be boolean")
        self.runner = runner
        self.limits = limits
        self.allowed_workspace_root = allowed_workspace_root
        self.allow_public_grader_visible_execution = allow_public_grader_visible_execution

    def _argv(self, *, spec_path: Path, workspace: Path, spec_digest: str,
              policy_id: str, policy_version: int,
              baseline_metrics: Mapping[str, float] | None, name: str) -> list[str]:
        root = _mount_source(spec_path.parent.resolve(), directory=True)
        spec = _mount_source(spec_path.resolve(), directory=False)
        candidate = self.runner._workspace(self.allowed_workspace_root, workspace)
        if root == candidate or root in candidate.parents or candidate in root.parents:
            raise IsolationError("evaluator and candidate directories overlap")
        if spec.parent != root:
            raise IsolationError("task spec must be directly inside trusted evaluator root")
        if not re.fullmatch(r"[0-9a-f]{64}", spec_digest):
            raise IsolationError("invalid trusted task spec digest")
        if not isinstance(policy_id, str) or not policy_id:
            raise IsolationError("missing policy identity")
        if type(policy_version) is not int or policy_version < 0:
            raise IsolationError("invalid policy version")
        baseline = dict(baseline_metrics or {})
        if any(type(k) is not str or type(v) not in (int, float)
               or not math.isfinite(float(v)) for k, v in baseline.items()):
            raise IsolationError("invalid metric baselines")
        baseline_json = json.dumps(baseline, sort_keys=True, allow_nan=False)
        if len(baseline_json) > 8192:
            raise IsolationError("baseline metric specification too large")
        return [
            *self.runner._common(self.limits, name),
            # Hardening assumption: no candidate-controlled Docker socket,
            # credentials, image, grader mount or controller-side secret.
            "--mount", f"type=bind,source={candidate},target=/candidate,readonly",
            "--mount", f"type=bind,source={root},target=/grader,readonly",
            "--tmpfs", "/scratch:rw,nosuid,nodev,size=256m",
            "--workdir=/",
            self.limits.image, "python", "-m", "vare.sandboxed_repository",
            "--worker", "--spec", f"/grader/{spec.name}",
            "--candidate", "/candidate",
            "--scratch", "/scratch",
            "--expected-spec-sha256", spec_digest,
            "--policy-id", policy_id,
            "--policy-version", str(policy_version),
            "--baseline-json", baseline_json,
        ]

    async def evaluate(
        self, *, spec_path: Path, workspace: Path,
        spec_digest: str, policy_id: str, policy_version: int,
        baseline_metrics: Mapping[str, float] | None = None,
    ) -> EnvironmentResult:
        if not self.allow_public_grader_visible_execution:
            raise IsolationError(
                "candidate-visible repository grading requires explicit public-task opt-in"
            )
        spec = EngineeringTaskSpec.load(spec_path)
        if spec.metadata.get("grader_visibility") != "public":
            raise IsolationError(
                "candidate-visible repository grading requires task metadata grader_visibility=public"
            )
        name = "vare-repo-" + secrets.token_hex(12)
        argv = self._argv(
            spec_path=spec_path, workspace=workspace,
            spec_digest=spec_digest, policy_id=policy_id,
            policy_version=policy_version,
            baseline_metrics=baseline_metrics, name=name,
        )
        result = await self.runner.execute(argv, self.limits)
        if result.returncode != 0:
            raise IsolationError("isolated repository evaluator failed closed")
        # A malformed result must NEVER be converted to a positive score.
        return _decode(result.stdout, task_id=spec.id,
                       spec_digest=spec_digest, policy_id=policy_id,
                       policy_version=policy_version)


def _worker(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec)
    candidate = Path(args.candidate)
    scratch = Path(args.scratch)
    if (not spec_path.is_file() or spec_path.is_symlink()
            or not candidate.is_dir() or candidate.is_symlink()
            or not scratch.is_dir() or scratch.is_symlink()
            or scratch == candidate or scratch == spec_path.parent):
        raise RuntimeError("invalid evaluator mount layout")
    spec = EngineeringTaskSpec.load(spec_path)
    if sha256_json(spec.to_dict()) != args.expected_spec_sha256:
        raise RuntimeError("trusted task spec was modified")
    baseline = json.loads(args.baseline_json)
    if type(baseline) is not dict or any(
        type(key) is not str or type(val) not in (float, int)
        or not math.isfinite(float(val)) for key, val in baseline.items()
    ):
        raise RuntimeError("invalid baseline metrics")
    # Never execute test commands directly on host or bind-mounted candidate.
    # Preserve symlinks as links; do not dereference candidate-supplied links
    # while copying. Tests execute under container restrictions, not host.
    with tempfile.TemporaryDirectory(prefix="vare-eval-", dir=scratch) as folder:
        private = Path(folder) / "workspace"
        shutil.copytree(candidate, private, symlinks=True)
        report = ExecutableEvaluator(
            spec, evaluator_root=spec_path.parent,
        ).evaluate(
            private,
            baseline_metrics=baseline,
            provenance={
                "policy_id": args.policy_id,
                "policy_version": args.policy_version,
                "task_spec_path": str(spec_path),
            },
        )
        if report.task_id != spec.id:
            raise RuntimeError("task identity mismatch")
        if not 0 <= report.score <= 1 or not 0 <= report.correctness <= 1:
            raise RuntimeError("invalid score from evaluator")
        # Workspace diffs, test diagnostics and stdout are bounded by the
        # enclosing Docker supervisor; oversized output fails the entire eval.
        sys.stdout.write(json.dumps(report.to_dict(), sort_keys=True, allow_nan=False) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated executable repository evaluator")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--policy-version", type=int, required=True)
    parser.add_argument("--baseline-json", default="{}")
    args = parser.parse_args(argv)
    if not args.worker:
        raise RuntimeError("worker must run under an operator-controlled container/VM")
    return _worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
