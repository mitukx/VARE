"""Operator-side JSON scoring for candidates that run without grader mounts.

This deliberately supports a narrow request/response task format. The trusted
case set and expected outputs stay in the caller; each candidate request runs
in a fresh container with only a read-only candidate workspace mounted.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Protocol, Sequence

from .container_isolation import IsolationError, IsolationLimits, IsolationResult
from .evidence import sha256_json

_MAX_CASE_JSON_BYTES = 63_999
_MAX_CASES = 256
_DEFAULT_TOTAL_TIMEOUT_S = 600.0


def _encode(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("JSON evaluation values must be finite JSON data") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON response contains duplicate object keys")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class JSONIOCase:
    """Trusted request and oracle result; only ``request`` reaches candidate."""

    case_id: str
    request: Any
    expected: Any

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id or len(self.case_id) > 128:
            raise ValueError("case_id must be a bounded nonempty string")
        if max(len(_encode(self.request)), len(_encode(self.expected))) > _MAX_CASE_JSON_BYTES:
            raise ValueError("request and expected output must be at most 64 kB each")


@dataclass(frozen=True, slots=True)
class JSONIOCaseResult:
    case_id: str
    passed: bool
    returncode: int | None
    wall_s: float
    response_sha256: str | None
    failure: str | None
    cleanup_confirmed: bool


@dataclass(frozen=True, slots=True)
class JSONIOEvaluationReport:
    task_id: str
    policy_id: str
    policy_version: int
    case_set_sha256: str
    passed: bool
    score: float
    evaluation_complete: bool
    cleanup_confirmed: bool
    total_wall_s: float
    cases: tuple[JSONIOCaseResult, ...]


class CandidateIORunner(Protocol):
    """Anything that can run one bounded candidate request."""

    async def run_candidate_io(
        self, limits: IsolationLimits, *, allowed_workspace_root: Path,
        workspace: Path, command: Sequence[str], stdin_data: bytes,
    ) -> IsolationResult:
        """Run one request and return the isolated process result."""


class TrustedJSONScorer:
    """Compare untrusted candidate outputs against operator-held JSON oracles.

    Candidate receives one request per fresh container and never receives its
    expected value, the full case set, a grader path, or previous case results.
    The trusted caller supplies a fixed operator-owned entrypoint command.
    """

    def __init__(
        self, *, runner: CandidateIORunner, limits: IsolationLimits,
        allowed_workspace_root: Path,
    ) -> None:
        self.runner = runner
        self.limits = limits
        self.allowed_workspace_root = allowed_workspace_root

    async def evaluate(
        self, *, task_id: str, policy_id: str, policy_version: int,
        workspace: Path, command: Sequence[str], cases: Sequence[JSONIOCase],
        total_timeout_s: float = _DEFAULT_TOTAL_TIMEOUT_S,
    ) -> JSONIOEvaluationReport:
        if not task_id or not policy_id:
            raise ValueError("task and policy identities are required")
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("policy_version must be a nonnegative integer")
        if not command or not all(type(arg) is str and arg for arg in command):
            raise ValueError("candidate command must be a nonempty fixed argv")
        if (not isinstance(total_timeout_s, (float, int)) or isinstance(total_timeout_s, bool)
                or not 0 < total_timeout_s <= 3600):
            raise ValueError("total evaluation timeout must be in (0, 3600] seconds")
        if not cases or len(cases) > _MAX_CASES:
            raise ValueError(f"case set must contain between 1 and {_MAX_CASES} cases")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("case IDs must be unique")

        case_set_digest = sha256_json([
            {"case_id": case.case_id, "request": case.request, "expected": case.expected}
            for case in cases
        ])
        results: list[JSONIOCaseResult] = []
        started_all = time.perf_counter()
        evaluation_complete = True
        cleanup_confirmed = True
        for case_index, case in enumerate(cases):
            request = _encode(case.request) + b"\n"
            started = time.perf_counter()
            remaining = total_timeout_s - (started - started_all)
            if remaining <= 0:
                evaluation_complete = False
                results.extend(JSONIOCaseResult(
                    case_id=skipped.case_id, passed=False, returncode=None,
                    wall_s=0.0, response_sha256=None,
                    failure="total_evaluation_budget_exhausted",
                    cleanup_confirmed=True,
                ) for skipped in cases[case_index:])
                break
            case_limits = replace(
                self.limits, timeout_s=min(float(self.limits.timeout_s), remaining),
            )
            try:
                run = await self.runner.run_candidate_io(
                    case_limits,
                    allowed_workspace_root=self.allowed_workspace_root,
                    workspace=workspace,
                    command=command,
                    stdin_data=request,
                )
            except IsolationError as exc:
                message = str(exc)
                cleanup_ok = "could not confirm Docker container cleanup" not in message
                cleanup_confirmed = cleanup_confirmed and cleanup_ok
                evaluation_complete = False
                results.append(JSONIOCaseResult(
                    case_id=case.case_id, passed=False, returncode=None,
                    wall_s=time.perf_counter() - started, response_sha256=None,
                    failure=message[:256], cleanup_confirmed=cleanup_ok,
                ))
                results.extend(JSONIOCaseResult(
                    case_id=skipped.case_id, passed=False, returncode=None,
                    wall_s=0.0, response_sha256=None,
                    failure="not_run_after_isolation_error", cleanup_confirmed=True,
                ) for skipped in cases[case_index + 1:])
                break
            elapsed = time.perf_counter() - started
            response_sha256 = run.stdout_sha256 or hashlib.sha256(
                run.stdout.encode("utf-8")
            ).hexdigest()
            failure: str | None = None
            passed = False
            if run.returncode != 0:
                failure = "candidate_nonzero_exit"
            else:
                try:
                    actual = json.loads(
                        run.stdout,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
                        object_pairs_hook=_unique_object,
                    )
                    passed = _encode(actual) == _encode(case.expected)
                    if not passed:
                        failure = "wrong_json_response"
                except (json.JSONDecodeError, ValueError, TypeError):
                    failure = "invalid_json_response"
            results.append(JSONIOCaseResult(
                case_id=case.case_id,
                passed=passed,
                returncode=run.returncode,
                wall_s=elapsed,
                response_sha256=response_sha256,
                failure=failure,
                cleanup_confirmed=True,
            ))

        successes = sum(result.passed for result in results)
        score = successes / len(cases)
        return JSONIOEvaluationReport(
            task_id=task_id,
            policy_id=policy_id,
            policy_version=policy_version,
            case_set_sha256=case_set_digest,
            passed=evaluation_complete and cleanup_confirmed and successes == len(cases),
            score=score,
            evaluation_complete=evaluation_complete,
            cleanup_confirmed=cleanup_confirmed,
            total_wall_s=time.perf_counter() - started_all,
            cases=tuple(results),
        )
