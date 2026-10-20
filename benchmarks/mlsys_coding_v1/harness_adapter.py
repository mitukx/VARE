"""Thin adapter onto the shared child-process grading harness.

PR #85 (https://github.com/mitukx/VARE/pull/85) owns that harness. The draft
description checked while this adapter was written specifies separate
containers and a snapshot copy. It does not name a nonce-tagged function-call
interface, so this module proposes one and falls back to the temporary shim
in ``candidate_call.py`` until the symbol exists.

Proposed parent-side call
-------------------------
Import ``vare.separated_candidate_harness.grade_function`` and call it with
one dict:

- ``workspace``: candidate directory (string). The child may run only there.
- ``entrypoint``: ``"module:function"`` from the task, not from the child.
- ``task_type``: ``"correctness"`` or ``"performance"``. Must equal
  ``metadata.task_type`` in the trusted ``task.json``. The child and the
  candidate workspace are not a source for this field.
- ``trusted_task_path``: that ``task.json``. It must sit outside ``workspace``.
- ``input_generator``: ``(index: int) -> (args: tuple, kwargs: dict)``.
  Every call returns fresh objects. The harness calls it again for every
  baseline run and every candidate run.
- ``reference``: parent-owned implementation. Raising ``ValueError`` means
  the candidate must raise ``ValueError`` too.
- ``checker``: ``(actual, expected, positional, keywords, args, kwargs)``.
  Called in the parent after a value-returning call. It raises
  ``AssertionError`` when the decoded result is wrong.
- ``repeats``: number of correctness calls, indexes ``0 .. repeats-1``.
- ``timing_repeats``: performance samples. ``0`` on a correctness task.
  Indexes passed to the generator continue after ``repeats``.
- ``timeout_s``: per child read.
- ``nonce``: parent-generated hex string. The harness must send it with
  every child message and accept a result only when the child echoes it.

The harness runs the candidate in a separate child. The child speaks only
nonce-tagged JSON. The parent compares values, takes every clock reading
with ``time.monotonic`` bound in the parent before the child starts, ignores
the child's exit code, and ignores the child's stdout. For
``task_type="performance"`` it interleaves baseline and candidate: even
sample index times the reference first, odd index times the candidate first.
The input generator is called again for every baseline run and every
candidate run, and each timing index must be a new draw. Reusing one input
object, or one input value, across the seven samples is not this interface.

Return dict:

- ``passed`` (bool)
- ``error`` (str or null)
- ``baseline_samples_s`` / ``candidate_samples_s`` (parent seconds)
- ``nonce_enforced`` (bool, must be true on this path)

If the symbol is missing, this module uses ``candidate_call.CandidateSession``
and records ``harness_id="temporary_shim_candidate_call"``,
``replaced_by="vare.separated_candidate_harness.grade_function"``, and
``nonce_enforced=false``. That path is a stand-in, not a second harness.
"""

from __future__ import annotations

import importlib
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from candidate_call import HARNESS_ID, REPLACED_BY, CandidateCallError, CandidateSession

PROPOSED_SYMBOL = "vare.separated_candidate_harness.grade_function"
TASK_TYPES = ("correctness", "performance")
# Same parent-bound monotonic clock the shim uses for candidate samples.
_PARENT_CLOCK = time.monotonic

Checker = Callable[
    [Any, Any, list[Any], dict[str, Any], tuple[Any, ...], dict[str, Any]], None
]
InputGenerator = Callable[[int], tuple[tuple[Any, ...], dict[str, Any]]]


class HarnessAdapterError(RuntimeError):
    """The request or the trusted metadata cannot be graded."""


@dataclass
class HarnessRequest:
    """One parent-owned grade. See the module docstring for field meanings."""

    workspace: Path
    entrypoint: str
    task_type: str
    trusted_task_path: Path
    input_generator: InputGenerator
    reference: Callable[..., Any]
    checker: Checker
    repeats: int
    timing_repeats: int = 0
    timeout_s: float = 20.0
    nonce: str = ""


@dataclass
class HarnessReport:
    """Parent decision. ``passed`` is never taken from a child exit code."""

    harness_id: str
    replaced_by: str | None
    nonce_enforced: bool
    nonce: str
    task_type: str
    passed: bool
    error: str | None
    baseline_samples_s: list[float]
    candidate_samples_s: list[float]
    interleave: str


def resolve_proposed_harness() -> Callable[[dict[str, Any]], Any] | None:
    """Return #85's grade function, or None when that symbol is absent."""
    try:
        module = importlib.import_module("vare.separated_candidate_harness")
    except ImportError:
        return None
    function = getattr(module, "grade_function", None)
    if not callable(function):
        return None
    return function


def load_task_type(task_json: Path) -> str:
    """Read ``metadata.task_type`` from a trusted task file."""
    try:
        payload = json.loads(Path(task_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessAdapterError("trusted task.json is unreadable") from exc
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    task_type = metadata.get("task_type") if isinstance(metadata, dict) else None
    if task_type not in TASK_TYPES:
        raise HarnessAdapterError("trusted metadata.task_type is missing")
    return str(task_type)


def _validate(request: HarnessRequest) -> str:
    module, separator, function = request.entrypoint.partition(":")
    if separator != ":" or not module.isidentifier() or not function.isidentifier():
        raise HarnessAdapterError("entrypoint must be module:function")
    if request.task_type not in TASK_TYPES:
        raise HarnessAdapterError("task_type is not correctness or performance")
    if request.repeats < 1 or request.timing_repeats < 0:
        raise HarnessAdapterError("repeat counts are invalid")
    if request.task_type == "correctness" and request.timing_repeats != 0:
        raise HarnessAdapterError("a correctness task cannot take timing samples")
    if request.task_type == "performance" and request.timing_repeats < 1:
        raise HarnessAdapterError("a performance task needs timing samples")
    if request.timeout_s <= 0:
        raise HarnessAdapterError("timeout must be positive")
    workspace = Path(request.workspace).resolve()
    trusted = Path(request.trusted_task_path).resolve()
    if not trusted.is_file():
        raise HarnessAdapterError("trusted task.json is missing")
    try:
        trusted.relative_to(workspace)
    except ValueError:
        inside = False
    else:
        inside = True
    if inside or workspace == trusted:
        raise HarnessAdapterError("trusted task.json is inside the candidate workspace")
    trusted_type = load_task_type(trusted)
    if trusted_type != request.task_type:
        raise HarnessAdapterError(
            f"task_type {request.task_type} does not match trusted metadata {trusted_type}"
        )
    nonce = request.nonce or secrets.token_hex(16)
    if not nonce or any(character not in "0123456789abcdef" for character in nonce):
        raise HarnessAdapterError("nonce must be lowercase hex")
    return nonce


def _freeze(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    raise HarnessAdapterError("input value is not a supported type")


def _containers(value: Any):
    if isinstance(value, (list, dict)):
        yield value
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return
    for child in children:
        yield from _containers(child)


def guard_fresh_inputs(generator: InputGenerator) -> InputGenerator:
    """Reject a reused input object or a reused input value.

    The parent applies this before every baseline run and every candidate
    run. A generator that hands back the warmup object, or a new object with
    the same contents, fails the grade.
    """
    seen_ids: set[int] = set()
    seen_values: set[Any] = set()
    # Hold the containers. A bare id can be recycled after the previous
    # input is no longer referenced, which would look like reuse.
    held: list[Any] = []

    def wrapped(index: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args, kwargs = generator(index)
        for container in _containers((args, kwargs)):
            identity = id(container)
            if identity in seen_ids:
                raise HarnessAdapterError("input object was reused")
            seen_ids.add(identity)
            held.append(container)
        fingerprint = _freeze((args, kwargs))
        if fingerprint in seen_values:
            raise HarnessAdapterError("input value was reused")
        seen_values.add(fingerprint)
        return args, kwargs

    return wrapped


def _draw(
    request: HarnessRequest, index: int
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    drawn = request.input_generator(index)
    if (
        not isinstance(drawn, tuple)
        or len(drawn) != 2
        or not isinstance(drawn[0], tuple)
        or not isinstance(drawn[1], dict)
    ):
        raise HarnessAdapterError(
            "input_generator must return (args tuple, kwargs dict)"
        )
    return drawn


def _as_report(
    payload: Any, *, nonce: str, task_type: str, harness_id: str
) -> HarnessReport:
    if not isinstance(payload, dict):
        raise HarnessAdapterError("harness result was not an object")
    if payload.get("nonce_enforced") is not True:
        raise HarnessAdapterError("proposed harness did not enforce the nonce")
    if payload.get("passed") is not True and payload.get("passed") is not False:
        raise HarnessAdapterError("proposed harness omitted passed")
    baseline = payload.get("baseline_samples_s", [])
    candidate = payload.get("candidate_samples_s", [])
    if not isinstance(baseline, list) or not isinstance(candidate, list):
        raise HarnessAdapterError("proposed harness omitted timing samples")
    error = payload.get("error")
    if error is not None and not isinstance(error, str):
        raise HarnessAdapterError("proposed harness error was not a string")
    return HarnessReport(
        harness_id=harness_id,
        replaced_by=None,
        nonce_enforced=True,
        nonce=nonce,
        task_type=task_type,
        passed=bool(payload["passed"]),
        error=error,
        baseline_samples_s=[float(sample) for sample in baseline],
        candidate_samples_s=[float(sample) for sample in candidate],
        interleave="even_baseline_first",
    )


def _candidate_call(
    session: CandidateSession, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, list[Any], dict[str, Any], float] | str:
    """Return the parent-timed call, or an error string. Exit status is unused."""
    try:
        return session.call(*args, **kwargs)
    except CandidateCallError as exc:
        return f"{exc.error_type}: {exc.detail}"[:300]


def _run_shim(request: HarnessRequest, nonce: str) -> HarnessReport:
    """Parent checks and parent clocks. The shim only transports one call."""
    module, _, function = request.entrypoint.partition(":")
    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    error: str | None = None
    passed = True
    try:
        session = CandidateSession(
            Path(request.workspace), module, function, timeout_s=request.timeout_s
        )
    except CandidateCallError as exc:
        return HarnessReport(
            harness_id=HARNESS_ID,
            replaced_by=REPLACED_BY,
            nonce_enforced=False,
            nonce=nonce,
            task_type=request.task_type,
            passed=False,
            error=f"{exc.error_type}: {exc.detail}"[:300],
            baseline_samples_s=[],
            candidate_samples_s=[],
            interleave="even_baseline_first",
        )
    try:
        for index in range(request.repeats):
            args, kwargs = _draw(request, index)
            failure = _expect_reference(session, request, args, kwargs)
            if failure is not None:
                passed = False
                error = failure
                break
        if passed and request.timing_repeats:
            for sample_index in range(request.timing_repeats):
                failure, baseline_s, candidate_s = _time_pair(
                    session, request, sample_index
                )
                if failure is not None:
                    passed = False
                    error = failure
                    baseline_samples = []
                    candidate_samples = []
                    break
                baseline_samples.append(baseline_s)
                candidate_samples.append(candidate_s)
    except HarnessAdapterError:
        session.close()
        raise
    except Exception as exc:  # noqa: BLE001 - a checker failure is a failed grade
        passed = False
        error = f"{type(exc).__name__}: {exc}"[:300]
        baseline_samples = []
        candidate_samples = []
    finally:
        session.close()
    return HarnessReport(
        harness_id=HARNESS_ID,
        replaced_by=REPLACED_BY,
        nonce_enforced=False,
        nonce=nonce,
        task_type=request.task_type,
        passed=passed,
        error=error,
        baseline_samples_s=baseline_samples,
        candidate_samples_s=candidate_samples,
        interleave="even_baseline_first",
    )


def _expect_reference(
    session: CandidateSession,
    request: HarnessRequest,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    try:
        expected = request.reference(*args, **kwargs)
    except ValueError:
        outcome = _candidate_call(session, args, kwargs)
        if isinstance(outcome, str):
            if outcome.startswith("ValueError"):
                return None
            return outcome or "expected ValueError"
        return "invalid input returned a value"
    outcome = _candidate_call(session, args, kwargs)
    if isinstance(outcome, str):
        return outcome or "candidate call failed"
    actual, positional, keywords, _elapsed = outcome
    request.checker(actual, expected, positional, keywords, args, kwargs)
    return None


def _time_role(
    session: CandidateSession,
    request: HarnessRequest,
    draw_index: int,
    role: str,
) -> tuple[str | None, float]:
    args, kwargs = _draw(request, draw_index)
    if role == "baseline":
        started = _PARENT_CLOCK()
        request.reference(*args, **kwargs)
        return None, _PARENT_CLOCK() - started
    outcome = _candidate_call(session, args, kwargs)
    if isinstance(outcome, str):
        return outcome or "candidate call failed", 0.0
    actual, positional, keywords, elapsed = outcome
    expected = request.reference(*args, **kwargs)
    request.checker(actual, expected, positional, keywords, args, kwargs)
    return None, elapsed


def _time_pair(
    session: CandidateSession, request: HarnessRequest, sample_index: int
) -> tuple[str | None, float, float]:
    """Fresh inputs for each role. Even samples time the reference first."""
    order = (
        ("baseline", "candidate")
        if sample_index % 2 == 0
        else ("candidate", "baseline")
    )
    recorded: dict[str, float] = {}
    base = request.repeats + sample_index * 2
    for offset, role in enumerate(order):
        failure, elapsed = _time_role(session, request, base + offset, role)
        if failure is not None:
            return failure, 0.0, 0.0
        recorded[role] = elapsed
    return None, recorded["baseline"], recorded["candidate"]


def run_harness(request: HarnessRequest) -> HarnessReport:
    """Grade one candidate through #85's harness, or the marked temporary shim."""
    nonce = _validate(request)
    request.nonce = nonce
    request.input_generator = guard_fresh_inputs(request.input_generator)
    proposed = resolve_proposed_harness()
    if proposed is not None:
        payload = {
            "workspace": str(Path(request.workspace).resolve()),
            "entrypoint": request.entrypoint,
            "task_type": request.task_type,
            "trusted_task_path": str(Path(request.trusted_task_path).resolve()),
            "input_generator": request.input_generator,
            "reference": request.reference,
            "checker": request.checker,
            "repeats": request.repeats,
            "timing_repeats": request.timing_repeats,
            "timeout_s": request.timeout_s,
            "nonce": nonce,
        }
        return _as_report(
            proposed(payload),
            nonce=nonce,
            task_type=request.task_type,
            harness_id=PROPOSED_SYMBOL,
        )
    return _run_shim(request, nonce)
