"""Run a candidate function outside the grader and outside its files.

The grader never imports candidate code. Each call starts a new container
with :class:`vare.container_isolation.DockerIsolationRunner` flags: no
network, a read-only root, dropped capabilities, and uid/gid 65534. The only
bind mount is a staged copy of the subject at ``/subject``. ``/grader`` is
not mounted. Arguments are JSON on the container stdin after import. Expected
values, hidden tests, and the clock stay in the parent.

A trusted worker sits between the parent and the container. The parent and
the worker exchange one nonce-framed result. The nonce is not sent into the
container, so container stdout cannot forge a frame. A zero exit code without
a single ``{"value": ...}`` line is a failure. The parent compares that value
with the expected value it kept locally. End of file after that line is not
extra output; only further bytes are a malformed result.

``task_kind`` comes from trusted task metadata. Output shape does not decide
whether a task is a performance task. The measured cost starts when the
container is created, so subject import is included, and ends when the value
line is read. The same interval is used for the baseline and the candidate.
Arguments are supplied only after import. Baseline and candidate calls in a
pair share one freshly built argument list and run in a randomized order.
The confidence interval is on the paired differences
``baseline_seconds - candidate_seconds``. ``paired_timing`` checks both
return values against a parent-owned expected value.
"""
from __future__ import annotations

from dataclasses import dataclass
import hmac
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import select
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from .container_isolation import DockerIsolationRunner, IsolationError, IsolationLimits

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MAX_ARGS = 64_000
_MAX_LINE = 100_000
# Two-sided 95% Student-t critical values, df = repeats - 1.
_T_975 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284264,
    4: 2.7764451051977987,
    5: 2.5705818366147395,
    6: 2.446911851144167,
    7: 2.364624251592784,
    8: 2.306004135204166,
    9: 2.2621571627409915,
    10: 2.2281388519649385,
    11: 2.200985160082949,
    12: 2.1788128296634177,
    13: 2.1603686564610127,
    14: 2.1447866879169273,
    15: 2.131449545559323,
    16: 2.1199052992210112,
    17: 2.1098155778333156,
    18: 2.100922040241607,
    19: 2.093024054408263,
    20: 2.0859634472658366,
    21: 2.079613844727683,
    22: 2.0738730679040147,
    23: 2.0686576104190406,
    24: 2.0638985616280203,
    25: 2.059538552753294,
    26: 2.055529438642871,
    27: 2.051830516480628,
    28: 2.048407141795244,
    29: 2.045229642132703,
}
# Runs inside the candidate container. stdout is the only result channel.
# Saved callables are bound before the subject import.
_CALLEE = r"""
import json
import os
import sys

_dumps = json.dumps
_loads = json.loads
_write = os.write
_read = sys.stdin.buffer.readline
header = _loads(_read())
sys.path.insert(0, "/subject")
module = __import__(header["module"])
function = getattr(module, header["function"])
_write(1, b"READY\n")
request = _loads(_read())
try:
    value = function(*request["args"], **request["kwargs"])
    payload = _dumps({"value": value}, sort_keys=True, separators=(",", ":"), allow_nan=False)
except BaseException:
    os._exit(2)
_write(1, payload.encode("utf-8") + b"\n")
os._exit(0)
"""


class CandidateHarnessError(ValueError):
    """The call specification itself is invalid."""


class _OutputLimit(Exception):
    """The child wrote more than the framed-result cap."""


@dataclass
class FunctionCall:
    """One parent-owned case. ``expected`` never enters the child."""

    case_id: str
    module: str
    function: str
    args: list[Any]
    expected: Any
    kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    failure: str | None
    wall_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "failure": self.failure,
            "wall_s": self.wall_s,
        }


@dataclass(frozen=True)
class CallReport:
    task_kind: str
    passed: bool
    cases: tuple[CaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_kind": self.task_kind,
            "passed": self.passed,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class TimingPair:
    order: str
    baseline_s: float | None
    candidate_s: float | None
    difference_s: float | None
    ratio: float | None
    values_match: bool
    failure: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "baseline_s": self.baseline_s,
            "candidate_s": self.candidate_s,
            "difference_s": self.difference_s,
            "ratio": self.ratio,
            "values_match": self.values_match,
            "failure": self.failure,
        }


@dataclass(frozen=True)
class TimingReport:
    """Parent clock and parent comparison.

    ``output_pass`` means both sides returned the caller-supplied expected
    value. It is not a speed gate.
    """

    task_kind: str
    output_pass: bool
    repeats: int
    seed: int
    pairs: tuple[TimingPair, ...]
    mean_difference_s: float | None
    ci95_low_difference_s: float | None
    ci95_high_difference_s: float | None
    t_crit: float | None
    df: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_kind": self.task_kind,
            "output_pass": self.output_pass,
            "repeats": self.repeats,
            "seed": self.seed,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "mean_difference_s": self.mean_difference_s,
            "ci95_low_difference_s": self.ci95_low_difference_s,
            "ci95_high_difference_s": self.ci95_high_difference_s,
            "t_crit": self.t_crit,
            "df": self.df,
        }


class _Lines:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.pending = bytearray()

    def readline(self, deadline: float) -> bytes:
        fd = self.stream.fileno()
        while b"\n" not in self.pending:
            if len(self.pending) > _MAX_LINE:
                raise _OutputLimit("candidate output exceeded the byte limit")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("candidate call timed out")
            ready, _, _ = select.select([fd], [], [], min(0.2, remaining))
            if not ready:
                continue
            block = os.read(fd, 65536)
            if not block:
                raise EOFError("candidate closed its output")
            self.pending.extend(block)
        line, _, rest = self.pending.partition(b"\n")
        if len(line) > _MAX_LINE:
            raise _OutputLimit("candidate output exceeded the byte limit")
        self.pending = bytearray(rest)
        return bytes(line)

    def has_pending(self) -> bool:
        """True only when bytes remain after the line already consumed.

        A closed pipe is readable. ``os.read`` then returns ``b""``. That EOF
        is not extra output: an honest child that writes one line and exits
        must not be scored ``malformed_result``.
        """
        if self.pending:
            return True
        fd = self.stream.fileno()
        ready, _, _ = select.select([fd], [], [], 0)
        if not ready:
            return False
        block = os.read(fd, 65536)
        if not block:
            return False
        self.pending.extend(block)
        return True


def student_t_975(df: int) -> float:
    """Two-sided 95% Student-t critical value for ``df`` between 1 and 29."""
    try:
        return _T_975[df]
    except KeyError as exc:
        raise CandidateHarnessError("repeats must be from 2 through 30") from exc


def paired_difference_ci(differences: Sequence[float]) -> tuple[float, float, float, float, int]:
    """Mean and Student-t 95% interval of paired differences."""
    count = len(differences)
    if count < 2:
        raise CandidateHarnessError("paired interval needs at least two differences")
    mean = sum(differences) / count
    variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
    half = student_t_975(count - 1) * math.sqrt(variance / count)
    return mean, mean - half, mean + half, student_t_975(count - 1), count - 1


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _check_name(value: str, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise CandidateHarnessError(f"{label} must be a single identifier")
    return value


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _limits(image: str, timeout_s: float) -> IsolationLimits:
    if (
        not isinstance(timeout_s, (int, float))
        or isinstance(timeout_s, bool)
        or not math.isfinite(float(timeout_s))
        or not 0 < float(timeout_s) <= 3600
    ):
        raise CandidateHarnessError("timeout must be positive and at most 3600 seconds")
    try:
        return IsolationLimits(image=image, timeout_s=float(timeout_s), max_output_bytes=_MAX_LINE)
    except (TypeError, ValueError) as exc:
        raise CandidateHarnessError("isolation image must be pinned by sha256 digest") from exc


def _stage_subject(subject: Path, root: Path) -> Path:
    if not subject.is_dir() or subject.is_symlink():
        raise CandidateHarnessError("subject must be a real directory")
    resolved = subject.resolve()
    grader = Path("/grader")
    if resolved == grader or grader in resolved.parents:
        raise CandidateHarnessError("subject must not be the grader tree")
    if (subject / "evaluator").exists():
        raise CandidateHarnessError("subject must not include an evaluator tree")
    destination = root / "subject"
    _copy_tree(subject, destination)
    for dirpath, _, filenames in os.walk(destination):
        os.chmod(dirpath, 0o755)
        for name in filenames:
            os.chmod(Path(dirpath) / name, 0o644)
    return destination


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(mode=0o755)
    for entry in sorted(source.iterdir(), key=lambda item: item.name):
        if entry.is_symlink():
            raise CandidateHarnessError("subject symlink is not copied")
        target = destination / entry.name
        if entry.is_dir():
            _copy_tree(entry, target)
            continue
        if not entry.is_file():
            raise CandidateHarnessError("subject contains an unsupported file")
        target.write_bytes(entry.read_bytes())


def _kill(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def _src_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1])
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not current else src + os.pathsep + current
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _same_token(got: bytes, nonce: str) -> bool:
    try:
        text = got.decode("utf-8")
    except UnicodeError:
        return False
    return len(text) == len(nonce) and hmac.compare_digest(text, nonce)


def _invoke(
    staged: Path,
    module: str,
    function: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    timeout_s: float,
    image: str,
    runner: DockerIsolationRunner,
) -> tuple[Any | None, float | None, str | None]:
    """Return ``(value, parent_wall_s, failure)``."""
    nonce = secrets.token_hex(32)
    name = "vare-call-" + secrets.token_hex(12)
    deadline = time.monotonic() + timeout_s
    spec = {
        "nonce": nonce,
        "container": name,
        "module": module,
        "function": function,
        "subject": str(staged),
        "image": image,
        "args": list(args),
        "kwargs": dict(kwargs),
        "timeout_s": timeout_s,
    }
    proc = subprocess.Popen(
        [sys.executable, "-B", "-m", "vare.isolated_candidate", "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=_src_env(),
    )
    assert proc.stdin is not None and proc.stdout is not None
    reader = _Lines(proc.stdout)
    try:
        proc.stdin.write((_canonical(spec) + "\n").encode("utf-8"))
        proc.stdin.flush()
        ready = reader.readline(deadline)
        startup_s = _ready_startup(ready, nonce)
        if startup_s is None:
            _value, failure = _parse_result(ready, nonce)
            return None, None, failure or "malformed_result"
        if reader.has_pending():
            return None, None, "malformed_result"
        call_started = time.perf_counter()
        proc.stdin.write(b"GO\n")
        proc.stdin.flush()
        result = reader.readline(deadline)
        elapsed = startup_s + (time.perf_counter() - call_started)
        parsed, failure = _parse_result(result, nonce)
        if failure is not None:
            return None, elapsed, failure
        return parsed, elapsed, None
    except _OutputLimit:
        return None, None, "malformed_result"
    except (TimeoutError, EOFError, BrokenPipeError):
        return None, None, "missing_result"
    finally:
        try:
            runner.force_remove_sync(name)
        except IsolationError:
            pass
        _kill(proc)


_STARTUP = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z")


def _ready_startup(line: bytes, nonce: str) -> float | None:
    """Parse ``VARE_READY <nonce> <seconds>`` from the trusted worker.

    The third field is the worker clock from container creation through
    subject import. It is not taken from the candidate.
    """
    parts = line.split(b" ")
    if len(parts) != 3 or parts[0] != b"VARE_READY" or not _same_token(parts[1], nonce):
        return None
    try:
        text = parts[2].decode("ascii")
    except UnicodeError:
        return None
    if _STARTUP.fullmatch(text) is None:
        return None
    value = float(text)
    if not math.isfinite(value):
        return None
    return value


def _parse_result(line: bytes, nonce: str) -> tuple[Any | None, str | None]:
    parts = line.split(b" ", 2)
    if len(parts) != 3 or parts[0] not in {b"VARE_RESULT", b"VARE_FAIL"}:
        return None, "malformed_result"
    if not _same_token(parts[1], nonce):
        return None, "malformed_result"
    if parts[0] == b"VARE_FAIL":
        reason = parts[2].decode("utf-8", "replace")
        if reason not in {"missing_result", "malformed_result"}:
            reason = "missing_result"
        return None, reason
    try:
        payload = json.loads(parts[2].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None, "malformed_result"
    if type(payload) is not dict or set(payload) != {"value"}:
        return None, "malformed_result"
    return payload["value"], None


def _container_command() -> list[str]:
    return ["python", "-u", "-I", "-B", "-c", _CALLEE]


def _serve(spec: dict[str, Any], parent: _Lines) -> int:
    nonce = str(spec["nonce"])
    timeout_s = float(spec["timeout_s"])
    deadline = time.monotonic() + timeout_s
    runner = DockerIsolationRunner()
    limits = _limits(str(spec["image"]), timeout_s)
    try:
        argv = runner.isolated_subject_argv(
            limits,
            name=str(spec["container"]),
            subject=Path(str(spec["subject"])),
            command=_container_command(),
        )
    except (IsolationError, CandidateHarnessError, OSError, ValueError):
        _emit_fail(nonce, "missing_result")
        return 0
    started = time.perf_counter()
    try:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=runner._client_env,
        )
    except OSError:
        _emit_fail(nonce, "missing_result")
        return 0
    assert child.stdin is not None and child.stdout is not None
    reader = _Lines(child.stdout)
    try:
        header = {"module": spec["module"], "function": spec["function"]}
        child.stdin.write((_canonical(header) + "\n").encode("utf-8"))
        child.stdin.flush()
        ready = reader.readline(deadline)
        startup_s = time.perf_counter() - started
        if ready != b"READY" or reader.has_pending():
            _emit_fail(nonce, "malformed_result")
            return 0
        _emit_ready(nonce, startup_s)
        go = parent.readline(deadline)
        if go != b"GO":
            _emit_fail(nonce, "missing_result")
            return 0
        request = {"args": spec["args"], "kwargs": spec["kwargs"]}
        child.stdin.write((_canonical(request) + "\n").encode("utf-8"))
        child.stdin.flush()
        raw = reader.readline(deadline)
        if reader.has_pending():
            _emit_fail(nonce, "malformed_result")
            return 0
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            _emit_fail(nonce, "malformed_result")
            return 0
        if type(payload) is not dict or set(payload) != {"value"}:
            _emit_fail(nonce, "malformed_result")
            return 0
        _emit(nonce, "VARE_RESULT", _canonical(payload))
        return 0
    except _OutputLimit:
        _emit_fail(nonce, "malformed_result")
        return 0
    except (TimeoutError, EOFError, BrokenPipeError):
        _emit_fail(nonce, "missing_result")
        return 0
    finally:
        try:
            runner.force_remove_sync(str(spec["container"]))
        except IsolationError:
            pass
        if child.poll() is None:
            child.kill()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass


def _emit_ready(nonce: str, startup_s: float) -> None:
    sys.stdout.write(f"VARE_READY {nonce} {startup_s:.9f}\n")
    sys.stdout.flush()


def _emit_fail(nonce: str, reason: str) -> None:
    _emit(nonce, "VARE_FAIL", reason)


def _emit(nonce: str, kind: str, body: str) -> None:
    sys.stdout.write(f"{kind} {nonce} {body}\n")
    sys.stdout.flush()


def _prepare_call(
    module: str, function: str, args: Sequence[Any], kwargs: Mapping[str, Any] | None,
) -> tuple[str, str, list[Any], dict[str, Any]]:
    checked_module = _check_name(module, "module")
    checked_function = _check_name(function, "function")
    try:
        encoded = _canonical({"args": list(args), "kwargs": dict(kwargs or {})})
    except (TypeError, ValueError) as exc:
        raise CandidateHarnessError("arguments must be JSON values") from exc
    if len(encoded.encode("utf-8")) > _MAX_ARGS:
        raise CandidateHarnessError("arguments exceed the input byte limit")
    copied = json.loads(encoded)
    return checked_module, checked_function, copied["args"], copied["kwargs"]


def _one_call(
    subject: Path,
    module: str,
    function: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    timeout_s: float,
    image: str,
    runner: DockerIsolationRunner,
) -> tuple[Any | None, float | None, str | None]:
    module, function, args, kwargs = _prepare_call(module, function, args, kwargs)
    _limits(image, timeout_s)
    source = Path(subject)
    if not (source / f"{module}.py").is_file():
        raise CandidateHarnessError("subject module is missing")
    with tempfile.TemporaryDirectory(prefix="vare-call-") as folder:
        staged = _stage_subject(source, Path(folder))
        return _invoke(staged, module, function, args, kwargs, timeout_s, image, runner)


def grade_calls(
    subject: Path,
    calls: Sequence[FunctionCall],
    *,
    task_kind: str,
    image: str,
    timeout_s: float = 30,
    runner: DockerIsolationRunner | None = None,
) -> CallReport:
    """Compare each child return value with the parent-owned expected value.

    ``task_kind`` must be ``correctness``. ``image`` is a digest-pinned
    container image. Exit status is not a pass. Call this from the trusted
    process that holds the hidden tests; the candidate container has no
    grader mount.
    """
    if task_kind != "correctness":
        raise CandidateHarnessError("task_kind must be correctness for grade_calls")
    if not calls:
        raise CandidateHarnessError("at least one call is required")
    active = runner or DockerIsolationRunner()
    results: list[CaseResult] = []
    for call in calls:
        value, wall, failure = _one_call(
            subject, call.module, call.function, call.args, call.kwargs or {},
            timeout_s, image, active,
        )
        if failure is None:
            try:
                matched = _canonical(value) == _canonical(call.expected)
            except (TypeError, ValueError):
                matched = False
                failure = "malformed_result"
            if failure is None and not matched:
                failure = "wrong_output"
            passed = failure is None
        else:
            passed = False
        results.append(CaseResult(call.case_id, passed, failure, wall))
    return CallReport(task_kind, all(item.passed for item in results), tuple(results))


def paired_timing(
    candidate_subject: Path,
    baseline_subject: Path,
    *,
    module: str,
    function: str,
    make_args: Callable[[int], tuple[Sequence[Any], Mapping[str, Any]]],
    expected: Any,
    repeats: int,
    seed: int,
    task_kind: str,
    image: str,
    timeout_s: float = 30,
    runner: DockerIsolationRunner | None = None,
) -> TimingReport:
    """Time baseline and candidate containers with the parent clock.

    ``task_kind`` must be ``performance``. ``expected`` is the parent-owned
    return value for every repetition. ``output_pass`` is true only when
    both containers return that value. Two equal wrong returns do not pass,
    and any failed repetition drops the confidence interval. Call
    ``grade_calls`` as well when correctness spans more than this one value.
    ``make_args`` runs in the parent once per repetition and its return value
    is serialized before either child starts, so a later child cannot reuse
    the earlier child's object identity. The speed threshold belongs to the
    caller. ``pairs[].baseline_s`` and ``pairs[].candidate_s`` include
    container start and subject import, then the call.
    """
    if task_kind != "performance":
        raise CandidateHarnessError("task_kind must be performance for paired_timing")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or not 2 <= repeats <= 30:
        raise CandidateHarnessError("repeats must be from 2 through 30")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise CandidateHarnessError("seed must be an integer")
    try:
        expected_text = _canonical(expected)
    except (TypeError, ValueError) as exc:
        raise CandidateHarnessError("expected value must be JSON") from exc
    _check_name(module, "module")
    _check_name(function, "function")
    _limits(image, timeout_s)
    active = runner or DockerIsolationRunner()
    order_rng = random.Random(seed)
    pairs: list[TimingPair] = []
    differences: list[float] = []
    output_pass = True
    for index in range(repeats):
        args, kwargs = make_args(index)
        order = ["baseline", "candidate"]
        if order_rng.randrange(2):
            order = ["candidate", "baseline"]
        measured: dict[str, tuple[Any | None, float | None, str | None]] = {}
        failure: str | None = None
        for role in order:
            root = baseline_subject if role == "baseline" else candidate_subject
            measured[role] = _one_call(
                root, module, function, args, kwargs, timeout_s, image, active,
            )
            if measured[role][2] is not None and failure is None:
                failure = measured[role][2]
        baseline = measured.get("baseline", (None, None, "missing_result"))
        candidate = measured.get("candidate", (None, None, "missing_result"))
        values_match = False
        if failure is None:
            try:
                values_match = (
                    _canonical(baseline[0]) == expected_text
                    and _canonical(candidate[0]) == expected_text
                )
            except (TypeError, ValueError):
                values_match = False
                failure = "malformed_result"
            if not values_match and failure is None:
                failure = "wrong_output"
        if failure is not None:
            output_pass = False
        difference = None
        ratio = None
        if baseline[1] is not None and candidate[1] is not None:
            difference = baseline[1] - candidate[1]
            if candidate[1] > 0 and baseline[1] > 0:
                ratio = baseline[1] / candidate[1]
            if failure is None:
                differences.append(difference)
        pairs.append(TimingPair(
            "-".join(order), baseline[1], candidate[1], difference, ratio, values_match, failure,
        ))
    mean: float | None = None
    low: float | None = None
    high: float | None = None
    crit: float | None = None
    df: int | None = None
    if output_pass and len(differences) == repeats:
        mean, low, high, crit, df = paired_difference_ci(differences)
    return TimingReport(
        task_kind, output_pass, repeats, seed, tuple(pairs), mean, low, high, crit, df,
    )


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if list(argv) != ["--worker"]:
        raise SystemExit("worker only")
    reader = _Lines(sys.stdin.buffer)
    try:
        raw = reader.readline(time.monotonic() + 30)
        spec = json.loads(raw)
    except (TimeoutError, EOFError, _OutputLimit, json.JSONDecodeError):
        return 1
    if type(spec) is not dict or "nonce" not in spec:
        return 1
    try:
        return _serve(spec, reader)
    except (CandidateHarnessError, IsolationError, OSError, ValueError, KeyError, TypeError):
        _emit_fail(str(spec["nonce"]), "missing_result")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
