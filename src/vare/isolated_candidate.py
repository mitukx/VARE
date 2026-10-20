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
Arguments are supplied only after import. Each repetition builds its own
arguments and expected value in the parent, from a seed the candidate never
sees. Baseline, candidate, and an empty-subject control share that case and
run in a randomized order. The primary confidence interval is on the paired
differences ``baseline_seconds - candidate_seconds``, which include container
start and subject import. An overhead-adjusted ratio subtracts the empty
control and is a secondary description; import stays in that cost.
Numeric answers use ``math.isclose`` with the task's ``rtol`` and ``atol``.
``two_x_speedup_decision`` is the 2x gate. It refuses when either unadjusted
mean is below twice the empty-control mean.
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
# The MLSys repeated-range workload sends bounded 5k-value/5k-query inputs.
_MAX_ARGS = 512_000
_MAX_LINE = 512_000
# Allow the isolated Docker worker to launch before candidate import timing.
_STARTUP_GRACE_S = 10.0
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
_protocol_fd = 1
if header.get("discard_candidate_stdout") is True:
    _protocol_fd = os.dup(2)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 1)
    os.dup2(_devnull, 2)
_write(_protocol_fd, b"IMPORTING\n")
module = __import__(header["module"])
function = getattr(module, header["function"])
_write(_protocol_fd, b"READY\n")
_exit = os._exit
_Exception = Exception
_BaseException = BaseException
_encode = str.encode
def _one(request):
    args = request["args"]
    for path in request.get("tuple_paths", []):
        index = path[0]
        if path[1] == "*":
            args[index] = [tuple(row) for row in args[index]]
    try:
        value = function(*args, **request["kwargs"])
        return {"value": value}
    except _Exception as exc:
        return {"exception": type(exc).__name__}
    except _BaseException:
        _exit(2)
if header.get("session") is True:
    while True:
        request = _loads(_read())
        if request == {"close": True}:
            break
        payload = _dumps(_one(request), sort_keys=True, separators=(",", ":"), allow_nan=False)
        _write(_protocol_fd, _encode(payload, "utf-8") + b"\n")
else:
    request = _loads(_read())
    payload = _dumps(_one(request), sort_keys=True, separators=(",", ":"), allow_nan=False)
    _write(_protocol_fd, _encode(payload, "utf-8") + b"\n")
_exit(0)
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
    expected_exception: str | None = None
    expected_exception_prefix: str | None = None
    tuple_paths: tuple[tuple[Any, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateException:
    """Exception type caught by the trusted isolated candidate worker."""

    type_name: str


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
    overhead_s: float | None = None
    adjusted_baseline_s: float | None = None
    adjusted_candidate_s: float | None = None
    adjusted_difference_s: float | None = None
    adjusted_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "baseline_s": self.baseline_s,
            "candidate_s": self.candidate_s,
            "difference_s": self.difference_s,
            "ratio": self.ratio,
            "values_match": self.values_match,
            "failure": self.failure,
            "overhead_s": self.overhead_s,
            "adjusted_baseline_s": self.adjusted_baseline_s,
            "adjusted_candidate_s": self.adjusted_candidate_s,
            "adjusted_difference_s": self.adjusted_difference_s,
            "adjusted_ratio": self.adjusted_ratio,
        }


@dataclass(frozen=True)
class TimingReport:
    """Parent clock and parent comparison.

    ``output_pass`` means both sides returned that repetition's expected
    value. It is not a speed gate. The unadjusted difference is the primary
    timing result. Adjusted fields subtract an empty-subject control and are
    secondary.
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
    mean_adjusted_difference_s: float | None = None
    ci95_low_adjusted_difference_s: float | None = None
    ci95_high_adjusted_difference_s: float | None = None
    mean_adjusted_ratio: float | None = None
    ci95_low_adjusted_ratio: float | None = None
    ci95_high_adjusted_ratio: float | None = None
    workload_sufficient: bool = False
    oracle_subject_ratio: float | None = None

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
            "mean_adjusted_difference_s": self.mean_adjusted_difference_s,
            "ci95_low_adjusted_difference_s": self.ci95_low_adjusted_difference_s,
            "ci95_high_adjusted_difference_s": self.ci95_high_adjusted_difference_s,
            "mean_adjusted_ratio": self.mean_adjusted_ratio,
            "ci95_low_adjusted_ratio": self.ci95_low_adjusted_ratio,
            "ci95_high_adjusted_ratio": self.ci95_high_adjusted_ratio,
            "workload_sufficient": self.workload_sufficient,
            "oracle_subject_ratio": self.oracle_subject_ratio,
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


def _tolerance(rtol: float, atol: float) -> tuple[float, float]:
    for label, value in (("rtol", rtol), ("atol", atol)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise CandidateHarnessError(f"{label} must be a non-negative finite number")
    return float(rtol), float(atol)


def values_close(
    actual: Any, expected: Any, *, rtol: float, atol: float,
    bool_as_number: bool = False,
) -> bool:
    """Compare JSON values. Numbers use ``math.isclose``; other values are exact."""
    if (type(actual) is bool or type(expected) is bool) and not bool_as_number:
        return type(actual) is type(expected) and actual == expected
    if type(actual) is bool:
        actual = int(actual)
    if type(expected) is bool:
        expected = int(expected)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        actual_f = float(actual)
        expected_f = float(expected)
        if not math.isfinite(actual_f) or not math.isfinite(expected_f):
            return False
        return math.isclose(actual_f, expected_f, rel_tol=rtol, abs_tol=atol)
    if type(actual) is str or type(expected) is str:
        return type(actual) is str and type(expected) is str and actual == expected
    if type(actual) is list and type(expected) is list:
        return len(actual) == len(expected) and all(
            values_close(left, right, rtol=rtol, atol=atol, bool_as_number=bool_as_number)
            for left, right in zip(actual, expected, strict=True)
        )
    if type(actual) is dict and type(expected) is dict:
        if set(actual) != set(expected):
            return False
        return all(
            values_close(
                actual[key], expected[key], rtol=rtol, atol=atol,
                bool_as_number=bool_as_number,
            ) for key in actual
        )
    return False


def workload_covers_startup(
    baseline_s: Sequence[float], candidate_s: Sequence[float], overhead_s: Sequence[float],
) -> bool:
    """True when both unadjusted means are at least twice the empty-control mean."""
    count = len(baseline_s)
    if count == 0 or len(candidate_s) != count or len(overhead_s) != count:
        return False
    if any(sample <= 0 for sample in overhead_s):
        return False
    floor = 2.0 * (sum(overhead_s) / count)
    return (sum(baseline_s) / count) >= floor and (sum(candidate_s) / count) >= floor


def oracle_subject_ratio_of(
    baseline_s: Sequence[float], candidate_s: Sequence[float],
) -> float | None:
    """Mean oracle wall divided by mean subject wall. ``None`` if a subject sample is not positive."""
    if not candidate_s or len(baseline_s) != len(candidate_s) or any(sample <= 0 for sample in candidate_s):
        return None
    return (sum(baseline_s) / len(baseline_s)) / (sum(candidate_s) / len(candidate_s))


def two_x_speedup_decision(report: TimingReport, *, threshold: float = 2.0) -> dict[str, Any]:
    """Apply a 2x rule only when the empty-control precondition holds.

    The ratio is recomputed from the parent samples. A stored
    ``workload_sufficient`` flag is not enough: samples below twice the
    empty-control mean fail with ``workload_below_empty_control``.
    """
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) <= 0
    ):
        raise CandidateHarnessError("speedup threshold must be a positive finite number")
    baselines: list[float] = []
    candidates: list[float] = []
    overheads: list[float] = []
    for pair in report.pairs:
        if pair.baseline_s is None or pair.candidate_s is None or pair.overhead_s is None:
            return {
                "success": False,
                "workload_sufficient": False,
                "oracle_subject_ratio": None,
                "reason": "missing_sample",
            }
        baselines.append(pair.baseline_s)
        candidates.append(pair.candidate_s)
        overheads.append(pair.overhead_s)
    ratio = oracle_subject_ratio_of(baselines, candidates)
    if not report.output_pass:
        return {
            "success": False,
            "workload_sufficient": False,
            "oracle_subject_ratio": ratio,
            "reason": "output_mismatch",
        }
    sufficient = workload_covers_startup(baselines, candidates, overheads)
    if not sufficient:
        return {
            "success": False,
            "workload_sufficient": False,
            "oracle_subject_ratio": ratio,
            "reason": "workload_below_empty_control",
        }
    if ratio is None or ratio < float(threshold):
        return {
            "success": False,
            "workload_sufficient": True,
            "oracle_subject_ratio": ratio,
            "reason": "below_threshold",
        }
    return {
        "success": True,
        "workload_sufficient": True,
        "oracle_subject_ratio": ratio,
        "reason": "performance",
    }


def overhead_adjusted(
    baseline_s: float, candidate_s: float, overhead_s: float,
) -> tuple[float, float, float, float | None]:
    """Subtract an empty-subject control from both walls.

    A control subtracted from both sides cancels in the difference and remains
    in the ratio. Import work that the empty subject does not do stays in the
    adjusted cost. The ratio is ``None`` when either adjusted time is not
    positive. This is a secondary description, not a promotion gate.
    """
    adjusted_baseline = baseline_s - overhead_s
    adjusted_candidate = candidate_s - overhead_s
    difference = adjusted_baseline - adjusted_candidate
    ratio = None
    if adjusted_baseline > 0 and adjusted_candidate > 0:
        ratio = adjusted_baseline / adjusted_candidate
    return adjusted_baseline, adjusted_candidate, difference, ratio


def paired_difference_ci(differences: Sequence[float]) -> tuple[float, float, float, float, int]:
    """Mean and Student-t 95% interval of paired samples."""
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


def _limits(image: str, timeout_s: float, memory_mb: int = 512) -> IsolationLimits:
    if (
        not isinstance(timeout_s, (int, float))
        or isinstance(timeout_s, bool)
        or not math.isfinite(float(timeout_s))
        or not 0 < float(timeout_s) <= 3600
    ):
        raise CandidateHarnessError("timeout must be positive and at most 3600 seconds")
    try:
        return IsolationLimits(
            image=image, timeout_s=float(timeout_s), memory_mb=memory_mb,
            max_output_bytes=_MAX_LINE,
        )
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
    tuple_paths: Sequence[Sequence[Any]] = (),
    memory_mb: int = 512,
    discard_candidate_stdout: bool = False,
    task_deadline: float | None = None,
) -> tuple[Any | None, float | None, str | None]:
    """Return ``(value, parent_wall_s, failure)``."""
    nonce = secrets.token_hex(32)
    name = "vare-call-" + secrets.token_hex(12)
    if task_deadline is not None and time.monotonic() >= task_deadline:
        return None, None, "missing_result"
    deadline = time.monotonic() + timeout_s + _STARTUP_GRACE_S
    if task_deadline is not None:
        deadline = min(deadline, task_deadline)
    spec = {
        "nonce": nonce,
        "container": name,
        "module": module,
        "function": function,
        "subject": str(staged),
        "image": image,
        "args": list(args),
        "kwargs": dict(kwargs),
        "tuple_paths": [list(path) for path in tuple_paths],
        "timeout_s": timeout_s,
        "memory_mb": memory_mb,
        "discard_candidate_stdout": discard_candidate_stdout,
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
        importing = ready == f"VARE_IMPORTING {nonce} 1".encode("ascii")
        if not importing:
            _value, failure = _parse_result(ready, nonce)
            return None, None, failure or "malformed_result"
        ready = reader.readline(min(deadline, time.monotonic() + timeout_s))
        startup_s = _ready_startup(ready, nonce)
        if startup_s is None:
            _value, failure = _parse_result(ready, nonce)
            return None, None, failure or "malformed_result"
        if reader.has_pending():
            return None, None, "malformed_result"
        call_started = time.perf_counter()
        proc.stdin.write(b"GO\n")
        proc.stdin.flush()
        result = reader.readline(min(deadline, time.monotonic() + timeout_s))
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


def _invoke_session(
    staged: Path,
    module: str,
    function: str,
    calls: Sequence[tuple[Sequence[Any], Mapping[str, Any], Sequence[Sequence[Any]]]],
    timeout_s: float,
    image: str,
    runner: DockerIsolationRunner,
    memory_mb: int = 512,
    discard_candidate_stdout: bool = False,
    task_deadline: float | None = None,
) -> list[tuple[Any | None, float | None, str | None]]:
    """Run an ordered case group in one isolated candidate process."""
    if not calls:
        return []
    if task_deadline is not None and time.monotonic() >= task_deadline:
        return [(None, None, "missing_result")] * len(calls)
    nonce = secrets.token_hex(32)
    name = "vare-call-" + secrets.token_hex(12)
    deadline = time.monotonic() + timeout_s + _STARTUP_GRACE_S
    if task_deadline is not None:
        deadline = min(deadline, task_deadline)
    spec = {
        "nonce": nonce, "container": name, "module": module,
        "function": function, "subject": str(staged), "image": image,
        "timeout_s": timeout_s, "session": True, "call_count": len(calls),
        "memory_mb": memory_mb,
        "discard_candidate_stdout": discard_candidate_stdout,
    }
    proc = subprocess.Popen(
        [sys.executable, "-B", "-m", "vare.isolated_candidate", "--worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        start_new_session=True, env=_src_env(),
    )
    assert proc.stdin is not None and proc.stdout is not None
    reader = _Lines(proc.stdout)
    results: list[tuple[Any | None, float | None, str | None]] = []
    try:
        proc.stdin.write((_canonical(spec) + "\n").encode("utf-8"))
        proc.stdin.flush()
        ready = reader.readline(
            min(deadline, time.monotonic() + timeout_s + _STARTUP_GRACE_S)
        )
        if ready != f"VARE_IMPORTING {nonce} 1".encode("ascii"):
            return [(None, None, "malformed_result")] * len(calls)
        ready = reader.readline(min(deadline, time.monotonic() + timeout_s))
        startup_s = _ready_startup(ready, nonce)
        if startup_s is None or reader.has_pending():
            return [(None, None, "malformed_result")] * len(calls)
        proc.stdin.write(b"GO\n")
        proc.stdin.flush()
        for args, kwargs, tuple_paths in calls:
            if task_deadline is not None and time.monotonic() >= task_deadline:
                results.extend([(None, None, "missing_result")] * (len(calls) - len(results)))
                break
            request = {
                "args": list(args), "kwargs": dict(kwargs),
                "tuple_paths": [list(path) for path in tuple_paths],
            }
            started = time.perf_counter()
            proc.stdin.write((_canonical(request) + "\n").encode("utf-8"))
            proc.stdin.flush()
            line = reader.readline(min(deadline, time.monotonic() + timeout_s))
            elapsed = time.perf_counter() - started
            value, failure = _parse_result(line, nonce)
            results.append((value, elapsed, failure))
            if failure is not None:
                results.extend([(None, None, "missing_result")] * (len(calls) - len(results)))
                break
        return results
    except (_OutputLimit, TimeoutError, EOFError, BrokenPipeError):
        results.extend([(None, None, "missing_result")] * (len(calls) - len(results)))
        return results
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
    if type(payload) is dict and set(payload) == {"exception"}:
        exception = payload["exception"]
        if type(exception) is str and exception.isidentifier():
            return CandidateException(exception), None
        return None, "malformed_result"
    if type(payload) is not dict or set(payload) != {"value"}:
        return None, "malformed_result"
    return payload["value"], None


def _container_command() -> list[str]:
    return ["python", "-u", "-I", "-B", "-c", _CALLEE]


def _serve(spec: dict[str, Any], parent: _Lines) -> int:
    nonce = str(spec["nonce"])
    timeout_s = float(spec["timeout_s"])
    deadline = time.monotonic() + timeout_s + _STARTUP_GRACE_S
    runner = DockerIsolationRunner()
    limits = _limits(str(spec["image"]), timeout_s, int(spec.get("memory_mb", 512)))
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
    discard_candidate_stdout = spec.get("discard_candidate_stdout") is True
    try:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL if discard_candidate_stdout else subprocess.PIPE,
            stderr=subprocess.PIPE if discard_candidate_stdout else subprocess.DEVNULL,
            env=runner._client_env,
        )
    except OSError:
        _emit_fail(nonce, "missing_result")
        return 0
    protocol_stream = child.stderr if discard_candidate_stdout else child.stdout
    assert child.stdin is not None and protocol_stream is not None
    reader = _Lines(protocol_stream)
    try:
        header = {
            "module": spec["module"], "function": spec["function"],
            "session": spec.get("session") is True,
            "discard_candidate_stdout": spec.get("discard_candidate_stdout") is True,
        }
        child.stdin.write((_canonical(header) + "\n").encode("utf-8"))
        child.stdin.flush()
        importing = reader.readline(deadline)
        if importing != b"IMPORTING":
            _emit_fail(nonce, "malformed_result")
            return 0
        _emit_importing(nonce)
        ready = reader.readline(time.monotonic() + timeout_s)
        startup_s = time.perf_counter() - started
        if ready != b"READY" or reader.has_pending():
            _emit_fail(nonce, "malformed_result")
            return 0
        _emit_ready(nonce, startup_s)
        go = parent.readline(time.monotonic() + _STARTUP_GRACE_S)
        if go != b"GO":
            _emit_fail(nonce, "missing_result")
            return 0
        if spec.get("session") is True:
            count = int(spec.get("call_count", 0))
            if count <= 0 or count > 100_000:
                _emit_fail(nonce, "malformed_result")
                return 0
            for _index in range(count):
                raw_request = parent.readline(time.monotonic() + _STARTUP_GRACE_S)
                try:
                    request = json.loads(raw_request.decode("utf-8"))
                    child.stdin.write((_canonical(request) + "\n").encode("utf-8"))
                    child.stdin.flush()
                    raw = reader.readline(time.monotonic() + timeout_s)
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError, TimeoutError, EOFError):
                    _emit_fail(nonce, "missing_result")
                    return 0
                valid_value = type(payload) is dict and set(payload) == {"value"}
                valid_exception = (
                    type(payload) is dict and set(payload) == {"exception"}
                    and type(payload["exception"]) is str
                    and payload["exception"].isidentifier()
                )
                if not valid_value and not valid_exception or reader.has_pending():
                    _emit_fail(nonce, "malformed_result")
                    return 0
                _emit(nonce, "VARE_RESULT", _canonical(payload))
            child.stdin.write(b'{"close":true}\n')
            child.stdin.flush()
            return 0
        request = {
            "args": spec["args"], "kwargs": spec["kwargs"],
            "tuple_paths": spec.get("tuple_paths", []),
        }
        child.stdin.write((_canonical(request) + "\n").encode("utf-8"))
        child.stdin.flush()
        raw = reader.readline(time.monotonic() + timeout_s)
        if reader.has_pending():
            _emit_fail(nonce, "malformed_result")
            return 0
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            _emit_fail(nonce, "malformed_result")
            return 0
        valid_value = type(payload) is dict and set(payload) == {"value"}
        valid_exception = (
            type(payload) is dict and set(payload) == {"exception"}
            and type(payload["exception"]) is str
            and payload["exception"].isidentifier()
        )
        if not valid_value and not valid_exception:
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


def _emit_importing(nonce: str) -> None:
    sys.stdout.write(f"VARE_IMPORTING {nonce} 1\n")
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
    tuple_paths: Sequence[Sequence[Any]] = (),
    memory_mb: int = 512,
    discard_candidate_stdout: bool = False,
    task_deadline: float | None = None,
) -> tuple[Any | None, float | None, str | None]:
    module, function, args, kwargs = _prepare_call(module, function, args, kwargs)
    _limits(image, timeout_s, memory_mb)
    source = Path(subject)
    if not (source / f"{module}.py").is_file():
        raise CandidateHarnessError("subject module is missing")
    with tempfile.TemporaryDirectory(prefix="vare-call-") as folder:
        staged = _stage_subject(source, Path(folder))
        return _invoke(
            staged, module, function, args, kwargs, timeout_s, image, runner,
            tuple_paths, memory_mb, discard_candidate_stdout, task_deadline,
        )


def grade_calls(
    subject: Path,
    calls: Sequence[FunctionCall],
    *,
    task_kind: str,
    image: str,
    timeout_s: float = 30,
    rtol: float = 0.0,
    atol: float = 0.0,
    runner: DockerIsolationRunner | None = None,
    preserve_session: bool = False,
    bool_as_number: bool = False,
    memory_mb: int = 512,
    discard_candidate_stdout: bool = False,
    task_deadline: float | None = None,
) -> CallReport:
    """Compare each child return value with the parent-owned expected value.

    ``task_kind`` must be ``correctness``. ``image`` is a digest-pinned
    container image. ``rtol`` and ``atol`` are the task's numeric tolerances
    for ``math.isclose``. Exit status is not a pass. Call this from the
    trusted process that holds the hidden tests; the candidate container has
    no grader mount.
    """
    if task_kind != "correctness":
        raise CandidateHarnessError("task_kind must be correctness for grade_calls")
    if not calls:
        raise CandidateHarnessError("at least one call is required")
    _limits(image, timeout_s, memory_mb)
    rel, abs_tol = _tolerance(rtol, atol)
    active = runner or DockerIsolationRunner()
    results: list[CaseResult] = []
    if preserve_session:
        first = calls[0]
        if any(call.module != first.module or call.function != first.function for call in calls):
            raise CandidateHarnessError("session calls must use one module and function")
        prepared = [
            _prepare_call(call.module, call.function, call.args, call.kwargs or {})
            for call in calls
        ]
        with tempfile.TemporaryDirectory(prefix="vare-call-session-") as folder:
            staged = _stage_subject(Path(subject), Path(folder))
            invoked = _invoke_session(
                staged, first.module, first.function,
                [(item[2], item[3], call.tuple_paths) for item, call in zip(prepared, calls, strict=True)],
                timeout_s, image, active, memory_mb, discard_candidate_stdout, task_deadline,
            )
    else:
        invoked = [
            _one_call(
                subject, call.module, call.function, call.args, call.kwargs or {},
                timeout_s, image, active, call.tuple_paths,
                memory_mb, discard_candidate_stdout, task_deadline,
            )
            for call in calls
        ]
    for call, (value, wall, failure) in zip(calls, invoked, strict=True):
        if failure is None:
            if call.expected_exception is not None:
                matched = (
                    isinstance(value, CandidateException)
                    and (
                        value.type_name.startswith(call.expected_exception_prefix)
                        if call.expected_exception_prefix is not None
                        else value.type_name == call.expected_exception
                    )
                )
            else:
                matched = values_close(
                    value, call.expected, rtol=rel, atol=abs_tol,
                    bool_as_number=bool_as_number,
                )
            if not matched:
                failure = "wrong_output"
            passed = failure is None
        else:
            passed = False
        results.append(CaseResult(call.case_id, passed, failure, wall))
    return CallReport(task_kind, all(item.passed for item in results), tuple(results))


def _case_for_repetition(
    make_case: Callable[[int], tuple[Sequence[Any], Mapping[str, Any], Any]],
    index: int,
) -> tuple[Sequence[Any], Mapping[str, Any], Any]:
    produced = make_case(index)
    if type(produced) is not tuple or len(produced) != 3:
        raise CandidateHarnessError("make_case must return args, kwargs, and expected")
    args, kwargs, expected = produced
    if (
        isinstance(args, (str, bytes))
        or not isinstance(args, Sequence)
        or isinstance(kwargs, (str, bytes))
        or not isinstance(kwargs, Mapping)
    ):
        raise CandidateHarnessError("make_case args and kwargs must be a sequence and a mapping")
    try:
        expected_value = _json_copy(expected)
    except (TypeError, ValueError) as exc:
        raise CandidateHarnessError("expected value must be JSON") from exc
    return args, kwargs, expected_value


def _overhead_subject(root: Path, module: str, function: str) -> Path:
    destination = root / "overhead"
    destination.mkdir()
    (destination / f"{module}.py").write_text(
        f"def {function}(*_args, **_kwargs):\n    return None\n",
        encoding="utf-8",
    )
    return destination


def paired_timing(
    candidate_subject: Path,
    baseline_subject: Path,
    *,
    module: str,
    function: str,
    make_case: Callable[[int], tuple[Sequence[Any], Mapping[str, Any], Any]],
    repeats: int,
    seed: int,
    task_kind: str,
    image: str,
    timeout_s: float = 30,
    rtol: float = 0.0,
    atol: float = 0.0,
    runner: DockerIsolationRunner | None = None,
    tuple_paths: Sequence[Sequence[Any]] = (),
) -> TimingReport:
    """Time baseline and candidate containers with the parent clock.

    ``task_kind`` must be ``performance``. ``make_case(index)`` runs in the
    parent and returns ``(args, kwargs, expected)`` for that repetition.
    Draw the inputs there from a seed the candidate never receives, and use a
    different expected value when the function's result depends on the input.
    One fixed expected value lets a candidate hardcode that answer.
    ``output_pass`` is true only when both containers return that
    repetition's expected value. Two equal wrong returns do not pass, and any
    failed repetition drops the confidence intervals. Call ``grade_calls`` as
    well when correctness spans more than these repetitions.

    ``pairs[].baseline_s`` and ``pairs[].candidate_s`` include container
    start, subject import, and the call. ``pairs[].overhead_s`` is an empty
    subject with the same arguments, run in the same shuffled order. Import
    of the real subject is not in that control, so it remains in the
    adjusted cost. ``adjusted_ratio`` removes the control from both walls.
    A shared control cancels in the difference, which is why the adjusted
    difference matches the unadjusted difference. Container start compresses
    unadjusted ratios toward 1. Before applying a 2x rule to those ratios,
    use ``two_x_speedup_decision``. It refuses unless both unadjusted means
    are at least twice the mean empty-control time. The adjusted ratio is
    secondary. Ignore it when an adjusted time is not positive, and do not
    promote on it alone. ``rtol`` and ``atol`` are the task numeric tolerances.
    """
    if task_kind != "performance":
        raise CandidateHarnessError("task_kind must be performance for paired_timing")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or not 2 <= repeats <= 30:
        raise CandidateHarnessError("repeats must be from 2 through 30")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise CandidateHarnessError("seed must be an integer")
    checked_module = _check_name(module, "module")
    checked_function = _check_name(function, "function")
    rel, abs_tol = _tolerance(rtol, atol)
    _limits(image, timeout_s)
    cases = [_case_for_repetition(make_case, index) for index in range(repeats)]
    active = runner or DockerIsolationRunner()
    order_rng = random.Random(seed)
    roles = ("baseline", "candidate", "overhead")
    pairs: list[TimingPair] = []
    differences: list[float] = []
    adjusted_differences: list[float] = []
    adjusted_ratios: list[float] = []
    output_pass = True
    with tempfile.TemporaryDirectory(prefix="vare-overhead-") as folder:
        overhead_root = _overhead_subject(Path(folder), checked_module, checked_function)
        for args, kwargs, expected in cases:
            order = list(roles)
            order_rng.shuffle(order)
            measured: dict[str, tuple[Any | None, float | None, str | None]] = {}
            failure: str | None = None
            for role in order:
                if role == "baseline":
                    root = baseline_subject
                elif role == "candidate":
                    root = candidate_subject
                else:
                    root = overhead_root
                measured[role] = _one_call(
                    root, checked_module, checked_function, args, kwargs,
                    timeout_s, image, active, tuple_paths,
                )
                if measured[role][2] is not None and failure is None:
                    failure = measured[role][2]
            baseline = measured.get("baseline", (None, None, "missing_result"))
            candidate = measured.get("candidate", (None, None, "missing_result"))
            overhead = measured.get("overhead", (None, None, "missing_result"))
            values_match = False
            if baseline[2] is None and candidate[2] is None:
                values_match = (
                    values_close(baseline[0], expected, rtol=rel, atol=abs_tol)
                    and values_close(candidate[0], expected, rtol=rel, atol=abs_tol)
                )
                if not values_match and failure is None:
                    failure = "wrong_output"
            if failure is not None:
                output_pass = False
            difference = None
            ratio = None
            adjusted: tuple[float, float, float, float | None] | None = None
            if baseline[1] is not None and candidate[1] is not None:
                difference = baseline[1] - candidate[1]
                if candidate[1] > 0 and baseline[1] > 0:
                    ratio = baseline[1] / candidate[1]
                if failure is None:
                    differences.append(difference)
            if (
                baseline[1] is not None
                and candidate[1] is not None
                and overhead[1] is not None
            ):
                adjusted = overhead_adjusted(baseline[1], candidate[1], overhead[1])
                if failure is None:
                    adjusted_differences.append(adjusted[2])
                    if adjusted[3] is not None:
                        adjusted_ratios.append(adjusted[3])
            pairs.append(TimingPair(
                "-".join(order),
                baseline[1],
                candidate[1],
                difference,
                ratio,
                values_match,
                failure,
                overhead[1],
                None if adjusted is None else adjusted[0],
                None if adjusted is None else adjusted[1],
                None if adjusted is None else adjusted[2],
                None if adjusted is None else adjusted[3],
            ))
    mean: float | None = None
    low: float | None = None
    high: float | None = None
    crit: float | None = None
    df: int | None = None
    mean_adjusted: float | None = None
    low_adjusted: float | None = None
    high_adjusted: float | None = None
    mean_ratio: float | None = None
    low_ratio: float | None = None
    high_ratio: float | None = None
    if output_pass and len(differences) == repeats:
        mean, low, high, crit, df = paired_difference_ci(differences)
        if len(adjusted_differences) == repeats:
            mean_adjusted, low_adjusted, high_adjusted, _crit, _df = paired_difference_ci(
                adjusted_differences,
            )
        if len(adjusted_ratios) == repeats:
            mean_ratio, low_ratio, high_ratio, _crit, _df = paired_difference_ci(adjusted_ratios)
    sample_baselines = [pair.baseline_s for pair in pairs]
    sample_candidates = [pair.candidate_s for pair in pairs]
    sample_overheads = [pair.overhead_s for pair in pairs]
    sufficient = False
    subject_ratio: float | None = None
    if all(
        isinstance(sample, float)
        for sample in (*sample_baselines, *sample_candidates, *sample_overheads)
    ):
        baselines_f = [float(sample) for sample in sample_baselines if isinstance(sample, float)]
        candidates_f = [float(sample) for sample in sample_candidates if isinstance(sample, float)]
        overheads_f = [float(sample) for sample in sample_overheads if isinstance(sample, float)]
        subject_ratio = oracle_subject_ratio_of(baselines_f, candidates_f)
        sufficient = output_pass and workload_covers_startup(baselines_f, candidates_f, overheads_f)
    return TimingReport(
        task_kind, output_pass, repeats, seed, tuple(pairs), mean, low, high, crit, df,
        mean_adjusted, low_adjusted, high_adjusted, mean_ratio, low_ratio, high_ratio,
        sufficient, subject_ratio,
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
