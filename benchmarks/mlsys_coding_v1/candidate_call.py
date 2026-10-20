"""TEMPORARY SHIM. Replace with ``vare.separated_candidate_harness.grade_function``.

PR #85 owns the shared child-process harness. This module is not that
harness: it does not check a nonce, it does not interleave baseline and
candidate runs, and it does not decide pass or fail. ``harness_adapter.py``
is the only supported caller. Delete this file when that import succeeds.

The child speaks length-prefixed JSON on private pipes. Its exit code and
stdout are discarded. The parent clock around one call is returned to the
adapter, which owns the comparison.
"""

from __future__ import annotations

import json
import math
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Self

# Stable id recorded on every shim grade so a result can say which code ran.
HARNESS_ID = "temporary_shim_candidate_call"
REPLACED_BY = "vare.separated_candidate_harness.grade_function"

MAX_MESSAGE_BYTES = 8_000_000
_HEADER = 8
# Bound at import, in the parent, before any candidate process starts.
# A candidate that replaces time.perf_counter or time.monotonic cannot
# replace this function object in the parent interpreter.
_PARENT_CLOCK = time.monotonic


class CandidateCallError(Exception):
    """The child did not produce a parent-accepted result."""

    def __init__(self, error_type: str, detail: str = "") -> None:
        self.error_type = error_type
        self.detail = detail
        super().__init__(detail or error_type)


def _encode(value: Any) -> dict[str, Any]:
    if value is None:
        return {"t": "none"}
    if type(value) is bool:
        return {"t": "bool", "v": value}
    if type(value) is int:
        return {"t": "int", "v": value}
    if type(value) is float:
        if math.isnan(value):
            return {"t": "float", "v": "nan"}
        if value == math.inf:
            return {"t": "float", "v": "inf"}
        if value == -math.inf:
            return {"t": "float", "v": "-inf"}
        return {"t": "float", "v": value}
    if type(value) is str:
        return {"t": "str", "v": value}
    if type(value) is list:
        return {"t": "list", "v": [_encode(item) for item in value]}
    if type(value) is tuple:
        return {"t": "tuple", "v": [_encode(item) for item in value]}
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("dict keys must be strings")
        return {"t": "dict", "v": {key: _encode(item) for key, item in value.items()}}
    raise ValueError(f"unsupported value type {type(value).__name__}")


def _decode(value: Any) -> Any:
    if type(value) is not dict or "t" not in value or set(value) - {"t", "v"}:
        raise ValueError("malformed value")
    kind = value["t"]
    payload = value.get("v")
    if kind == "none" and set(value) == {"t"}:
        return None
    if kind == "bool" and type(payload) is bool:
        return payload
    if kind == "int" and type(payload) is int:
        return payload
    if kind == "float" and payload == "nan":
        return math.nan
    if kind == "float" and payload == "inf":
        return math.inf
    if kind == "float" and payload == "-inf":
        return -math.inf
    if kind == "float" and type(payload) is float and math.isfinite(payload):
        return payload
    if kind == "str" and type(payload) is str:
        return payload
    if kind == "list" and type(payload) is list:
        return [_decode(item) for item in payload]
    if kind == "tuple" and type(payload) is list:
        return tuple(_decode(item) for item in payload)
    if (
        kind == "dict"
        and type(payload) is dict
        and all(type(key) is str for key in payload)
    ):
        return {key: _decode(item) for key, item in payload.items()}
    raise ValueError("malformed value")


def _dumps(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _loads(raw: bytes) -> Any:
    payload = json.loads(raw.decode("utf-8"))
    if type(payload) is not dict:
        raise ValueError("message must be an object")
    return payload


def _frame(payload: dict[str, Any]) -> bytes:
    body = _dumps(payload)
    if len(body) > MAX_MESSAGE_BYTES:
        raise ValueError("message is too large")
    return len(body).to_bytes(_HEADER, "big") + body


class CandidateSession:
    """One child process for every call of one candidate function."""

    def __init__(
        self, workspace: Path, module: str, function: str, *, timeout_s: float = 20.0
    ) -> None:
        if not module.isidentifier() or not function.isidentifier():
            raise ValueError("module and function names must be identifiers")
        if timeout_s <= 0:
            raise ValueError("timeout must be positive")
        self.timeout_s = timeout_s
        request_read, request_write = os.pipe()
        result_read, result_write = os.pipe()
        worker = Path(__file__).resolve()
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(worker),
                "worker",
                str(request_read),
                str(result_write),
                str(Path(workspace).resolve()),
                module,
                function,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(request_read, result_write),
            close_fds=True,
            start_new_session=True,
            cwd=Path(workspace).resolve(),
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        os.close(request_read)
        os.close(result_write)
        self._request = request_write
        self._result = result_read
        os.set_blocking(self._result, False)
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._result, selectors.EVENT_READ)
        try:
            ready = self._read_message()
        except CandidateCallError:
            self.close()
            raise
        if ready != {"ready": True}:
            self.close()
            raise CandidateCallError("protocol", "child did not become ready")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        process = getattr(self, "_process", None)
        self._process = None
        for fd_name in ("_request", "_result"):
            fd = getattr(self, fd_name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, fd_name, None)
        selector = getattr(self, "_selector", None)
        if selector is not None:
            selector.close()
            self._selector = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def call(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Any, list[Any], dict[str, Any], float]:
        """Return value, positional args after the call, keyword args, and parent seconds."""
        started = _PARENT_CLOCK()
        self._write_message(
            {"args": _encode(list(args)), "kwargs": _encode(dict(kwargs))}
        )
        message = self._read_message()
        elapsed = _PARENT_CLOCK() - started
        if type(message) is not dict or type(message.get("ok")) is not bool:
            raise CandidateCallError(
                "protocol", "child result was not a recognized object"
            )
        if message["ok"] is False:
            error_type = message.get("error_type")
            if type(error_type) is not str or not error_type.isidentifier():
                raise CandidateCallError("protocol", "child error type was not usable")
            detail = message.get("error")
            if type(detail) is not str:
                detail = ""
            raise CandidateCallError(error_type, detail[:300])
        try:
            value = _decode(message.get("value"))
            positional = _decode(message.get("args"))
            keywords = _decode(message.get("kwargs"))
        except (ValueError, TypeError) as exc:
            raise CandidateCallError(
                "protocol", "child value could not be decoded"
            ) from exc
        if type(positional) is not list or type(keywords) is not dict:
            raise CandidateCallError("protocol", "child did not return its arguments")
        return value, positional, keywords, elapsed

    def require_value_error(self, *args: Any, **kwargs: Any) -> None:
        try:
            self.call(*args, **kwargs)
        except CandidateCallError as exc:
            if exc.error_type == "ValueError":
                return
            raise AssertionError(f"expected ValueError, got {exc.error_type}") from None
        raise AssertionError("invalid input must fail")

    def _write_message(self, payload: dict[str, Any]) -> None:
        if self._request is None:
            raise CandidateCallError("protocol", "candidate process is closed")
        frame = _frame(payload)
        view = memoryview(frame)
        while view:
            try:
                written = os.write(self._request, view)
            except BrokenPipeError as exc:
                raise CandidateCallError(
                    "protocol", "candidate process closed the request pipe"
                ) from exc
            view = view[written:]

    def _read_exact(self, count: int) -> bytes:
        if self._result is None or self._selector is None:
            raise CandidateCallError("protocol", "candidate process is closed")
        chunks = []
        remaining = count
        deadline = time.monotonic() + self.timeout_s
        while remaining:
            leftover = deadline - time.monotonic()
            if leftover <= 0:
                raise CandidateCallError("protocol", "candidate process timed out")
            events = self._selector.select(leftover)
            if not events:
                raise CandidateCallError("protocol", "candidate process timed out")
            try:
                chunk = os.read(self._result, remaining)
            except BlockingIOError:
                continue
            if not chunk:
                raise CandidateCallError(
                    "protocol", "candidate process closed the result pipe"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_message(self) -> dict[str, Any]:
        header = self._read_exact(_HEADER)
        length = int.from_bytes(header, "big")
        if length > MAX_MESSAGE_BYTES:
            raise CandidateCallError("protocol", "candidate message is too large")
        try:
            return _loads(self._read_exact(length))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CandidateCallError(
                "protocol", "candidate message was not JSON"
            ) from exc


def _read_exact_blocking(fd: int, count: int) -> bytes | None:
    chunks = []
    remaining = count
    while remaining:
        chunk = os.read(fd, remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _worker(argv: list[str]) -> None:
    request_fd = int(argv[0])
    result_fd = int(argv[1])
    workspace, module_name, function_name = argv[2], argv[3], argv[4]
    write = os.write
    read_exact = _read_exact_blocking
    encode = _encode
    decode = _decode
    dumps = _dumps
    loads = _loads
    importer = __import__

    def emit(payload: dict[str, Any]) -> None:
        frame = dumps(payload)
        if len(frame) > MAX_MESSAGE_BYTES:
            raise ValueError("message is too large")
        message = len(frame).to_bytes(_HEADER, "big") + frame
        view = memoryview(message)
        while view:
            written = write(result_fd, view)
            view = view[written:]

    sys.argv = ["candidate"]
    sys.path.insert(0, workspace)
    module = importer(module_name)
    function = getattr(module, function_name)
    emit({"ready": True})
    while True:
        header = read_exact(request_fd, _HEADER)
        if header is None:
            return
        length = int.from_bytes(header, "big")
        if length > MAX_MESSAGE_BYTES:
            return
        body = read_exact(request_fd, length)
        if body is None:
            return
        request = loads(body)
        args = decode(request["args"])
        kwargs = decode(request["kwargs"])
        try:
            value = function(*args, **kwargs)
        except SystemExit as exc:
            emit(
                {"ok": False, "error_type": "SystemExit", "error": str(exc.code)[:300]}
            )
            continue
        except Exception as exc:  # noqa: BLE001 - the parent decides whether this error is expected
            emit(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:300]}
            )
            continue
        emit(
            {
                "ok": True,
                "value": encode(value),
                "args": encode(list(args)),
                "kwargs": encode(dict(kwargs)),
            }
        )


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        try:
            _worker(sys.argv[2:])
        except Exception:  # noqa: BLE001 - a broken child must not look like a pass
            raise SystemExit(1) from None
