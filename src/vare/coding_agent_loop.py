"""Bounded coding-agent loop for the MLSys task pack.

The historical local runner stopped when a model emitted a tool call whose
patch was not a unified diff, then counted the following prose as a missed
tool call. This loop keeps that transcript shape, adds exact-span edits and
public tests, and can commit each turn to ``TrajectoryStore`` so a later
process resumes from the last committed workspace.

Hidden grader files are not mounted into the candidate process. The existing
task script stays on the host and calls the candidate through a jailed
interpreter. That is a process and mount boundary, not a claim that a
candidate can be prevented from escaping a user namespace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
from typing import Any, Callable

from .evidence import canonical_json, sha256_json
from .trajectory_store import AgentStep, PolicyCheckpoint, TrajectoryStore
from .types import Task


TOOL_NAMES = (
    "list_files",
    "read_file",
    "search_text",
    "replace_text",
    "write_file",
    "apply_patch",
    "check_syntax",
    "run_public_tests",
    "finish",
)
_MAX_FILE_CHARS = 20_000
_MAX_WRITE_CHARS = 65_536
_PATCH_HINT = (
    "apply_patch requires a unified diff with --- a/<path> and +++ b/<path> "
    "headers for an allowlisted file. To replace one exact span, call "
    "replace_text with path, old, and new. To replace a whole allowlisted "
    "file, call write_file."
)


class InjectedFault(RuntimeError):
    """Raised when a test or worker asks the loop to stop at a commit boundary."""


@dataclass
class Generation:
    text: str
    new_tokens: int


Generate = Callable[[str, int], Generation]


@dataclass
class EpisodeResult:
    stop_reason: str
    messages: list[dict[str, str]]
    transcript: list[dict[str, Any]]
    new_tokens: int
    tool_executions: int
    committed_steps: int
    elapsed_seconds: float
    source_hashes_before: dict[str, str]
    source_hashes_after: dict[str, str]
    diff: str
    attempts: int = 1
    lost_tool_executions: int = 0
    resumed_from_step: int = 0


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tagged tool calls, Qwen XML function calls, and one bare tool JSON object.

    Returns a list of ``{"name", "arguments"}`` or ``{"error", "raw"}``.
    A response with no tool markup yields an empty list.
    """
    calls: list[dict[str, Any]] = []
    cursor = 0
    marker = "<tool_call>"
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            break
        end = text.find("</tool_call>", start)
        if end < 0:
            calls.append({"error": "unterminated tool_call", "raw": text[start:start + 500]})
            break
        body = text[start + len(marker):end].strip()
        cursor = end + len("</tool_call>")
        if body.startswith("{"):
            try:
                payload, _offset = json.JSONDecoder().raw_decode(body)
            except json.JSONDecodeError as exc:
                calls.append({"error": f"invalid tool JSON: {exc}", "raw": body[:500]})
                continue
            calls.extend(_calls_from_json(payload))
            continue
        if "<function=" in body:
            parsed = _calls_from_xml(body)
            calls.extend(parsed or [{"error": "unparsed function block", "raw": body[:500]}])
            continue
        calls.append({"error": "tool_call was not JSON or a function block", "raw": body[:500]})
    if calls or "<tool_call>" in text or "<function=" in text:
        return calls
    bare = _bare_json_call(text)
    return bare or calls


def _bare_json_call(text: str) -> list[dict[str, Any]]:
    """Accept one raw tool JSON object when the model omits ``<tool_call>`` tags.

    Trailing prose is rejected so a JSON example inside an explanation is not executed.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].startswith("```"):
            stripped = "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    if start < 0:
        return []
    try:
        payload, offset = json.JSONDecoder().raw_decode(stripped[start:])
    except json.JSONDecodeError:
        return []
    if stripped[start + offset:].strip():
        return []
    return _calls_from_json(payload)


def _calls_from_json(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return [{"error": "tool JSON must be an object", "raw": str(payload)[:500]}]
    name = payload.get("name")
    arguments = payload.get("arguments", payload.get("parameters", {}))
    if not isinstance(name, str) or name not in TOOL_NAMES:
        return [{"error": f"unsupported tool {name!r}", "raw": json.dumps(payload)[:500]}]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return [{"error": "tool arguments were not a JSON object", "raw": arguments[:500]}]
    if not isinstance(arguments, dict):
        return [{"error": "tool arguments must be an object", "raw": str(arguments)[:500]}]
    return [{"name": name, "arguments": arguments}]


def _calls_from_xml(body: str) -> list[dict[str, Any]]:
    calls = []
    cursor = 0
    token = "<function="
    while True:
        start = body.find(token, cursor)
        if start < 0:
            break
        name_end = body.find(">", start)
        if name_end < 0:
            break
        name = body[start + len(token):name_end].strip()
        close = body.find("</function>", name_end)
        if close < 0:
            break
        block = body[name_end + 1:close]
        cursor = close + len("</function>")
        arguments: dict[str, str] = {}
        position = 0
        while True:
            param = block.find("<parameter=", position)
            if param < 0:
                break
            key_end = block.find(">", param)
            value_end = block.find("</parameter>", key_end)
            if key_end < 0 or value_end < 0:
                break
            key = block[param + len("<parameter="):key_end].strip()
            arguments[key] = block[key_end + 1:value_end].strip("\n")
            position = value_end + len("</parameter>")
        if name not in TOOL_NAMES:
            calls.append({"error": f"unsupported tool {name!r}", "raw": block[:500]})
        else:
            calls.append({"name": name, "arguments": arguments})
    return calls


def system_prompt(task_prompt: str) -> str:
    return textwrap.dedent(
        """
        You are an offline code repair agent. The workspace already contains the task files.
        Hidden tests and the grader are not available. test_public.py is only a smoke check.
        Every assistant turn must call one tool in this exact form:
        <tool_call>
        {"name": "replace_text", "arguments": {"path": "FILE.py", "old": "exact existing text", "new": "replacement"}}
        </tool_call>
        Tools: list_files, read_file, search_text, replace_text, write_file, apply_patch, check_syntax, run_public_tests, finish.
        replace_text requires the old text to occur exactly once. write_file replaces an entire allowlisted file.
        apply_patch requires a unified diff. run_public_tests executes test_public.py and is not the grader.
        Do not claim a change succeeded unless a tool result says it was applied. Call finish only after the source changes.
        TASK:
        """
    ).strip() + "\n" + task_prompt.strip()


def render_prompt(messages: list[dict[str, str]], *, close_think: bool) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "tool":
            role = "user"
            content = "<tool_response>\n" + content + "\n</tool_response>"
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    prefix = "<|im_start|>assistant\n"
    if close_think:
        prefix += "<think>\n\n</think>\n"
    parts.append(prefix)
    return "".join(parts)


def _resolve_allowed(raw: str, allowed: set[str]) -> str | None:
    normalized = str(raw).replace("\\", "/").lstrip("./")
    matches = [path for path in allowed if normalized == path or normalized.endswith("/" + path)]
    return matches[0] if len(matches) == 1 else None


class WorkspaceTools:
    def __init__(self, workspace: Path, allowed: set[str], *, public_test: str = "test_public.py") -> None:
        self.workspace = workspace.resolve()
        self.allowed = set(allowed)
        self.public_test = public_test

    def file_contents(self) -> dict[str, str]:
        return {
            name: (self.workspace / name).read_text(encoding="utf-8")
            for name in sorted(self.allowed)
            if (self.workspace / name).is_file()
        }

    def hashes(self) -> dict[str, str]:
        return {name: sha256_text(text) for name, text in self.file_contents().items()}

    def restore(self, contents: dict[str, str] | None) -> None:
        subprocess.run(["git", "checkout", "--", "."], cwd=self.workspace, check=True, capture_output=True)
        subprocess.run(
            ["git", "clean", "-fd", "-e", ".git"],
            cwd=self.workspace,
            check=True,
            capture_output=True,
        )
        if not contents:
            return
        for name, text in contents.items():
            if name not in self.allowed:
                raise ValueError(f"snapshot names a file outside the allowlist: {name}")
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def diff(self) -> str:
        proc = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary"],
            cwd=self.workspace,
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.stdout

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "list_files":
            return {"files": sorted(self.allowed)}
        if name == "read_file":
            return self._read(arguments)
        if name == "search_text":
            return self._search(arguments)
        if name == "replace_text":
            return self._replace(arguments)
        if name == "write_file":
            return self._write(arguments)
        if name == "apply_patch":
            return self._patch(arguments)
        if name == "check_syntax":
            return self._syntax(arguments)
        if name == "run_public_tests":
            return self._public_tests()
        if name == "finish":
            changed = bool(self.diff().strip())
            if not changed:
                return {"finished": False, "error": "source is unchanged; edit a file before finish"}
            return {"finished": True, "summary": str(arguments.get("summary", ""))[:500]}
        return {"error": f"unsupported tool {name}"}

    def _read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_allowed(str(arguments.get("path", "")), self.allowed)
        if path is None:
            return {"error": "path is not allowed", "allowed_paths": sorted(self.allowed)}
        text = (self.workspace / path).read_text(encoding="utf-8")
        if len(text) > _MAX_FILE_CHARS:
            return {"resolved_path": path, "truncated": True, "content": text[:_MAX_FILE_CHARS]}
        return {"resolved_path": path, "content": text}

    def _search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", ""))
        if not query:
            return {"error": "query must not be empty"}
        matches = []
        for path in sorted(self.allowed):
            for line_no, line in enumerate((self.workspace / path).read_text(encoding="utf-8").splitlines(), 1):
                if query in line:
                    matches.append({"path": path, "line": line_no, "text": line[:300]})
                    if len(matches) >= 40:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches}

    def _replace(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_allowed(str(arguments.get("path", "")), self.allowed)
        old = arguments.get("old")
        new = arguments.get("new")
        if path is None:
            return {"error": "path is not allowed", "allowed_paths": sorted(self.allowed)}
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            return {"error": "old and new must be strings and old must be non-empty"}
        if old == new:
            return {"error": "old and new are identical"}
        if len(old) > _MAX_WRITE_CHARS or len(new) > _MAX_WRITE_CHARS:
            return {"error": "replacement exceeds the size limit"}
        file_path = self.workspace / path
        text = file_path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            return {"error": "old text must occur exactly once", "occurrences": count, "path": path}
        file_path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return {"applied": True, "path": path, "sha256": sha256_file(file_path)}

    def _write(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_allowed(str(arguments.get("path", "")), self.allowed)
        content = arguments.get("content")
        if path is None:
            return {"error": "path is not allowed", "allowed_paths": sorted(self.allowed)}
        if not isinstance(content, str):
            return {"error": "content must be a string"}
        if len(content) > _MAX_WRITE_CHARS:
            return {"error": "content exceeds the size limit"}
        file_path = self.workspace / path
        if not file_path.is_file():
            return {"error": "write_file only replaces an existing allowlisted file"}
        file_path.write_text(content, encoding="utf-8")
        return {"applied": True, "path": path, "sha256": sha256_file(file_path)}

    def _patch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        patch = arguments.get("patch", "")
        if not isinstance(patch, str) or not patch.strip():
            return {"error": "patch must be a non-empty string", "hint": _PATCH_HINT}
        if "--- a/" not in patch or "+++ b/" not in patch:
            return {"error": "patch is not a unified diff", "hint": _PATCH_HINT}
        paths = []
        for line in patch.splitlines():
            if line.startswith(("--- a/", "+++ b/")):
                candidate = line[6:]
                if candidate != "/dev/null":
                    paths.append(candidate)
        if not paths or any(path not in self.allowed for path in paths):
            return {"error": "patch paths must be existing allowlisted source files", "hint": _PATCH_HINT}
        checked = subprocess.run(
            ["git", "apply", "--check", "--"],
            cwd=self.workspace,
            input=patch,
            text=True,
            capture_output=True,
            check=False,
        )
        if checked.returncode != 0:
            return {"error": "patch failed validation", "detail": checked.stderr[:500], "hint": _PATCH_HINT}
        applied = subprocess.run(
            ["git", "apply", "--"],
            cwd=self.workspace,
            input=patch,
            text=True,
            capture_output=True,
            check=False,
        )
        if applied.returncode != 0:
            return {"error": "patch failed", "detail": applied.stderr[:500]}
        return {"applied": True, "paths": sorted(set(paths))}

    def _syntax(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_allowed(str(arguments.get("path", "")), self.allowed)
        if path is None:
            return {"error": "path is not allowed", "allowed_paths": sorted(self.allowed)}
        try:
            ast.parse((self.workspace / path).read_text(encoding="utf-8"), filename=path)
        except SyntaxError as exc:
            return {"syntax_valid": False, "path": path, "error": str(exc)}
        return {"syntax_valid": True, "path": path}

    def _public_tests(self) -> dict[str, Any]:
        if self.public_test not in self.allowed and not (self.workspace / self.public_test).is_file():
            return {"error": "public test file is not in the workspace"}
        result = run_jail(
            mounts={"workspace": self.workspace},
            argv=["/usr/bin/python3", self.public_test],
            cwd="/workspace",
            timeout=20,
            max_output=8_000,
        )
        return {
            "returncode": result["returncode"],
            "timed_out": result["timed_out"],
            "stdout": result["stdout"][-4000:],
            "stderr": result["stderr"][-4000:],
            "passed": result["returncode"] == 0 and not result["timed_out"],
            "jail": True,
        }


_JAIL_BOOTSTRAP = r"""
import os, sys
root, cwd = sys.argv[1], sys.argv[2]
mounts = json_mounts = sys.argv[3]
fd = int(sys.argv[4])
argv = sys.argv[5:]
import json as _json
pairs = _json.loads(mounts)
for src, dst in (("/usr", root + "/usr"), ("/lib", root + "/lib"), ("/lib64", root + "/lib64")):
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        if os.system("mount --bind " + src + " " + dst) != 0:
            raise SystemExit(91)
for name, host in pairs.items():
    dest = root + "/" + name
    os.makedirs(dest, exist_ok=True)
    if os.system("mount --bind " + host + " " + dest) != 0:
        raise SystemExit(92)
tmp = root + "/tmp"
os.makedirs(tmp, exist_ok=True)
if os.system("mount -t tmpfs tmpfs " + tmp) != 0:
    raise SystemExit(93)
if fd >= 0:
    os.dup2(fd, 3)
os.closerange(4, 256)
os.chroot(root)
os.chdir(cwd)
os.environ.clear()
os.environ.update({
    "PATH": "/usr/bin",
    "HOME": "/tmp",
    "TMPDIR": "/tmp",
    "LANG": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
})
os.execv(argv[0], argv)
"""


def run_jail(
    *,
    mounts: dict[str, Path],
    argv: list[str],
    cwd: str,
    timeout: float,
    max_output: int,
) -> dict[str, Any]:
    """Run argv in a fresh user, mount, and network namespace rooted at a temp dir."""
    root = Path(tempfile.mkdtemp(prefix="vare-jail-"))
    try:
        for name in ("usr", "lib", "lib64", "tmp"):
            (root / name).mkdir(exist_ok=True)
        payload = json.dumps({name: str(path.resolve()) for name, path in mounts.items()})
        proc = subprocess.run(
            ["unshare", "--user", "--map-root-user", "--mount", "--net", sys.executable, "-c", _JAIL_BOOTSTRAP, str(root), cwd, payload, "-1", *argv],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        limited = len(proc.stdout.encode()) > max_output or len(proc.stderr.encode()) > max_output
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[:max_output],
            "stderr": proc.stderr[:max_output],
            "timed_out": False,
            "output_limited": limited,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        return {
            "returncode": 124,
            "stdout": stdout[:max_output],
            "stderr": stderr[:max_output],
            "timed_out": True,
            "output_limited": False,
        }
    finally:
        subprocess.run(["umount", "-l", str(root / "tmp")], check=False, capture_output=True)
        for name in mounts:
            subprocess.run(["umount", "-l", str(root / name)], check=False, capture_output=True)
        for name in ("usr", "lib", "lib64"):
            subprocess.run(["umount", "-l", str(root / name)], check=False, capture_output=True)
        # The mounts live in the unshare namespace and disappear with it.
        try:
            for child in sorted(root.rglob("*"), reverse=True):
                if child.is_symlink() or child.is_file():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    child.rmdir()
            root.rmdir()
        except OSError:
            pass


def _rpc_encode(value: Any) -> Any:
    """Mark tuples before ``json.dumps``. The stdlib encoder treats tuples as lists and does not call ``default``."""
    if isinstance(value, tuple):
        return {"__tuple__": [_rpc_encode(item) for item in value]}
    if isinstance(value, list):
        return [_rpc_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _rpc_encode(item) for key, item in value.items()}
    return value


def _rpc_decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_rpc_decode(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__tuple__"} and isinstance(value["__tuple__"], list):
            return tuple(_rpc_decode(item) for item in value["__tuple__"])
        return {key: _rpc_decode(item) for key, item in value.items()}
    return value


def _recv_line(sock: socket.socket, pending: bytearray) -> bytes | None:
    while b"\n" not in pending:
        chunk = sock.recv(65536)
        if not chunk:
            return None
        pending.extend(chunk)
    line, _, rest = pending.partition(b"\n")
    pending.clear()
    pending.extend(rest)
    return line


_DRIVER = r"""
import importlib, json, socket, sys
sys.path.insert(0, "/workspace")
sock = socket.socket(fileno=3)
pending = bytearray()

def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__tuple__"} and isinstance(value["__tuple__"], list):
            return tuple(_decode(item) for item in value["__tuple__"])
        return {key: _decode(item) for key, item in value.items()}
    return value

def _encode(value):
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value

while True:
    while b"\n" not in pending:
        chunk = sock.recv(65536)
        if not chunk:
            raise SystemExit(0)
        pending.extend(chunk)
    line, _, rest = pending.partition(b"\n")
    pending.clear()
    pending.extend(rest)
    req = None
    try:
        req = _decode(json.loads(line))
        module = importlib.import_module(req["module"])
        fn = getattr(module, req["function"])
        args = req.get("args") or []
        kwargs = req.get("kwargs") or {}
        value = fn(*args, **kwargs)
        response = {"id": req.get("id"), "ok": True, "value": value, "args_after": args}
    except Exception as exc:
        request_id = req.get("id") if isinstance(req, dict) else None
        response = {"id": request_id, "ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500]}
    sock.sendall(json.dumps(_encode(response)).encode() + b"\n")
"""


class CandidateWorker:
    """One jailed interpreter. The host sends JSON calls; the module under test is imported only there."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.root = Path(tempfile.mkdtemp(prefix="vare-candidate-"))
        self.driver_dir = Path(tempfile.mkdtemp(prefix="vare-driver-"))
        (self.driver_dir / "call.py").write_text(_DRIVER, encoding="utf-8")
        parent, child = socket.socketpair()
        self.parent = parent
        mounts = json.dumps({"workspace": str(self.workspace), "driver": str(self.driver_dir)})
        for name in ("usr", "lib", "lib64", "tmp"):
            (self.root / name).mkdir(exist_ok=True)
        self.proc = subprocess.Popen(
            [
                "unshare", "--user", "--map-root-user", "--mount", "--net",
                sys.executable, "-c", _JAIL_BOOTSTRAP,
                str(self.root), "/workspace", mounts, str(child.fileno()),
                "/usr/bin/python3", "/driver/call.py",
            ],
            pass_fds=(child.fileno(),),
            close_fds=True,
        )
        child.close()
        parent.settimeout(120)
        self._pending = bytearray()
        self._lock = threading.Lock()
        self._next_id = 1

    def call(self, module: str, function: str, args: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self.parent.sendall(json.dumps(_rpc_encode({
                "id": request_id,
                "module": module,
                "function": function,
                "args": args,
                "kwargs": kwargs,
            })).encode() + b"\n")
            line = _recv_line(self.parent, self._pending)
            if line is None:
                raise RuntimeError("candidate worker closed the result channel")
            payload = _rpc_decode(json.loads(line))
            if payload.get("id") != request_id:
                raise RuntimeError("candidate worker returned a mismatched result id")
            return payload

    def close(self) -> None:
        self.parent.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.driver_dir.joinpath("call.py").unlink(missing_ok=True)
        self.driver_dir.rmdir()
        try:
            self.root.rmdir()
        except OSError:
            pass


def _writeback(original: Any, updated: Any) -> None:
    if isinstance(original, list) and isinstance(updated, list):
        if (
            len(original) == len(updated)
            and all(isinstance(left, list) and isinstance(right, list) for left, right in zip(original, updated))
        ):
            for left, right in zip(original, updated):
                _writeback(left, right)
            return
        original.clear()
        original.extend(updated)


def _stub_source(module: str, socket_path: str) -> str:
    return textwrap.dedent(
        f"""
        import json, os, socket
        MODULE = {module!r}
        SOCKET = {socket_path!r}

        def _writeback(original, updated):
            if isinstance(original, list) and isinstance(updated, list):
                if len(original) == len(updated) and all(isinstance(a, list) and isinstance(b, list) for a, b in zip(original, updated)):
                    for left, right in zip(original, updated):
                        _writeback(left, right)
                    return
                original.clear()
                original.extend(updated)

        def _encode(value):
            if isinstance(value, tuple):
                return {{"__tuple__": [_encode(item) for item in value]}}
            if isinstance(value, list):
                return [_encode(item) for item in value]
            if isinstance(value, dict):
                return {{key: _encode(item) for key, item in value.items()}}
            return value

        def _decode(value):
            if isinstance(value, list):
                return [_decode(item) for item in value]
            if isinstance(value, dict):
                if set(value) == {{"__tuple__"}} and isinstance(value["__tuple__"], list):
                    return tuple(_decode(item) for item in value["__tuple__"])
                return {{key: _decode(item) for key, item in value.items()}}
            return value

        def _call(name, args, kwargs):
            sock = socket.socket(socket.AF_UNIX)
            sock.connect(SOCKET)
            sock.sendall(json.dumps(_encode({{"module": MODULE, "function": name, "args": args, "kwargs": kwargs}})).encode() + b"\\n")
            pending = bytearray()
            while b"\\n" not in pending:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                pending.extend(chunk)
            sock.close()
            line = pending.split(b"\\n", 1)[0]
            if not line:
                raise RuntimeError("candidate channel closed")
            response = _decode(json.loads(line))
            if response.get("ok") is not True:
                if response.get("error_type") == "ValueError":
                    raise ValueError(response.get("error", ""))
                raise RuntimeError(response.get("error_type", "Error") + ": " + response.get("error", ""))
            updated = response.get("args_after")
            if isinstance(updated, list):
                for original, new in zip(args, updated):
                    _writeback(original, new)
            return response.get("value")

        def __getattr__(name):
            def call(*args, **kwargs):
                return _call(name, args, kwargs)
            return call
        """
    ).lstrip()


def grade_separated(
    *,
    task_dir: Path,
    workspace: Path,
    timeout_s: float,
) -> dict[str, Any]:
    """Run the task's existing grader script without importing candidate code on the host.

    The grader process receives a stub module. Every function call is executed
    in ``CandidateWorker``. Evaluator ``oracle/`` is not copied next to the
    grader script and is not mounted in the candidate namespace.
    """
    task_dir = task_dir.resolve()
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    command = spec["tests"][0]
    script_name = command["argv"][1]
    module_name = next(
        path.stem
        for path in (task_dir / "subject").glob("*.py")
        if path.name != "test_public.py"
    )
    evaluator_src = task_dir / "evaluator"
    with tempfile.TemporaryDirectory(prefix="vare-grade-") as tmp:
        tmp_path = Path(tmp)
        grader_dir = tmp_path / "grader"
        grader_dir.mkdir()
        for path in evaluator_src.iterdir():
            if path.name == "oracle" or path.name == "__pycache__" or not path.is_file():
                continue
            (grader_dir / path.name).write_bytes(path.read_bytes())
        stub_dir = tmp_path / "stub"
        stub_dir.mkdir()
        candidate_copy = tmp_path / "candidate"
        shutil.copytree(
            workspace,
            candidate_copy,
            ignore=shutil.ignore_patterns(".git", "__pycache__"),
        )
        evidence_before = {
            path.relative_to(workspace).as_posix(): sha256_file(path)
            for path in workspace.rglob("*")
            if path.is_file() and ".git" not in path.parts
        }
        sock_path = tmp_path / "rpc.sock"
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(sock_path))
        server.listen(8)
        worker = CandidateWorker(candidate_copy)
        (stub_dir / f"{module_name}.py").write_text(_stub_source(module_name, str(sock_path)), encoding="utf-8")
        stop = threading.Event()

        def serve() -> None:
            server.settimeout(0.2)
            while not stop.is_set():
                try:
                    conn, _addr = server.accept()
                except TimeoutError:
                    continue
                except OSError:
                    return
                with conn:
                    pending = bytearray()
                    line = _recv_line(conn, pending)
                    if line is None:
                        continue
                    try:
                        request = _rpc_decode(json.loads(line))
                        response = worker.call(
                            request["module"],
                            request["function"],
                            request.get("args") or [],
                            request.get("kwargs") or {},
                        )
                    except Exception as exc:
                        response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500]}
                    conn.sendall(json.dumps(_rpc_encode(response)).encode() + b"\n")

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        started = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                [sys.executable, script_name, str(stub_dir)],
                cwd=grader_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin"),
                    "LANG": "C.UTF-8",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                },
            )
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
            returncode = 124
        finally:
            stop.set()
            server.close()
            worker.close()
        limited = len(stdout.encode()) > int(spec["limits"]["max_output_bytes"])
        passed = returncode == 0 and not timed_out and not limited
        return {
            "task_id": spec["id"],
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "correctness": 1.0 if passed else 0.0,
            "tests": [{
                "name": command.get("name", "hidden-grade"),
                "argv": command["argv"],
                "returncode": returncode,
                "wall_s": round(time.perf_counter() - started, 6),
                "stdout": stdout[: int(spec["limits"]["max_output_bytes"])],
                "stderr": stderr[:4000],
                "timed_out": timed_out,
                "output_limited": limited,
            }],
            "metrics": [],
            "integrity_ok": evidence_before == {
                path.relative_to(workspace).as_posix(): sha256_file(path)
                for path in workspace.rglob("*")
                if path.is_file() and ".git" not in path.parts
            },
            "integrity_failures": [],
            "workspace_diff": "",
            "provenance": {
                "grader": "separated-candidate-worker",
                "candidate_mounts": ["workspace", "driver"],
                "oracle_mounted": False,
            },
        }


@dataclass
class DurableEpisode:
    store: TrajectoryStore
    trajectory_id: str
    policy: PolicyCheckpoint
    worker_id: str
    snapshot_dir: Path
    clock: Callable[[], float] = time.time
    lease_seconds: float = 45
    heartbeat: bool = True
    lease: Any = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def enroll(self, task: Task, expected_steps: int) -> None:
        self.store.register_policy(self.policy)
        self.store.submit(
            self.trajectory_id,
            task,
            expected_steps=expected_steps,
            group_id=self.trajectory_id,
            member=0,
            group_size=1,
            lifetime_seconds=3600,
            now=self.clock(),
        )

    def claim(self) -> None:
        self.lease = self.store.claim(
            "execute",
            self.worker_id,
            lease_seconds=self.lease_seconds,
            now=self.clock(),
        )
        if self.lease is None:
            raise RuntimeError("trajectory lease is still held or the trajectory is gone")
        if self.heartbeat:
            self._stop.clear()
            self._thread = threading.Thread(target=self._beat, daemon=True)
            self._thread.start()

    def _beat(self) -> None:
        while not self._stop.wait(max(self.lease_seconds / 3, 0.05)):
            try:
                self.store.heartbeat(self.lease, lease_seconds=self.lease_seconds, now=self.clock())
            except Exception:
                return

    def committed_steps(self) -> int:
        return int(self.store.resume_state(self.trajectory_id)["next_step"])

    def load_manifest(self, index: int) -> dict[str, Any] | None:
        path = self.snapshot_dir / f"{index:04d}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
        if sha256_json(body) != payload.get("manifest_sha256"):
            raise RuntimeError("committed snapshot hash mismatch")
        return payload

    def resume_manifest(self) -> dict[str, Any] | None:
        state = self.store.resume_state(self.trajectory_id)
        if state["next_step"] <= 0:
            return None
        recorded = state["state"].get("manifest_sha256")
        manifest = self.load_manifest(state["next_step"] - 1)
        if manifest is None or manifest.get("manifest_sha256") != recorded:
            raise RuntimeError("resume snapshot does not match the committed step")
        return manifest

    def commit(self, index: int, manifest_body: dict[str, Any], action: dict[str, Any], observation: dict[str, Any]) -> None:
        digest = sha256_json(manifest_body)
        document = dict(manifest_body)
        document["manifest_sha256"] = digest
        path = self.snapshot_dir / f"{index:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        step = AgentStep(
            index=index,
            event_id=f"turn-{index}",
            policy=self.policy,
            action=action,
            observation=observation,
            state={"manifest_sha256": digest, "turn": index + 1},
        )
        self.store.heartbeat(self.lease, lease_seconds=self.lease_seconds, now=self.clock())
        self.store.append_step(self.lease, step, now=self.clock())

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


def run_episode(
    *,
    tools: WorkspaceTools,
    task_prompt: str,
    generate: Generate,
    max_turns: int,
    max_new_tokens: int,
    max_total_new_tokens: int,
    max_no_tool_retries: int,
    wall_seconds: float,
    close_think: bool,
    durable: DurableEpisode | None = None,
    fault_after_steps: int | None = None,
    fault_phase: str | None = None,
    raise_fault: bool = True,
    attempt: int = 1,
    lost_tool_executions: int = 0,
    spent_tokens: int = 0,
    attempt_log: Path | None = None,
) -> EpisodeResult:
    if fault_phase not in {None, "before_commit", "after_commit"}:
        raise ValueError("fault_phase must be before_commit or after_commit")
    started = time.perf_counter()
    resumed_from = 0
    manifest = None
    messages = [
        {"role": "system", "content": system_prompt(task_prompt)},
        {"role": "user", "content": task_prompt},
    ]
    if durable is not None:
        durable.claim()
        manifest = durable.resume_manifest()
        if manifest is not None:
            tools.restore(manifest["files"])
            messages = manifest["messages"]
            resumed_from = int(manifest["turn"])
        else:
            # A crash before commit can leave a tool's bytes on disk. Resume
            # from the last committed snapshot, which is the git baseline when
            # nothing has been committed.
            tools.restore(None)
        if manifest is not None and spent_tokens == 0:
            spent_tokens = int(manifest.get("new_tokens") or 0)
    before = tools.hashes()
    transcript: list[dict[str, Any]] = []
    new_tokens = spent_tokens
    tool_executions = 0
    no_tool_retries = 0
    stop_reason = "turn_budget_exhausted"
    start_turn = resumed_from
    try:
        for turn in range(start_turn, max_turns):
            if time.perf_counter() - started >= wall_seconds:
                stop_reason = "wall_time_exhausted"
                break
            remaining = max_total_new_tokens - new_tokens
            if remaining <= 0:
                stop_reason = "token_budget_exhausted"
                break
            prompt = render_prompt(messages, close_think=close_think)
            generated = generate(prompt, min(max_new_tokens, remaining))
            new_tokens += generated.new_tokens
            calls = parse_tool_calls(generated.text)
            entry: dict[str, Any] = {
                "turn": turn + 1,
                "model_output": generated.text,
                "new_tokens": generated.new_tokens,
                "tool_calls": [],
            }
            messages.append({"role": "assistant", "content": generated.text})
            executable = [call for call in calls if "name" in call]
            errors = [call for call in calls if "error" in call]
            if not executable:
                no_tool_retries += 1
                reminder = (
                    "The previous response did not contain a usable tool call and no source file was committed. "
                    "Call replace_text, write_file, or apply_patch with a unified diff. Do not describe the edit."
                )
                if errors:
                    reminder += " Parser error: " + errors[0]["error"]
                entry["tool_calls"].extend(errors)
                transcript.append(entry)
                if no_tool_retries > max_no_tool_retries:
                    stop_reason = "no_tool_call_retries_exhausted"
                    break
                messages.append({"role": "user", "content": reminder})
                _append_attempt_log(attempt_log, entry, new_tokens, tool_executions)
                _commit_turn(
                    durable, turn, tools, messages, entry,
                    action={"kind": "inference", "name": "no_tool"},
                    observation={"no_tool_retries": no_tool_retries, "new_tokens": new_tokens},
                    fault_after_steps=fault_after_steps,
                    fault_phase=fault_phase,
                    raise_fault=raise_fault,
                )
                continue
            finished = False
            for call in executable:
                tool_executions += 1
                try:
                    result = tools.execute(call["name"], call["arguments"])
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                entry["tool_calls"].append({"name": call["name"], "arguments": call["arguments"], "result": result})
                messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)[:8000]})
                if call["name"] == "finish" and result.get("finished") is True:
                    finished = True
                    stop_reason = "agent_finished"
                    break
            transcript.append(entry)
            _append_attempt_log(attempt_log, entry, new_tokens, tool_executions)
            _commit_turn(
                durable, turn, tools, messages, entry,
                action={
                    "kind": "tool",
                    "operation_id": f"turn-{turn}",
                    "semantics": "at_least_once",
                    "name": executable[0]["name"],
                },
                observation={"tool_names": [call["name"] for call in executable], "finished": finished, "new_tokens": new_tokens},
                fault_after_steps=fault_after_steps,
                fault_phase=fault_phase,
                raise_fault=raise_fault,
            )
            if finished:
                break
        else:
            stop_reason = "turn_budget_exhausted"
    finally:
        if durable is not None:
            durable.close()
    committed = durable.committed_steps() if durable is not None else len(transcript)
    return EpisodeResult(
        stop_reason=stop_reason,
        messages=messages,
        transcript=transcript,
        new_tokens=new_tokens,
        tool_executions=tool_executions,
        committed_steps=committed,
        elapsed_seconds=round(time.perf_counter() - started, 3),
        source_hashes_before=before,
        source_hashes_after=tools.hashes(),
        diff=tools.diff(),
        attempts=attempt,
        lost_tool_executions=lost_tool_executions,
        resumed_from_step=resumed_from,
    )


def _append_attempt_log(path: Path | None, entry: dict[str, Any], new_tokens: int, tool_executions: int) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "turn": entry.get("turn"),
            "new_tokens": new_tokens,
            "tool_executions": tool_executions,
            "tool_names": [call.get("name") for call in entry.get("tool_calls", [])],
        }, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _commit_turn(
    durable: DurableEpisode | None,
    turn: int,
    tools: WorkspaceTools,
    messages: list[dict[str, str]],
    entry: dict[str, Any],
    *,
    action: dict[str, Any],
    observation: dict[str, Any],
    fault_after_steps: int | None,
    fault_phase: str | None,
    raise_fault: bool,
) -> None:
    if fault_after_steps is not None and fault_phase == "before_commit" and turn + 1 == fault_after_steps:
        _fire_fault(raise_fault)
    if durable is None:
        if fault_after_steps is not None and fault_phase == "after_commit" and turn + 1 == fault_after_steps:
            _fire_fault(raise_fault)
        return
    body = {
        "turn": turn + 1,
        "files": tools.file_contents(),
        "messages": messages,
        "new_tokens": int(observation.get("new_tokens", 0)),
        "entry": {"turn": entry["turn"], "tool_names": [call.get("name") for call in entry["tool_calls"]]},
    }
    durable.commit(turn, body, action, observation)
    if fault_after_steps is not None and fault_phase == "after_commit" and turn + 1 == fault_after_steps:
        _fire_fault(raise_fault)


def _fire_fault(raise_fault: bool) -> None:
    if raise_fault:
        raise InjectedFault("injected fault")
    os._exit(86)


def memory_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:")):
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    info = Path("/proc/meminfo").read_text(encoding="utf-8")
    for key in ("MemAvailable", "SwapTotal", "SwapFree"):
        for line in info.splitlines():
            if line.startswith(key + ":"):
                values[key] = int(line.split()[1]) * 1024
    return values


def canonical_episode(result: EpisodeResult) -> str:
    return canonical_json({
        "stop_reason": result.stop_reason,
        "new_tokens": result.new_tokens,
        "tool_executions": result.tool_executions,
        "committed_steps": result.committed_steps,
        "diff_sha256": sha256_text(result.diff),
    })
