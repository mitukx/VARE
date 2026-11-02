#!/usr/bin/env python3
"""Evaluate the pinned urllib3 task through a local HTTP server in Seatbelt."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import selectors
import signal
import socket
import subprocess
import sys
import time
from typing import Any


TASK_ID = "urllib3-preserve-ambiguous-request-target-v1"
MAX_OUTPUT_BYTES = 262_144
CASE_TIMEOUT_SECONDS = 3.0
MAX_REQUEST_LINE_BYTES = 8_192
CHILD_CODE = r'''
import json, sys
from urllib.parse import urlsplit

expected = urlsplit(sys.argv[1])
workspace = sys.argv[2]
import urllib3
required = {
    urllib3.poolmanager.PoolManager.urlopen.__code__,
    urllib3.connectionpool.HTTPConnectionPool.urlopen.__code__,
    urllib3.connectionpool.HTTPConnectionPool._make_request.__code__,
    urllib3.connection.HTTPConnection.connect.__code__,
    urllib3.connection.HTTPConnection._new_conn.__code__,
    urllib3.util.connection.create_connection.__code__,
}

def guard(event, args):
    if event != "socket.connect":
        return
    address = args[1]
    if (not isinstance(address, tuple) or len(address) < 2
            or address[1] != expected.port
            or address[0] not in {expected.hostname, "127.0.0.1", "::1", "localhost"}):
        raise PermissionError("connection is outside the task-local grader endpoint")
    seen = set()
    frame = sys._getframe(1)
    while frame is not None:
        seen.add(frame.f_code)
        frame = frame.f_back
    if not required.issubset(seen):
        raise PermissionError("grader connection did not use the urllib3 request path")

sys.addaudithook(guard)
response = urllib3.PoolManager().urlopen(
    method="GET", url=sys.argv[1], retries=False, timeout=1.0
)
print(json.dumps({"status": response.status}))
response.release_conn()
'''


def candidate_scope_valid(workspace: Path) -> tuple[bool, str | None]:
    """Permit only URL-normalization logic changes in the production method."""
    relative = "src/urllib3/connectionpool.py"
    try:
        baseline = subprocess.run(
            ["git", "-C", str(workspace), "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        candidate_path = workspace / relative
        if candidate_path.is_symlink() or not candidate_path.resolve().is_relative_to(workspace.resolve()):
            return False, "candidate source path is a symlink or escapes its workspace"
        candidate = candidate_path.read_text(encoding="utf-8")
        trees = [ast.parse(source) for source in (baseline, candidate)]
        normalized = []
        methods = []
        for tree in trees:
            matches = [
                method
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == "HTTPConnectionPool"
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and method.name == "urlopen"
            ]
            if len(matches) != 1:
                return False, "expected one HTTPConnectionPool.urlopen definition"
            methods.append(copy.deepcopy(matches[0]))
            matches[0].body = [ast.Pass()]
            normalized.append(ast.dump(tree, include_attributes=False))
        if normalized[0] != normalized[1]:
            return False, "candidate changed syntax outside HTTPConnectionPool.urlopen"

        baseline_method, candidate_method = methods

        def header_index(method: ast.FunctionDef | ast.AsyncFunctionDef) -> int | None:
            for index, statement in enumerate(method.body):
                if not isinstance(statement, ast.If):
                    continue
                test = statement.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "headers"
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Is)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value is None
                ):
                    return index
            return None

        def is_url_encoding_block(statement: ast.stmt) -> bool:
            return (
                isinstance(statement, ast.If)
                and isinstance(statement.test, ast.Call)
                and isinstance(statement.test.func, ast.Attribute)
                and isinstance(statement.test.func.value, ast.Name)
                and statement.test.func.value.id == "url"
                and statement.test.func.attr == "startswith"
            )

        baseline_header = header_index(baseline_method)
        candidate_header = header_index(candidate_method)
        if baseline_header is None or candidate_header is None:
            return False, "production method lost its stable header-initialization boundary"
        baseline_suffix = list(baseline_method.body[baseline_header:])
        candidate_suffix = list(candidate_method.body[candidate_header:])
        relocated_encoding = False
        for index, statement in enumerate(baseline_suffix[1:], start=1):
            if is_url_encoding_block(statement):
                del baseline_suffix[index]
                relocated_encoding = True
                break
        if (
            ast.dump(ast.Module(body=candidate_suffix, type_ignores=[]), include_attributes=False)
            not in {
                ast.dump(ast.Module(body=baseline_suffix, type_ignores=[]), include_attributes=False),
                ast.dump(
                    ast.Module(body=list(baseline_method.body[baseline_header:]), type_ignores=[]),
                    include_attributes=False,
                ),
            }
        ):
            return False, "candidate changed the request/connection path after URL normalization"

        if ast.dump(baseline_method.body[0], include_attributes=False) != ast.dump(
            candidate_method.body[0], include_attributes=False
        ):
            return False, "candidate changed the production method contract/docstring"
        prefix = candidate_method.body[1:candidate_header]
        allowed_names = {"url", "parsed_url", "destination_scheme", "parse_url", "to_str", "_encode_target"}
        allowed_statement_types = (ast.Assign, ast.If)
        allowed_expression_types = (
            ast.Assign, ast.If, ast.Name, ast.Constant, ast.Call, ast.Attribute, ast.Load, ast.Store,
            ast.UnaryOp, ast.Not, ast.BoolOp, ast.And, ast.Or, ast.Compare, ast.Is,
        )
        for statement in prefix:
            if not isinstance(statement, allowed_statement_types):
                return False, "URL prefix contains a statement outside the normalization contract"
            for node in ast.walk(statement):
                if not isinstance(node, allowed_expression_types):
                    return False, f"URL prefix contains disallowed syntax: {type(node).__name__}"
                if isinstance(node, ast.Name) and node.id not in allowed_names:
                    return False, f"URL prefix refers to a disallowed name: {node.id}"
                if isinstance(node, ast.Attribute):
                    if not isinstance(node.value, ast.Name) or (
                        (node.value.id, node.attr) not in {
                            ("url", "startswith"), ("parsed_url", "scheme"), ("parsed_url", "url")
                        }
                    ):
                        return False, "URL prefix accesses an attribute outside the normalization contract"
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id not in {
                        "parse_url", "to_str", "_encode_target"
                    }:
                        return False, "URL prefix calls a function outside the normalization contract"
                    if not (
                        isinstance(node.func, ast.Name)
                        or (
                            isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name)
                            and node.func.value.id == "url"
                            and node.func.attr == "startswith"
                        )
                    ):
                        return False, "URL prefix uses a dynamic call target"
        if not relocated_encoding:
            # The untouched baseline is an admissible failure control.
            candidate_body_dump = ast.dump(
                ast.Module(body=candidate_method.body, type_ignores=[]), include_attributes=False
            )
            baseline_body_dump = ast.dump(
                ast.Module(body=methods[0].body, type_ignores=[]), include_attributes=False
            )
            if candidate_body_dump == baseline_body_dump:
                return True, None
            return False, "candidate did not move the existing URL-normalization block"
        return True, None
    except (OSError, UnicodeError, SyntaxError, subprocess.CalledProcessError) as exc:
        return False, f"candidate source scope could not be verified: {type(exc).__name__}: {exc}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_profile(path: Path, workspace: Path, port: int) -> None:
    """Deny candidate reads/writes outside its checkout and non-grader traffic."""
    denied_home = Path.home().resolve()
    work_parent = workspace.parent.resolve()
    profile = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec)",
        # Python and its stdlib can read runtime files; user/work/evaluator
        # paths are denied below, then only the candidate checkout is reopened.
        "(allow file-read*)",
        f'(deny file-read* (subpath "{denied_home}"))',
        f'(deny file-read* (subpath "{work_parent}"))',
        f'(allow file-read* (subpath "{workspace}"))',
        # The candidate runtime is read-only. Evaluation assets and other files
        # remain denied by the default rule.
        f'(allow network-outbound (remote ip "localhost:{port}"))',
    ]
    path.write_text("\n".join(profile) + "\n", encoding="utf-8")


def _child_limits() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (10, 11))
    except (OSError, ValueError) as exc:
        os.write(2, f"RLIMIT_CPU setup failed: {type(exc).__name__}: {exc}\n".encode())
        raise
    file_size = 16 * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
    except (OSError, ValueError) as exc:
        os.write(2, f"RLIMIT_FSIZE setup failed: {type(exc).__name__}: {exc}\n".encode())
        raise


def read_request_line(conn: socket.socket, deadline: float) -> tuple[str | None, str | None]:
    """Read one bounded request line without allowing slow-drip deadline extension."""
    raw = bytearray()
    while b"\r\n" not in raw and len(raw) < MAX_REQUEST_LINE_BYTES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, "request-line deadline expired"
        conn.settimeout(min(0.025, remaining))
        try:
            chunk = conn.recv(min(1024, MAX_REQUEST_LINE_BYTES - len(raw)))
        except socket.timeout:
            continue
        except OSError as exc:
            return None, f"{type(exc).__name__}: {exc}"
        if not chunk:
            return None, "connection closed before request line"
        raw.extend(chunk)
    if b"\r\n" not in raw:
        return None, "request line exceeded size limit"
    return bytes(raw).split(b"\r\n", 1)[0].decode("latin-1"), None
def run_case(
    *,
    workspace: Path,
    profile_path: Path,
    python: str,
    case_id: str,
    path: str,
) -> dict[str, Any]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(0.025)
    port = listener.getsockname()[1]
    write_profile(profile_path, workspace, port)

    env = {
        "PATH": "<HOMEBREW_PREFIX>/bin:/usr/bin:/bin",
        "HOME": str(workspace),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(workspace),
        "PYTHONPATH": str(workspace / "src"),
    }
    command = [
        "/usr/bin/sandbox-exec",
        "-f",
        str(profile_path),
        python,
        "-c",
        CHILD_CODE,
        f"http://127.0.0.1:{port}{path}",
        str(workspace),
    ]
    process_result: dict[str, Any] | None = None
    request_line: str | None = None
    server_error: str | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=_child_limits,
        )
        # The local server runs in the grader process. The sandboxed candidate
        # can see the request input and receive a fixed 200 response, but cannot
        # read the test vectors or grader code.
        selector = selectors.DefaultSelector()
        assert process.stdout is not None and process.stderr is not None
        for stream in (process.stdout, process.stderr):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        stream_names = {process.stdout.fileno(): "stdout", process.stderr.fileno(): "stderr"}
        deadline = time.monotonic() + CASE_TIMEOUT_SECONDS
        timed_out = False
        output_limited = False
        killed = False
        accepted = False
        while selector.get_map() or process.poll() is None or not accepted:
            if not accepted:
                try:
                    conn, _ = listener.accept()
                    accepted = True
                    try:
                        request_line, read_error = read_request_line(conn, deadline)
                        if request_line is not None:
                            conn.settimeout(min(0.025, max(0.001, deadline - time.monotonic())))
                            conn.sendall(
                                b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            )
                        elif server_error is None:
                            server_error = read_error
                    finally:
                        conn.close()
                except socket.timeout:
                    pass
                except OSError as exc:
                    accepted = True
                    server_error = f"{type(exc).__name__}: {exc}"

            now = time.monotonic()
            if not accepted and process.poll() is not None:
                # A deterministic application failure (for example the
                # pinned base parser exception) must not wait out the server
                # deadline when the candidate exited without connecting.
                accepted = True
            if not killed and now >= deadline:
                timed_out = True
                killed = True
                accepted = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for key, _ in selector.select(timeout=0.025):
                stream = key.fileobj
                name = stream_names[stream.fileno()]
                try:
                    chunk = os.read(stream.fileno(), 65_536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                used = sum(len(value) for value in buffers.values())
                room = max(0, MAX_OUTPUT_BYTES - used)
                if room:
                    buffers[name].extend(chunk[:room])
                if len(chunk) > room and not killed:
                    output_limited = True
                    killed = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        try:
            returncode = process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait()
            timed_out = True
        selector.close()
        process_result = {
            "returncode": returncode,
            "wall_seconds": round(time.monotonic() - (deadline - CASE_TIMEOUT_SECONDS), 6),
            "timed_out": timed_out,
            "output_limited": output_limited,
            "stdout": buffers["stdout"].decode("utf-8", errors="replace"),
            "stderr": buffers["stderr"].decode("utf-8", errors="replace"),
        }
    finally:
        listener.close()

    expected_line = f"GET {path} HTTP/1.1"
    stdout = "" if process_result is None else process_result["stdout"]
    try:
        status = json.loads(stdout.strip())
    except (json.JSONDecodeError, ValueError):
        status = None
    passed = bool(
        process_result is not None
        and process_result["returncode"] == 0
        and not process_result["timed_out"]
        and not process_result["output_limited"]
        and request_line == expected_line
        and isinstance(status, dict)
        and status == {"status": 200}
        and server_error is None
    )
    return {
        "case_id": case_id,
        "path": path,
        "expected_request_line": expected_line,
        "observed_request_line": request_line,
        "passed": passed,
        "server_error": server_error,
        "process": process_result,
    }


def git_value(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(workspace), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def integrity_calibration(workspace: Path, task_root: Path) -> int:
    """Support VARE's generic locked-input tamper calibration without running a task."""
    try:
        task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
        lock = json.loads((task_root / "protocol.lock.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        print(json.dumps({
            "grader_error": "ValueError",
            "message": "task descriptor does not match the locked protocol",
        }))
        return 1

    locked = lock.get("locked_hashes", {})
    assets = {
        "task_json_sha256": task_root / "task.json",
        "task_brief_sha256": task_root / "TASK.md",
        "evaluator_sha256": Path(__file__).resolve(),
    }
    mismatches = [
        key for key, path in assets.items()
        if not locked.get(key) or not path.is_file() or sha256_file(path) != locked[key]
    ]
    if task.get("task_id") != TASK_ID or lock.get("task_id") != TASK_ID or mismatches:
        print(json.dumps({
            "grader_error": "ValueError",
            "message": "task descriptor does not match the locked protocol",
            "assets": mismatches,
        }))
        return 1

    candidate_paths = task.get("source", {}).get("candidate_paths", [])
    missing = [relative for relative in candidate_paths if not (workspace / relative).is_file()]
    if missing:
        print(json.dumps({
            "grader_error": "FileNotFoundError",
            "message": "candidate checkout is missing source files",
            "missing": missing,
        }))
        return 1

    # This compatibility mode is only an integrity-calibration contract; the
    # task-specific CLI below performs the actual isolated evaluation.
    print(json.dumps({
        "grader_error": "RuntimeError",
        "message": "integrity calibration mode does not execute candidates",
    }))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--source-record", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--python", default="<HOMEBREW_PREFIX>/bin/python3.11")
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    task_root = args.task_root.expanduser().resolve()
    if args.source_record is None and args.task_id is None:
        return integrity_calibration(workspace, task_root)
    if args.source_record is None or args.task_id is None:
        parser.error("--source-record and --task-id must be provided together")
    repo_root = task_root.parents[2]
    record_path = args.source_record.expanduser().resolve()
    task_path = task_root / "task.json"
    lock_path = task_root / "protocol.lock.json"
    grader_path = Path(__file__).resolve()
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(json.dumps({"grader_error": f"locked input unreadable: {type(exc).__name__}: {exc}"}))
        return 2

    if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        print(json.dumps({"grader_error": "macOS Seatbelt (sandbox-exec) is required"}))
        return 2
    if task.get("task_id") != TASK_ID or args.task_id != TASK_ID or lock.get("task_id") != TASK_ID:
        print(json.dumps({"grader_error": "task identity mismatch"}))
        return 2
    if record.get("task_id") != TASK_ID:
        print(json.dumps({"grader_error": "source record task identity mismatch"}))
        return 2
    if workspace != Path(record.get("workspace_path", "")).resolve():
        print(json.dumps({"grader_error": "workspace/source-record identity mismatch"}))
        return 2
    if workspace == task_root or workspace.is_relative_to(task_root) or workspace.is_relative_to(repo_root):
        print(json.dumps({"grader_error": "candidate workspace overlaps evaluator/repository"}))
        return 2
    if record_path == workspace or record_path.is_relative_to(workspace):
        print(json.dumps({"grader_error": "source record must be outside candidate workspace"}))
        return 2
    expected_revision = record.get("source_revision")
    if expected_revision != task["source"]["base_revision"] or record.get("source_tree") != task["source"]["base_tree"]:
        print(json.dumps({"grader_error": "source revision/tree is not a locked task revision"}))
        return 2
    if not workspace.is_dir() or not record_path.is_file():
        print(json.dumps({"grader_error": "candidate workspace/source record missing"}))
        return 2

    locked_hashes = lock.get("locked_hashes", {})
    asset_paths = {
        "task_json_sha256": task_path,
        "task_brief_sha256": task_root / "TASK.md",
        "evaluator_sha256": grader_path,
        "grader_entrypoint_sha256": repo_root / "scripts/grade_urllib3_3352_task.py",
        "prepare_script_sha256": repo_root / "scripts/prepare_urllib3_3352_task.py",
        "calibration_script_sha256": repo_root / "scripts/calibrate_urllib3_3352_task.py",
        "vare_spec_sha256": task_root / "vare_spec.json",
        "model_screen_script_sha256": repo_root / "scripts/run_urllib3_3352_model_screen.py",
        "adversarial_test_script_sha256": repo_root / "scripts/adversarial_test_urllib3_3352.py",
        "evaluator_regression_test_sha256": repo_root / "tests/test_urllib3_3352_evaluator.py",
        "prompt_sha256": task_root / "prompt.txt",
        "source_context_sha256": task_root / "source_context.txt",
    }
    integrity_errors = []
    for key, path in asset_paths.items():
        expected = locked_hashes.get(key)
        if not expected or not path.is_file() or sha256_file(path) != expected:
            integrity_errors.append(key)
    record_hash = sha256_file(record_path)
    if integrity_errors:
        print(json.dumps({"grader_error": "integrity lock mismatch", "assets": integrity_errors}))
        return 2

    try:
        baseline_head = git_value(workspace, "rev-parse", "HEAD")
        baseline_tree = git_value(workspace, "rev-parse", "HEAD^{tree}")
        remotes = git_value(workspace, "remote")
        if (
            baseline_head != record.get("workspace_baseline_commit")
            or baseline_tree != record.get("workspace_baseline_tree")
            or baseline_tree != lock.get("source", {}).get("base_workspace_tree")
            or remotes
            or record.get("remote_count") != 0
            or record.get("version_shim_sha256") != lock["source"].get("version_shim_sha256")
        ):
            print(json.dumps({"grader_error": "candidate workspace baseline changed or contains remotes"}))
            return 2
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(json.dumps({"grader_error": f"candidate git baseline invalid: {exc}"}))
        return 2

    source_scope_valid, source_scope_error = candidate_scope_valid(workspace)
    cases = lock["acceptance"]["fail_to_pass"] + lock["acceptance"]["pass_to_pass"]
    result_cases = []
    profile_path = task_root / "evaluator" / ".runtime-profile.sb"
    try:
        if source_scope_valid:
            for case in cases:
                result_cases.append(
                    run_case(
                        workspace=workspace,
                        profile_path=profile_path,
                        python=args.python,
                        case_id=case["case_id"],
                        path=case["path"],
                    )
                )
                # A candidate process is never permitted to write the evaluator;
                # recompute the lock after every command to fail closed.
                if any(sha256_file(path) != locked_hashes[key] for key, path in asset_paths.items()):
                    integrity_errors.append("asset_changed_during_evaluation")
                    break
    finally:
        profile_path.unlink(missing_ok=True)

    try:
        status = subprocess.run(
            ["git", "-C", str(workspace), "status", "--porcelain", "-z", "--untracked-files=all"],
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8", errors="strict")
        changed_paths = [line[3:] for line in status.split("\0") if len(line) >= 4]
        allowed = set(task["source"]["candidate_paths"])
        unexpected = [path for path in changed_paths if path not in allowed]
        patch = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--binary", "HEAD", "--", *sorted(allowed)],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        print(json.dumps({"grader_error": f"candidate diff inspection failed: {exc}"}))
        return 2

    if any(sha256_file(path) != locked_hashes[key] for key, path in asset_paths.items()):
        integrity_errors.append("asset_changed_after_evaluation")
    f2p_ids = {case["case_id"] for case in lock["acceptance"]["fail_to_pass"]}
    p2p_ids = {case["case_id"] for case in lock["acceptance"]["pass_to_pass"]}
    by_id = {case["case_id"]: case for case in result_cases}
    f2p_passed = {case_id: by_id.get(case_id, {}).get("passed", False) for case_id in sorted(f2p_ids)}
    p2p_passed = {case_id: by_id.get(case_id, {}).get("passed", False) for case_id in sorted(p2p_ids)}
    passed = (
        len(result_cases) == len(cases)
        and all(f2p_passed.values())
        and all(p2p_passed.values())
        and not unexpected
        and source_scope_valid
        and not integrity_errors
        and bool(patch)
    )
    output = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "passed": passed,
        "source_revision": expected_revision,
        "source_tree": record["source_tree"],
        "workspace_baseline_commit": baseline_head,
        "workspace_baseline_tree": baseline_tree,
        "source_record_sha256": record_hash,
        "candidate_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "candidate_changed_paths": changed_paths,
        "candidate_source_scope_valid": source_scope_valid,
        "candidate_source_scope_error": source_scope_error,
        "unexpected_changed_paths": unexpected,
        "fail_to_pass": f2p_passed,
        "pass_to_pass": p2p_passed,
        "integrity_errors": integrity_errors,
        "cases": result_cases,
        "limits": {
            "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
            "max_output_bytes": MAX_OUTPUT_BYTES,
            "memory_limit_mb": None,
            "memory_limit_enforced": False,
            "memory_limit_note": "macOS resource API rejected RLIMIT_AS; no memory cap is claimed",
            "cpu_limit_seconds": 10,
            "network_allowlist": "grader localhost ephemeral TCP port only",
            "candidate_filesystem": "read-only candidate checkout; home/work/evaluator reads denied",
        },
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
