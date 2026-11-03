"""Restricted, CPU-only grader for the Transformers MXFP4 dtype task.

Candidate-controlled source is parsed by this process but never imported here.
The target module is imported only by a short-lived Seatbelt child. The parent
computes the expected values independently and never trusts a worker PASS flag.
"""

from __future__ import annotations

import ast
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import secrets
import shutil
import signal
import stat
import struct
import subprocess
import sys
import sysconfig
import threading
import time
from typing import Any


TASK_ID = "transformers-mxfp4-target-dtype-train-01"
BASELINE_COMMIT = "b47ec788c4ea3051b207afd94281fb75594b0205"
KNOWN_FIX_COMMIT = "676859711989dd9178fbbe8968e5a1f86d519712"
BASELINE_SOURCE_SHA256 = "ec83a032b49ae7bbc7dce3de04b45dab2b715187e84b0806eeb3a73738780bb5"
KNOWN_FIX_SOURCE_SHA256 = "9516b3eb647ad937c766109bfe1792aa936326414ec88787447f43a10c52271e"
KNOWN_FIX_TEST_SHA256 = "9453dd3059a42415717af2b9f4259145801599bddccf107b336b3f4277a1fa6a"
LICENSE_SHA256 = "77fd4710def9ec3c0f6225800e0235f15a425abd4a8b03559127fcd782612049"
EXPECTED_PYTHON = "3.12.12"
EXPECTED_TORCH = "2.9.1"
EXPECTED_ARCHITECTURE = "arm64"
CALIBRATION_SEED = 20261104
TARGET_PATH = "src/transformers/integrations/mxfp4.py"
WORKER_PATH = Path(__file__).with_name("candidate_worker.py")
TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parents[2]
PROTOCOL_PATH = TASK_ROOT / "protocol.json"
LOCK_PATH = TASK_ROOT / "protocol.lock.json"
LOCKED_FILES = {
    "protocol.json": PROTOCOL_PATH,
    "task.json": TASK_ROOT / "task.json",
    "TASK.md": TASK_ROOT / "TASK.md",
    "qualification.json": TASK_ROOT / "qualification.json",
    "grade.py": Path(__file__),
    "candidate_worker.py": WORKER_PATH,
    "calibration_runner.py": REPO_ROOT / "scripts/calibrate_transformers_mxfp4_dtype_grader.py",
    "workspace_preparer.py": REPO_ROOT / "scripts/prepare_transformers_mxfp4_dtype_workspace.py",
    "task_test.py": REPO_ROOT / "tests/test_transformers_mxfp4_dtype_task.py",
    "baseline_input.py": TASK_ROOT / "calibration_inputs/mxfp4.py.baseline",
    "known_fix_input.py": TASK_ROOT / "calibration_inputs/mxfp4.py.known_fix",
    "report_forgery_fixture.py": REPO_ROOT / "tests/fixtures/transformers_mxfp4_dtype/report_forgery.py",
}
FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
BASELINE_NORMALIZED_AST_SHA256 = "a55146e53e0911bc9893a1a96d83be31a9327461b7787f3d68780465a7f332d5"
# Each entry is a pre-reviewed form used to calibrate the behavioral oracle:
# baseline, two behaviorally correct repairs, and two incorrect mutants. This
# remains a restricted patch-contract grader, not a general repair metric.
SUPPORTED_TARGET_AST_PAIRS = {
    ("1d304b931fce5f4534be6089182198067b37be473513fb2b268559e9f3b1fa41", "d844a3ca84cf6fdb368ef7db11c42fe3028f0bef6c33a8fc624168ca9ea7ad70"),
    ("ee28aa83b9102669201f9b04dc641184dda445e1353548e56732ca5d81563e10", "f4067c181b37fe50ad095d72a94485f7396915e28dc33d4261093a3d0b13a543"),
    ("983778491b79a25737dbbcfb0fcbeb67a0d86917f5b47962c7285cf7ce0d1120", "d844a3ca84cf6fdb368ef7db11c42fe3028f0bef6c33a8fc624168ca9ea7ad70"),
    ("53d50c3fcba22b4c54fea3ae5a6751e7f7cea23b62494057e0d597b2623d2a13", "f4067c181b37fe50ad095d72a94485f7396915e28dc33d4261093a3d0b13a543"),
    ("ee28aa83b9102669201f9b04dc641184dda445e1353548e56732ca5d81563e10", "5b5c12245bdfe3d871ac05c0079e51a2eb8e19122150d9d5b9e91f7d9080b657"),
}
DTYPES = ("float32", "bfloat16", "float16", "float64")
PROJECTIONS = ("gate_up_proj", "down_proj")
MAX_WALL_SECONDS = 60.0
MAX_RSS_BYTES = 2 * 1024**3
MAX_OUTPUT_BYTES = 1_000_000
MAX_SOURCE_BYTES = 2 * 1024**2
MAX_GIT_OUTPUT_BYTES = 8 * 1024**2


class GraderError(ValueError):
    """The pinned task source, candidate patch, or evidence is invalid."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _read_regular_file(path: Path) -> bytes:
    """Read one regular file without following a candidate-controlled symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise GraderError("expected a regular file")
        if metadata.st_size > MAX_SOURCE_BYTES:
            raise GraderError("file exceeds the frozen size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            contents = stream.read(MAX_SOURCE_BYTES + 1)
        if len(contents) > MAX_SOURCE_BYTES:
            raise GraderError("file grew beyond the frozen size limit")
        return contents
    finally:
        os.close(descriptor)


def _verify_protocol_lock() -> dict[str, Any]:
    """Fail closed if task, protocol, grader, or worker bytes drifted."""
    try:
        lock_bytes = _read_regular_file(LOCK_PATH)
        lock = json.loads(lock_bytes)
        protocol = json.loads(_read_regular_file(PROTOCOL_PATH))
        expected = lock["files"]
        current = {
            name: sha256_bytes(_read_regular_file(path))
            for name, path in LOCKED_FILES.items()
        }
        if lock.get("task_id") != TASK_ID:
            raise GraderError("task identity differs from protocol lock")
        if protocol.get("task_id") != TASK_ID:
            raise GraderError("protocol task identity differs")
        if protocol.get("freeze_status") != "FROZEN_BEFORE_FINAL_CPU_CONFIRMATION":
            raise GraderError("protocol is not frozen for confirmation")
        if lock.get("frozen_at_utc") != protocol.get("freeze_time_utc"):
            raise GraderError("protocol freeze time differs from protocol lock")
        if lock.get("source", {}).get("baseline_revision") != BASELINE_COMMIT:
            raise GraderError("baseline revision differs from protocol lock")
        if lock.get("source", {}).get("known_fix_revision") != KNOWN_FIX_COMMIT:
            raise GraderError("known-fix revision differs from protocol lock")
        source = lock.get("source", {})
        pinned_source = {
            "baseline_source_sha256": BASELINE_SOURCE_SHA256,
            "known_fix_source_sha256": KNOWN_FIX_SOURCE_SHA256,
            "known_fix_test_sha256": KNOWN_FIX_TEST_SHA256,
            "license_sha256": LICENSE_SHA256,
        }
        if any(source.get(name) != digest for name, digest in pinned_source.items()):
            raise GraderError("upstream source or license hash differs from protocol lock")
        if source.get("license") != "Apache-2.0":
            raise GraderError("upstream license identity differs from protocol lock")
        evaluator = lock.get("evaluator", {})
        locked_pairs = {tuple(pair) for pair in evaluator.get("supported_target_ast_pairs", [])}
        if (
            evaluator.get("baseline_normalized_ast_sha256") != BASELINE_NORMALIZED_AST_SHA256
            or locked_pairs != SUPPORTED_TARGET_AST_PAIRS
        ):
            raise GraderError("restricted patch contract differs from protocol lock")
        if lock.get("calibration", {}).get("seed") != CALIBRATION_SEED:
            raise GraderError("calibration seed differs from protocol lock")
        if expected != current:
            raise GraderError("frozen task or evaluator file hash mismatch")
        return {
            "protocol_lock_sha256": sha256_bytes(lock_bytes),
            "files": current,
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise GraderError(f"protocol lock verification failed:{type(exc).__name__}") from exc


def _target_nodes(tree: ast.Module) -> tuple[ast.ClassDef, ast.FunctionDef, ast.FunctionDef]:
    owners = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Mxfp4Dequantize"]
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "dequantize_convertops"]
    if len(owners) != 1 or len(helpers) != 1:
        raise GraderError("expected one MXFP4 conversion class and helper")
    methods = [node for node in owners[0].body if isinstance(node, ast.FunctionDef) and node.name == "convert"]
    if len(methods) != 1:
        raise GraderError("expected one MXFP4 conversion method")
    return owners[0], methods[0], helpers[0]


def source_contract_bytes(source_bytes: bytes) -> dict[str, Any]:
    """Admit locked calibration forms while rejecting other executable deltas.

    This is intentionally a restricted patch-contract evaluator, not a general
    repository-repair metric. Behavioral checks remain mandatory after this
    static gate.
    """
    try:
        tree = ast.parse(source_bytes.decode("utf-8"))
        owner, method, helper = _target_nodes(tree)
    except (SyntaxError, UnicodeError, GraderError) as exc:
        return {"accepted": False, "reason": f"source_parse_or_target_failure:{type(exc).__name__}"}
    method_hash = sha256_bytes(ast.dump(method, include_attributes=False).encode())
    helper_hash = sha256_bytes(ast.dump(helper, include_attributes=False).encode())
    target_pair = (method_hash, helper_hash)
    normalized = copy.deepcopy(tree)
    normalized_owner, normalized_method, normalized_helper = _target_nodes(normalized)
    normalized_owner.body[normalized_owner.body.index(normalized_method)] = ast.Pass()
    normalized.body[normalized.body.index(normalized_helper)] = ast.Pass()
    normalized_hash = sha256_bytes(ast.dump(normalized, include_attributes=False).encode())
    accepted = (
        normalized_hash == BASELINE_NORMALIZED_AST_SHA256
        and target_pair in SUPPORTED_TARGET_AST_PAIRS
    )
    return {
        "accepted": accepted,
        "reason": "locked_calibration_form" if accepted else "unsupported_patch_contract",
        "normalized_ast_sha256": normalized_hash,
        "method_ast_sha256": method_hash,
        "helper_ast_sha256": helper_hash,
    }


def source_contract(source: Path) -> dict[str, Any]:
    """Apply the locked AST contract to a regular source file."""
    try:
        return source_contract_bytes(_read_regular_file(source))
    except (OSError, GraderError) as exc:
        return {"accepted": False, "reason": f"source_read_failure:{type(exc).__name__}"}


def make_cases(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    for projection in PROJECTIONS:
        for dtype in DTYPES:
            blocks = [
                [
                    [[rng.randrange(256) for _ in range(16)] for _ in range(3)]
                    for _ in range(4)
                ]
                for _ in range(2)
            ]
            scales = [
                [[rng.randrange(117, 130) for _ in range(3)] for _ in range(4)]
                for _ in range(2)
            ]
            cases.append({
                "case_id": f"{projection}:{dtype}",
                "projection": projection,
                "dtype": dtype,
                "blocks": blocks,
                "scales": scales,
            })
    return cases


def _cast_reference(value: float, dtype: str) -> float:
    """Round the independent FP64 oracle to a Torch destination dtype."""
    if dtype == "float64":
        return value
    if dtype == "float32":
        return struct.unpack("=f", struct.pack("=f", value))[0]
    if dtype == "float16":
        return struct.unpack("=e", struct.pack("=e", value))[0]
    if dtype == "bfloat16":
        bits = struct.unpack("=I", struct.pack("=f", value))[0]
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return struct.unpack("=f", struct.pack("=I", rounded))[0]
    raise GraderError(f"unsupported locked destination dtype: {dtype}")


def independent_reference(blocks: list[Any], scales: list[Any], dtype: str) -> list[Any]:
    """Decode FP4 nibbles in Python and round to the requested destination dtype."""
    experts = len(blocks)
    rows = len(blocks[0])
    groups = len(blocks[0][0])
    byte_count = len(blocks[0][0][0])
    output = [[[0.0 for _ in range(rows)] for _ in range(groups * byte_count * 2)] for _ in range(experts)]
    for expert in range(experts):
        for row in range(rows):
            for group in range(groups):
                exponent = int(scales[expert][row][group]) - 127
                for byte_index, packed in enumerate(blocks[expert][row][group]):
                    output[expert][group * byte_count * 2 + 2 * byte_index][row] = math.ldexp(
                        FP4_VALUES[packed & 0x0F], exponent
                    )
                    output[expert][group * byte_count * 2 + 2 * byte_index + 1][row] = math.ldexp(
                        FP4_VALUES[(packed >> 4) & 0x0F], exponent
                    )
    return [
        [[_cast_reference(value, dtype) for value in row] for row in expert]
        for expert in output
    ]


def _numeric_close(actual: Any, expected: Any, *, tolerance: float = 1e-6) -> bool:
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _numeric_close(left, right, tolerance=tolerance)
            for left, right in zip(actual, expected)
        )
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)


def _quote(path: Path) -> str:
    value = str(path.resolve(strict=True))
    if any(char in value for char in "\n\r\t\\\""):
        raise GraderError("sandbox path contains unsupported characters")
    return f'"{value}"'


def cpu_sandbox_profile(
    *,
    candidate_root: Path,
    scratch: Path,
    python: Path,
    python_prefix: Path,
    python_base_prefix: Path,
    runtime_root: Path,
    forbidden_paths: tuple[Path, ...],
) -> str:
    """Build a CPU-only macOS Seatbelt profile; no Metal or GPU services."""
    lines = ["(version 1)", "(deny default)", "(deny network*)", "(deny process-fork)"]
    lines.extend([
        "(allow sysctl-read)",
        f"(allow process-exec (literal {_quote(python)}))",
        f"(allow process-exec (literal {_quote(python.resolve(strict=True))}))",
        '(allow file-read* (literal "/dev/null"))',
        '(allow file-write* (literal "/dev/null"))',
        '(allow file-read* (literal "/dev/urandom"))',
        '(allow file-read-data (literal "/"))',
    ])
    for path in (
        Path("/System"), Path("/usr/bin"), Path("/usr/lib"), Path("/bin"),
        Path("/sbin"), python_prefix, python_base_prefix, runtime_root,
        candidate_root, scratch,
    ):
        if path.exists():
            lines.append(f"(allow file-read* (subpath {_quote(path)}))")
    lines.append(f"(allow file-write* (subpath {_quote(scratch)}))")
    # These denials follow broader runtime/candidate allows by design.
    for path in forbidden_paths:
        if path.exists():
            rule = "subpath" if path.is_dir() else "literal"
            lines.append(f"(deny file-read* ({rule} {_quote(path)}))")
            lines.append(f"(deny file-write* ({rule} {_quote(path)}))")
    return "\n".join(lines) + "\n"


def _process_group_metrics(group_id: int) -> tuple[int, int]:
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss="],
        check=True, capture_output=True, text=True, timeout=5,
    )
    total_rss = 0
    member_count = 0
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        process, _, process_group, kib = map(int, fields)
        if process_group == group_id:
            total_rss += kib * 1024
            if process != group_id:
                member_count += 1
    return total_rss, member_count


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _read_bounded(stream: Any, chunks: list[bytes], state: dict[str, Any]) -> None:
    total = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        total += len(chunk)
        if total > MAX_OUTPUT_BYTES:
            state["overflow"] = True
            return
        chunks.append(chunk)


def _run_worker(
    candidate_root: Path,
    profile_path: Path,
    scratch: Path,
    request: dict[str, Any],
) -> dict[str, Any]:
    python = Path(sys.executable).resolve()
    runtime_root = Path(sysconfig.get_paths()["purelib"]).resolve()
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": str(runtime_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
        "LANG": "C",
        "HOME": str(scratch / "home"),
        "TMPDIR": str(scratch / "tmp"),
    }
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    start = time.monotonic()
    proc = subprocess.Popen(
        ["/usr/bin/sandbox-exec", "-f", str(profile_path), str(python), "-S", str(candidate_root / "candidate_worker.py")],
        cwd=candidate_root, env=environment, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    chunks: list[bytes] = []
    reader_state: dict[str, Any] = {}
    reader = threading.Thread(target=_read_bounded, args=(proc.stdout, chunks, reader_state), daemon=True)
    reader.start()
    proc.stdin.write(payload)
    proc.stdin.close()
    peak_rss = 0
    unexpected_processes = 0
    timed_out = rss_exceeded = output_exceeded = False
    monitor_error = None
    while proc.poll() is None:
        try:
            group_rss, group_members = _process_group_metrics(proc.pid)
            peak_rss = max(peak_rss, group_rss)
            unexpected_processes = max(unexpected_processes, group_members)
        except Exception as exc:  # An unavailable monitor is an infrastructure failure.
            monitor_error = f"{type(exc).__name__}: {exc}"
            _terminate(proc)
            break
        if reader_state.get("overflow"):
            output_exceeded = True
            _terminate(proc)
            break
        if peak_rss > MAX_RSS_BYTES:
            rss_exceeded = True
            _terminate(proc)
            break
        if time.monotonic() - start > MAX_WALL_SECONDS:
            timed_out = True
            _terminate(proc)
            break
        time.sleep(0.05)
    proc.wait()
    try:
        group_rss, group_members = _process_group_metrics(proc.pid)
        peak_rss = max(peak_rss, group_rss)
        unexpected_processes = max(unexpected_processes, group_members)
        if group_members:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as exc:
        monitor_error = f"{type(exc).__name__}: {exc}"
    reader.join(timeout=5)
    return {
        "returncode": proc.returncode,
        "output": b"".join(chunks),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "sampled_process_tree_peak_rss_bytes": peak_rss,
        "unexpected_process_count": unexpected_processes,
        "timed_out": timed_out,
        "rss_limit_exceeded": rss_exceeded,
        "output_limit_exceeded": output_exceeded or bool(reader_state.get("overflow")),
        "monitor_error": monitor_error,
    }


def _result_failures(report: dict[str, Any], request: dict[str, Any]) -> list[dict[str, Any]]:
    expected_by_id = {case["case_id"]: case for case in request["cases"]}
    rows = report.get("cases")
    if not isinstance(rows, list) or len(rows) != len(expected_by_id):
        raise GraderError("worker case inventory differs from the locked request")
    actual_by_id = {row.get("case_id"): row for row in rows if isinstance(row, dict)}
    if set(actual_by_id) != set(expected_by_id):
        raise GraderError("worker case identities are duplicated or unexpected")
    results = []
    for case_id, expected in expected_by_id.items():
        actual = actual_by_id[case_id]
        expected_shape = [2, 96, 4]
        dtype_ok = actual.get("dtype") == f"torch.{expected['dtype']}"
        shape_ok = actual.get("shape") == expected_shape
        values_ok = _numeric_close(
            actual.get("values"),
            independent_reference(expected["blocks"], expected["scales"], expected["dtype"]),
        )
        results.append({
            "case_id": case_id, "dtype_ok": dtype_ok, "shape_ok": shape_ok,
            "values_ok": values_ok, "passed": dtype_ok and shape_ok and values_ok,
        })
    return results


def grade_source(
    source_path: Path,
    output_dir: Path,
    *,
    seed: int | None = None,
    protected_paths: tuple[Path, ...] = (),
    source_workspace: Path | None = None,
) -> dict[str, Any]:
    """Run hidden behavioral cases and independently validate the untrusted output."""
    if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        return {"grader_status": "INFRASTRUCTURE_FAIL", "reason": "macOS Seatbelt is unavailable"}
    try:
        protocol_info = _verify_protocol_lock()
    except GraderError as exc:
        return {"grader_status": "SECURITY_FAIL", "reason": str(exc)}
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = "not-installed"
    runtime_info = {
        "python": sys.version.split()[0],
        "torch": torch_version,
        "os": platform.system(),
        "architecture": platform.machine(),
    }
    if (
        runtime_info["python"] != EXPECTED_PYTHON
        or runtime_info["torch"] != EXPECTED_TORCH
        or runtime_info["os"] != "Darwin"
        or runtime_info["architecture"] != EXPECTED_ARCHITECTURE
    ):
        return {
            "grader_status": "INFRASTRUCTURE_FAIL",
            "reason": "runtime does not match the frozen calibration environment",
            "runtime": runtime_info,
        }
    try:
        source_bytes = _read_regular_file(source_path)
    except (OSError, GraderError) as exc:
        return {
            "grader_status": "SECURITY_FAIL",
            "reason": f"candidate source is not a regular file:{type(exc).__name__}",
        }
    source_info = source_contract_bytes(source_bytes)
    source_sha256 = sha256_bytes(source_bytes)
    if output_dir.is_symlink() or output_dir.exists():
        raise FileExistsError(f"refusing to overwrite grading evidence: {output_dir}")
    output_dir.mkdir(parents=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "grader_status": "SECURITY_FAIL",
        "source_contract": source_info,
        "candidate_source_sha256": source_sha256,
        "worker_sha256": protocol_info["files"]["candidate_worker.py"],
        "protocol_lock_sha256": protocol_info["protocol_lock_sha256"],
        "protocol_file_hashes": protocol_info["files"],
        "worker_report_authenticated": False,
        "worker_nonce_is_correlation_only": True,
        "candidate_runs": [],
    }
    if not source_info["accepted"]:
        result["reason"] = "candidate source is outside the restricted patch contract"
        (output_dir / "grade.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result

    candidate_root = output_dir / "candidate-source"
    scratch = output_dir / "scratch"
    (candidate_root / "src/transformers/integrations").mkdir(parents=True)
    for leaf in ("home", "tmp"):
        (scratch / leaf).mkdir(parents=True)
    (candidate_root / TARGET_PATH).write_bytes(source_bytes)
    worker_bytes = _read_regular_file(WORKER_PATH)
    (candidate_root / "candidate_worker.py").write_bytes(worker_bytes)
    for path in candidate_root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    candidate_root.chmod(0o555)

    runtime_root = Path(sysconfig.get_paths()["purelib"]).resolve()
    installed_transformers = runtime_root / "transformers"
    if source_workspace is None:
        lexical_source = Path(os.path.abspath(os.fspath(source_path)))
        source_workspace = lexical_source.parents[3]
    forbidden = {
        *protected_paths,
        source_workspace,
        source_workspace / ".git",
        TASK_ROOT,
        Path.home() / "VARE",
        Path.home() / ".cache/huggingface/hub",
    }
    if installed_transformers.exists():
        forbidden.add(installed_transformers)
    profile = cpu_sandbox_profile(
        candidate_root=candidate_root,
        scratch=scratch,
        python=Path(sys.executable).resolve(),
        python_prefix=Path(sys.prefix).resolve(),
        python_base_prefix=Path(sys.base_prefix).resolve(),
        runtime_root=runtime_root,
        forbidden_paths=tuple(sorted(forbidden, key=str)),
    )
    profile_path = output_dir / "candidate.sb"
    profile_path.write_text(profile, encoding="utf-8")
    run_seed = secrets.randbits(63) if seed is None else seed
    cases = make_cases(run_seed)
    request = {"schema_version": 1, "nonce": secrets.token_hex(32), "cases": cases}
    request_bytes = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    worker_run = _run_worker(candidate_root, profile_path, scratch, request)
    output_bytes = worker_run.pop("output")
    (output_dir / "worker-output.log").write_bytes(output_bytes)
    result.update({
        "grader_status": "INFRASTRUCTURE_FAIL",
        "candidate_worker_output_sha256": sha256_bytes(output_bytes),
        "sandbox_profile_sha256": sha256_file(profile_path),
        "request_sha256": sha256_bytes(request_bytes),
        "challenge_seed": run_seed,
        "runtime": {
            **runtime_info,
            "platform": platform.platform(),
        },
        "case_count": len(cases),
        "candidate_runs": [worker_run],
    })
    if worker_run["timed_out"] or worker_run["rss_limit_exceeded"] or worker_run["monitor_error"]:
        result["grader_status"] = "INFRASTRUCTURE_FAIL"
        result["reason"] = "worker resource monitor failed or a resource limit was exceeded"
    elif worker_run["output_limit_exceeded"] or worker_run["unexpected_process_count"]:
        result["grader_status"] = "SECURITY_FAIL"
        result["reason"] = "worker exceeded output limits or left an unexpected process"
    elif worker_run["returncode"] != 0:
        output_text = output_bytes.decode("utf-8", errors="replace")
        result["grader_status"] = "SECURITY_FAIL" if "Operation not permitted" in output_text else "INFRASTRUCTURE_FAIL"
        result["reason"] = "candidate worker exited unsuccessfully"
    else:
        try:
            report = json.loads(output_bytes.decode("utf-8"))
            if not isinstance(report, dict) or set(report) != {"schema_version", "nonce", "cases"}:
                raise GraderError("worker report schema mismatch")
            if report["schema_version"] != 1 or report["nonce"] != request["nonce"]:
                raise GraderError("worker report identity or correlation mismatch")
            case_results = _result_failures(report, request)
            behavior_passed = all(row["passed"] for row in case_results)
            result["case_results"] = case_results
            result["grader_status"] = "PASS" if behavior_passed else "TEST_FAIL"
        except (UnicodeError, json.JSONDecodeError, GraderError, KeyError, TypeError) as exc:
            result["grader_status"] = "SECURITY_FAIL"
            result["reason"] = f"untrusted worker output failed independent validation:{type(exc).__name__}"
    (output_dir / "grade.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _read_git_stream(
    stream: Any,
    chunks: list[bytes],
    process: subprocess.Popen[bytes],
    state: dict[str, bool],
    name: str,
) -> None:
    total = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        remaining = MAX_GIT_OUTPUT_BYTES - total
        if remaining > 0:
            chunks.append(chunk[:remaining])
        total += len(chunk)
        if total > MAX_GIT_OUTPUT_BYTES:
            state[name] = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return


def _git(workspace: Path, *args: str) -> bytes:
    git = "/usr/bin/git" if Path("/usr/bin/git").is_file() else shutil.which("git")
    if not git:
        raise GraderError("git is unavailable")
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    command = [
        git,
        "-c", "core.fsmonitor=false",
        "-c", "core.hooksPath=/dev/null",
        "-c", "core.pager=cat",
        "-c", "diff.external=",
        "-C", str(workspace),
        *args,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    state: dict[str, bool] = {}
    readers = (
        threading.Thread(
            target=_read_git_stream,
            args=(process.stdout, stdout_chunks, process, state, "stdout_overflow"),
            daemon=True,
        ),
        threading.Thread(
            target=_read_git_stream,
            args=(process.stderr, stderr_chunks, process, state, "stderr_overflow"),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()
    try:
        returncode = process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    for reader in readers:
        reader.join(timeout=2)
    if any(state.values()):
        raise GraderError("candidate Git inventory exceeded the frozen output limit")
    stdout = b"".join(stdout_chunks)
    stderr = b"".join(stderr_chunks)
    if returncode != 0:
        raise subprocess.CalledProcessError(
            returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return stdout


def _changed_paths(workspace: Path) -> tuple[str, ...]:
    tracked = _git(
        workspace, "diff", "--name-only", "--no-ext-diff", "--no-renames", "-z", "HEAD", "--"
    )
    untracked = _git(workspace, "ls-files", "--others", "--exclude-standard", "-z")
    ignored = _git(
        workspace, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"
    )
    paths = {
        PurePosixPath(os.fsdecode(value)).as_posix()
        for value in (tracked + untracked + ignored).split(b"\0")
        if value
    }
    return tuple(sorted(paths))


def grade_workspace(workspace: Path, output_dir: Path) -> dict[str, Any]:
    """Check a pinned upstream workspace, then grade only its single target edit."""
    try:
        _verify_protocol_lock()
    except GraderError as exc:
        return {"grader_status": "SECURITY_FAIL", "reason": str(exc)}
    workspace = Path(os.path.abspath(os.fspath(workspace)))
    try:
        metadata = os.lstat(workspace)
    except OSError as exc:
        return {"grader_status": "INFRASTRUCTURE_FAIL", "reason": f"workspace_unavailable:{type(exc).__name__}"}
    git_metadata = workspace / ".git"
    if not stat.S_ISDIR(metadata.st_mode) or workspace.is_symlink():
        return {"grader_status": "SECURITY_FAIL", "reason": "workspace is not a real directory"}
    if git_metadata.is_symlink() or not git_metadata.is_dir():
        return {"grader_status": "SECURITY_FAIL", "reason": "workspace Git metadata is not a private directory"}
    try:
        head = _git(workspace, "rev-parse", "HEAD").decode().strip()
        shallow = _git(workspace, "rev-parse", "--is-shallow-repository").decode().strip()
        remotes = _git(workspace, "remote").decode().strip()
        paths = _changed_paths(workspace)
    except (subprocess.CalledProcessError, GraderError, subprocess.TimeoutExpired) as exc:
        return {"grader_status": "SECURITY_FAIL", "reason": f"workspace_git_preflight:{type(exc).__name__}"}
    except OSError as exc:
        return {"grader_status": "INFRASTRUCTURE_FAIL", "reason": f"workspace_git_preflight:{type(exc).__name__}"}
    if head != BASELINE_COMMIT or shallow != "true" or remotes:
        return {"grader_status": "SECURITY_FAIL", "reason": "workspace revision, shallow state, or remote configuration differs"}
    if any(path != TARGET_PATH for path in paths):
        return {"grader_status": "SECURITY_FAIL", "reason": "candidate changed a non-allowlisted path", "changed_paths": paths}
    source = workspace / TARGET_PATH
    result = grade_source(
        source,
        output_dir,
        protected_paths=(workspace, git_metadata),
        source_workspace=workspace,
    )
    result["candidate_revision"] = head
    result["changed_paths"] = paths
    grade_file = output_dir / "grade.json"
    if grade_file.exists():
        grade_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = grade_workspace(args.workspace, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("grader_status") in {"PASS", "TEST_FAIL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
