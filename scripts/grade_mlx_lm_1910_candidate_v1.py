#!/usr/bin/env python3
"""Grade an MLX-LM #1910 candidate in a pinned isolated container."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import tempfile
import time
from pathlib import Path
from typing import Any

SPEC = importlib.util.spec_from_file_location(
    "mlx1910_calibration", Path(__file__).with_name("qualify_mlx_lm_1910_v6.py")
)
assert SPEC and SPEC.loader
CAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAL)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_execution_interference(source: bytes) -> list[str]:
    """Reject process control and output forgery outside this parser task's contract.

    The candidate module is imported by both pytest and the isolated oracle.
    This task only permits a parser repair; process exits, dynamic imports, and
    import-time output can otherwise make the candidate forge a successful
    evaluator return code without running either check.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"candidate source is not valid Python: {exc.msg}"]

    findings: list[str] = []
    denied_modules = {
        "os", "sys", "signal", "subprocess", "ctypes", "builtins", "importlib",
        "runpy", "posix", "nt", "_posixsubprocess",
    }
    denied_names = {
        "__import__", "exec", "eval", "compile", "print", "exit", "quit", "open",
        "getattr", "setattr", "globals", "locals", "vars", "__builtins__",
    }
    denied_attributes = {
        "_exit", "exit", "system", "popen", "kill", "dup2", "write", "fileno",
        "stdout", "stderr",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in denied_modules:
                    findings.append(f"process-control import is not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] in denied_modules:
                findings.append(f"process-control import is not allowed: {node.module}")
        elif isinstance(node, ast.Name) and node.id in denied_names:
            findings.append(f"evaluator-interference name is not allowed: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in denied_attributes:
            findings.append(f"process-control attribute is not allowed: {node.attr}")
    return sorted(set(findings))


def grade_source(repo: Path, candidate: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite grader evidence: {output}")
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("candidate source must be a regular file")
    candidate_bytes = candidate.read_bytes()
    if len(candidate_bytes) > 1_000_000 or b"\x00" in candidate_bytes:
        raise ValueError("candidate source is oversized or contains NUL bytes")
    output.mkdir(parents=True)
    (output / "raw").mkdir()
    started = time.monotonic()
    record: dict[str, Any] = {"candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest()}
    interference = candidate_execution_interference(candidate_bytes)
    if interference:
        result = {
            "schema_version": 1,
            "protocol_id": "mlsys-repo-task-mlx-lm-1910-candidate-grader-v1",
            "status": "FAIL",
            "candidate": {**record, "execution_boundary": {"passed": False, "findings": interference}},
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "claim_limit": "Development-task semantic calibration only; no held-out capability claim.",
        }
        (output / "grade.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    record["execution_boundary"] = {"passed": True, "findings": []}
    with tempfile.TemporaryDirectory(prefix="vare-mlx1910-grade-", dir=str(output)) as temp:
        source = Path(temp) / "candidate"
        CAL.archive_commit(repo, CAL.BASE, source)
        target = source / CAL.TARGET
        target.chmod(0o644)
        target.write_bytes(candidate_bytes)
        target.chmod(0o444)
        test_path = source / CAL.TEST
        test_path.parent.mkdir(parents=True, exist_ok=True)
        fixed_test = CAL.run(["git", "-C", str(repo), "show", f"{CAL.FIXED}:{CAL.TEST}"])
        if fixed_test.returncode or CAL.sha256_bytes(fixed_test.stdout) != CAL.TEST_SHA256:
            raise RuntimeError("pinned upstream public regression test failed identity validation")
        if test_path.exists():
            if test_path.is_symlink() or not test_path.is_file():
                raise RuntimeError("upstream public test path is not a regular file")
            test_path.chmod(0o644)
        test_path.write_bytes(fixed_test.stdout)
        test_path.chmod(0o444)
        shim = source / "mlx_lm" / "_version.py"
        shim_content = f'__version__ = "0+source.{CAL.BASE[:7]}"\n'.encode()
        shim.write_bytes(shim_content)
        shim.chmod(0o444)

        public = CAL.docker_run(source, ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                         CAL.TEST, "-k", "test_qwen3_coder_untyped_param",
                                         "--disable-warnings", "--maxfail=1"], timeout=120)
        (output / "raw" / "upstream-test.log").write_bytes(public.stdout)
        record["upstream_test"] = {"returncode": public.returncode,
                                   "output_sha256": CAL.sha256_bytes(public.stdout),
                                   "wall_seconds": round(time.monotonic() - started, 3)}

        payload = json.dumps([case[1] for case in CAL.CASES], separators=(",", ":")).encode()
        oracle_started = time.monotonic()
        oracle = CAL.docker_run(source, ["python", "-c", CAL.CHILD], stdin=payload, timeout=60)
        (output / "raw" / "oracle.json").write_bytes(oracle.stdout)
        actual = None
        parse_error = None
        if oracle.returncode == 0:
            try:
                actual = json.loads(oracle.stdout.decode().splitlines()[-1])
            except Exception as exc:  # diagnostic data from isolated candidate
                parse_error = f"{type(exc).__name__}: {exc}"
        cases = []
        if isinstance(actual, list) and len(actual) == len(CAL.CASES):
            for (case_id, _, expected), got in zip(CAL.CASES, actual):
                cases.append({"case_id": case_id, "passed": got == {"ok": True, "value": expected},
                              "actual": got})
        record["oracle"] = {"returncode": oracle.returncode,
                            "output_sha256": CAL.sha256_bytes(oracle.stdout),
                            "wall_seconds": round(time.monotonic() - oracle_started, 3),
                            "parse_error": parse_error, "cases": cases,
                            "passed": len(cases) == len(CAL.CASES) and all(c["passed"] for c in cases)}
    passed = record["upstream_test"]["returncode"] == 0 and record["oracle"]["passed"]
    result = {"schema_version": 1, "protocol_id": "mlsys-repo-task-mlx-lm-1910-candidate-grader-v1",
              "status": "PASS" if passed else "FAIL", "candidate": record,
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "claim_limit": "Development-task semantic calibration only; no held-out capability claim."}
    (output / "grade.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = grade_source(args.git_source.resolve(strict=True), args.candidate_source.resolve(strict=True), args.output)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
