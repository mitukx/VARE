#!/usr/bin/env python3
"""Independently grade one VLLM #64830 Development candidate."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Any

try:
    from .pinned_workspace_git import pinned_git, read_regular_source
except ImportError:
    from pinned_workspace_git import pinned_git, read_regular_source

try:
    from .prepare_mlsys_vllm_64830_qualification_overlay import extract_class_method
except ImportError:
    from prepare_mlsys_vllm_64830_qualification_overlay import extract_class_method


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(workspace: Path, *args: str) -> bytes:
    return pinned_git(workspace, *args)


def candidate_status(workspace: Path) -> tuple[str, ...]:
    paths: set[str] = set()
    for args in (
        ("diff", "--name-only", "HEAD", "--"),
        ("ls-files", "--others", "--exclude-standard"),
        ("ls-files", "--others", "--ignored", "--exclude-standard"),
    ):
        output = git_output(workspace, *args).decode("utf-8").splitlines()
        paths.update(path for path in output if path)
    return tuple(sorted(paths))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_pytest(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> tuple[int | None, bytes, bool]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        try:
            output, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = process.communicate()
        return process.returncode, output, timed_out
    finally:
        # The grader must not leave a candidate-controlled descendant running.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _pytest_counts(output: str) -> dict[str, int] | None:
    summary_lines = [
        line.strip().strip("= ").strip()
        for line in output.splitlines()
        if re.search(r"\d+\s+(?:passed|failed|errors?|skipped)\b", line)
    ]
    if not summary_lines:
        return None
    counts: dict[str, int] = {}
    for count, label in re.findall(
        r"\b(\d+)\s+(passed|failed|errors?|skipped)\b", summary_lines[-1]
    ):
        canonical = "errors" if label in {"error", "errors"} else label
        counts[canonical] = int(count)
    return counts


def grade_candidate(
    *,
    workspace: Path,
    baseline_revision: str,
    baseline_source_sha256: str,
    source_relative_path: str,
    test_file: Path,
    test_file_sha256: str,
    pytest_config: Path,
    pytest_config_sha256: str,
    method_worker: Path,
    method_worker_sha256: str,
    method_profile: Path | None,
    method_profile_sha256: str | None,
    python: Path,
    output_dir: Path,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    workspace = workspace.resolve(strict=True)
    test_file = test_file.resolve(strict=True)
    pytest_config = pytest_config.resolve(strict=True)
    method_worker = method_worker.resolve(strict=True)
    if method_profile is not None:
        method_profile = method_profile.resolve(strict=True)
    python = python.resolve(strict=True)
    source_relative = Path(source_relative_path)
    if source_relative.is_absolute() or ".." in source_relative.parts:
        raise ValueError("candidate source path is not a safe relative path")
    source_candidate = workspace / source_relative
    if source_candidate.is_symlink():
        raise ValueError("candidate source path must not be a symlink")
    trusted_paths = [test_file, pytest_config, method_worker, python]
    if method_profile is not None:
        trusted_paths.append(method_profile)
    for trusted_path in trusted_paths:
        if trusted_path == workspace or workspace in trusted_path.parents:
            raise ValueError("grader inputs and runtime must be outside the candidate workspace")
    output_parent = output_dir.parent.resolve(strict=True)
    output_dir = output_parent / output_dir.name
    if output_dir == workspace or workspace in output_dir.parents or output_dir in workspace.parents:
        raise ValueError("grader output must be disjoint from the candidate workspace")
    if output_dir.is_symlink() or output_dir.exists():
        raise FileExistsError(f"refusing to overwrite grader output: {output_dir}")
    output_dir.mkdir(parents=True)
    home_dir = output_dir / "home"
    temp_dir = output_dir / "tmp"
    base_temp_dir = output_dir / "pytest-tmp"
    for directory in (home_dir, temp_dir, base_temp_dir):
        directory.mkdir()

    expected_head = baseline_revision
    actual_head = git_output(workspace, "rev-parse", "HEAD").decode().strip()
    baseline_source = git_output(
        workspace, "show", f"{baseline_revision}:{source_relative_path}"
    )
    baseline_sha = sha256_bytes(baseline_source)
    candidate_source = read_regular_source(workspace, source_relative_path)
    source_sha = sha256_bytes(candidate_source)
    changed_paths = candidate_status(workspace)
    diff_bytes = git_output(
        workspace,
        "diff",
        "--no-ext-diff",
        "--binary",
        "--no-color",
        baseline_revision,
        "--",
        source_relative_path,
    )
    diff_path = output_dir / "candidate.diff"
    diff_path.write_bytes(diff_bytes)

    result: dict[str, Any] = {
        "schema_version": 1,
        "grader_id": "mlsys-vllm-64830-agent-independent-grader-v1",
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "workspace": str(workspace),
        "source_relative_path": source_relative_path,
        "baseline_revision": baseline_revision,
        "expected_head": expected_head,
        "actual_head": actual_head,
        "baseline_source_sha256": baseline_sha,
        "expected_baseline_source_sha256": baseline_source_sha256,
        "candidate_source_sha256": source_sha,
        "candidate_diff_sha256": sha256_bytes(diff_bytes),
        "candidate_diff_path": str(diff_path),
        "candidate_diff_bytes": len(diff_bytes),
        "changed_paths": list(changed_paths),
        "grader_call_count": 0,
        "feedback_to_model": False,
        "classification": "NOT_GRADED",
        "test_file_sha256": sha256_file(test_file),
        "expected_test_file_sha256": test_file_sha256,
        "pytest_config_sha256": sha256_file(pytest_config),
        "expected_pytest_config_sha256": pytest_config_sha256,
        "method_worker_sha256": sha256_file(method_worker),
        "expected_method_worker_sha256": method_worker_sha256,
        "method_profile_sha256": sha256_file(method_profile) if method_profile else None,
        "expected_method_profile_sha256": method_profile_sha256,
    }

    if result["test_file_sha256"] != test_file_sha256:
        workspace_errors = ["trusted test overlay hash mismatch"]
    else:
        workspace_errors = []
    if result["pytest_config_sha256"] != pytest_config_sha256:
        workspace_errors.append("pytest configuration hash mismatch")
    if result["method_worker_sha256"] != method_worker_sha256:
        workspace_errors.append("candidate method worker hash mismatch")
    if method_profile is not None:
        if method_profile_sha256 is None or result["method_profile_sha256"] != method_profile_sha256:
            workspace_errors.append("candidate method sandbox profile hash mismatch")
    elif method_profile_sha256 is not None:
        workspace_errors.append("method sandbox profile is missing")

    if actual_head != expected_head:
        workspace_errors.append("candidate workspace HEAD differs from frozen baseline")
    if baseline_sha != baseline_source_sha256:
        workspace_errors.append("frozen baseline source hash mismatch")
    if changed_paths and set(changed_paths) != {source_relative_path}:
        workspace_errors.append("candidate modified files outside the one-file allowlist")
    if workspace_errors:
        result["workspace_errors"] = workspace_errors
        result["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _write_json(output_dir / "grade.json", result)
        return result

    try:
        method_source = extract_class_method(
            candidate_source.decode("utf-8"),
            "VllmConfig",
            "_get_v1_model_runner_unsupported_features",
        )
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        result.update(
            {
                "classification": "INVALID_CANDIDATE_SOURCE",
                "candidate_source_error": f"{type(error).__name__}: {error}",
                "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )
        _write_json(output_dir / "grade.json", result)
        return result
    method_file = output_dir / "candidate_method.py"
    method_file.write_text(method_source, encoding="utf-8")
    log_path = output_dir / "pytest.log"
    command = [
        str(python),
        "-B",
        "-m",
        "pytest",
        "-c",
        str(pytest_config),
        "-q",
        "-vv",
        "--color=no",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(base_temp_dir),
        "-o",
        f"log_file={log_path}",
        str(test_file),
    ]
    environment = {
        "HOME": str(home_dir),
        "LANG": "C.UTF-8",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(temp_dir),
        "VARE_METHOD_FILE": str(method_file),
        "VARE_METHOD_WORKER": str(method_worker),
    }
    if method_profile is not None:
        environment["VARE_METHOD_PROFILE"] = str(method_profile)
        environment["VARE_SANDBOX_EXEC"] = "/usr/bin/sandbox-exec"
    try:
        returncode, console, timed_out = _run_pytest(
            command,
            cwd=output_dir,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
        console_path = output_dir / "pytest-console.log"
        console_path.write_bytes(console)
        console_text = console.decode("utf-8", errors="replace")
        summary = _pytest_counts(console_text)
        passed = (
            returncode == 0
            and summary == {"passed": 2}
            and not timed_out
        )
        target_assertions = re.findall(
            r"^E\s+(?:AssertionError:\s*)?assert .*== \['(?:HiSparse|watermarking)'\]$",
            console_text,
            re.MULTILINE,
        )
        semantic_failure = (
            returncode == 1
            and not timed_out
            and summary is not None
            and summary.get("failed", 0) > 0
            and summary.get("failed", 0) == len(target_assertions)
            and summary.get("errors", 0) == 0
            and summary.get("skipped", 0) == 0
        )
        classification = (
            "TARGET_GATE_PASSED"
            if passed
            else "TARGET_GATE_FAILED"
            if semantic_failure
            else "GRADER_ERROR"
        )
        result.update(
            {
                "grader_call_count": 1,
                "pytest_command": command,
                "pytest_returncode": returncode,
                "pytest_summary": summary,
                "pytest_console_sha256": sha256_bytes(console),
                "pytest_log_sha256": sha256_file(log_path) if log_path.is_file() else None,
                "candidate_method_sha256": sha256_file(method_file),
                "classification": "INCONCLUSIVE_TIMEOUT" if timed_out else classification,
            }
        )
    except OSError as error:
        result.update(
            {
                "grader_call_count": 1,
                "pytest_command": command,
                "grader_error": f"{type(error).__name__}: {error}",
                "classification": "GRADER_ERROR",
            }
        )
    result["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_json(output_dir / "grade.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--baseline-source-sha256", required=True)
    parser.add_argument("--source-relative-path", required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--test-file-sha256", required=True)
    parser.add_argument("--pytest-config", type=Path, required=True)
    parser.add_argument("--pytest-config-sha256", required=True)
    parser.add_argument("--method-worker", type=Path, required=True)
    parser.add_argument("--method-worker-sha256", required=True)
    parser.add_argument("--method-profile", type=Path)
    parser.add_argument("--method-profile-sha256")
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    result = grade_candidate(
        workspace=args.workspace,
        baseline_revision=args.baseline_revision,
        baseline_source_sha256=args.baseline_source_sha256,
        source_relative_path=args.source_relative_path,
        test_file=args.test_file,
        test_file_sha256=args.test_file_sha256,
        pytest_config=args.pytest_config,
        pytest_config_sha256=args.pytest_config_sha256,
        method_worker=args.method_worker,
        method_worker_sha256=args.method_worker_sha256,
        method_profile=args.method_profile,
        method_profile_sha256=args.method_profile_sha256,
        python=args.python,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["classification"] == "TARGET_GATE_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
