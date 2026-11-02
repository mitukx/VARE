#!/usr/bin/env python3
"""Independent, offline, source-bound grader for the MLX-LM MoE task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any


BASELINE = "77c33b14373ac70d7abd6f82af15962852adadbb"
EDITABLE = (
    "mlx_lm/models/granitemoe.py",
    "mlx_lm/models/granitemoehybrid.py",
    "mlx_lm/models/lfm2_moe.py",
)
TEST_SELECTORS = (
    "TestTunerTrainer::test_granitemoe_backward",
    "TestTunerTrainer::test_granitemoehybrid_backward",
    "TestTunerTrainer::test_lfm2_moe_backward",
)
MAX_WALL_SECONDS = 240
MAX_RSS_BYTES = 8 * 1024**3
MAX_OUTPUT_BYTES = 1_048_576


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash a directory's relative paths, file contents and symlink targets."""
    digest = hashlib.sha256()
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            item = b"L\0" + os.readlink(path).encode("utf-8")
        elif path.is_file():
            item = b"F\0" + bytes.fromhex(sha256_file(path))
        elif path.is_dir():
            item = b"D\0"
        else:
            raise ValueError(f"unsupported filesystem object in source tree: {path}")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(item)
    return digest.hexdigest()


def _load_lock(task_root: Path) -> tuple[dict[str, Any], str]:
    grader_file = Path(__file__).resolve()
    lock_path = task_root / "protocol.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(grader_file) != lock["grader"]["sha256"]:
        raise ValueError("independent grader source hash mismatch")
    if lock["source"]["baseline_revision"] != BASELINE:
        raise ValueError("protocol baseline revision mismatch")
    if tuple(lock["grader"]["selectors"]) != TEST_SELECTORS:
        raise ValueError("protocol test-selector mismatch")
    repository_root = task_root.parents[2]
    runner = (repository_root / lock["calibration"]["runner_path"]).resolve(strict=True)
    if repository_root not in runner.parents:
        raise ValueError("calibration runner path escapes the VARE repository")
    if sha256_file(runner) != lock["calibration"]["runner_sha256"]:
        raise ValueError("calibration runner source hash mismatch")
    return lock, sha256_file(lock_path)


def _git(workspace: Path, *args: str) -> str:
    git = "/usr/bin/git" if Path("/usr/bin/git").exists() else shutil.which("git")
    if not git:
        raise RuntimeError("git is unavailable")
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    result = subprocess.run(
        [
            git,
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.pager=cat",
            "-C",
            str(workspace),
            *args,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        timeout=30,
    )
    return result.stdout.strip()


def changed_paths(workspace: Path) -> tuple[str, ...]:
    tracked = _git(
        workspace,
        "diff",
        "--name-only",
        "--no-ext-diff",
        "--no-renames",
        BASELINE,
        "--",
    )
    untracked = _git(workspace, "ls-files", "--others", "--exclude-standard")
    paths = {path for path in (*tracked.splitlines(), *untracked.splitlines()) if path}
    return tuple(sorted(paths))


def _all_links(root: Path) -> list[str]:
    links: list[str] = []
    for path in root.rglob("*"):
        if path == root / ".git" or root / ".git" in path.parents:
            continue
        if path.is_symlink():
            links.append(path.relative_to(root).as_posix())
    return sorted(links)


def _quote(path: Path) -> str:
    value = str(path.resolve())
    if any(char in value for char in ('"', "\\", "\n", "\r")):
        raise ValueError("sandbox path contains unsupported characters")
    return json.dumps(value)


def _metal_cache_paths() -> tuple[Path, ...]:
    host_tmp = Path(tempfile.gettempdir()).resolve()
    cache_root = host_tmp.parent / "C"
    return tuple(
        path
        for path in (
            cache_root / "com.apple.metalfe",
            cache_root / "com.apple.gpuarchiver",
        )
        if path.is_dir()
    )


def sandbox_profile(
    *,
    candidate: Path,
    trusted_source: Path,
    grader_test: Path,
    runtime_root: Path,
    scratch: Path,
    python_prefix: Path,
    metal_cache_paths: tuple[Path, ...],
) -> str:
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process*)",
        "(allow sysctl-read)",
        "(deny network*)",
        '(allow file-read* (literal "/dev/null"))',
        '(allow file-write* (literal "/dev/null"))',
    ]
    for path in (
        Path("/System"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/Library/Developer/CommandLineTools"),
        python_prefix,
        runtime_root,
        candidate,
        trusted_source,
        grader_test.parent,
        scratch,
    ):
        if path.exists():
            lines.append(f"(allow file-read* (subpath {_quote(path)}))")
    lines.append(f"(allow file-write* (subpath {_quote(scratch)}))")
    for cache_path in metal_cache_paths:
        lines.append(
            "(allow file-issue-extension (require-all "
            '(extension-class "com.apple.app-sandbox.read-write") '
            f"(subpath {_quote(cache_path)})))"
        )
    return "\n".join(lines) + "\n"


def process_tree_rss(root_pid: int) -> int:
    output = subprocess.check_output(["/bin/ps", "-axo", "pid=,ppid=,rss="], text=True)
    rows: dict[int, int] = {}
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        pid, ppid, rss_kib = map(int, fields)
        rows[pid] = rss_kib * 1024
        children.setdefault(ppid, []).append(pid)
    pending = [root_pid]
    seen: set[int] = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rows.get(pid, 0)
        pending.extend(children.get(pid, []))
    return total


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _read_output(pipe, chunks: list[bytes], state: dict[str, Any]) -> None:
    total = 0
    while True:
        block = pipe.read(65536)
        if not block:
            break
        remaining = MAX_OUTPUT_BYTES - total
        if remaining > 0:
            chunks.append(block[:remaining])
        total += len(block)
        if total > MAX_OUTPUT_BYTES:
            state["overflow"] = True
    state["total_bytes"] = total


def _parse_pytest_counts(output: bytes) -> dict[str, int]:
    counts: dict[str, int] = {}
    for number, raw_label in re.findall(
        rb"\b(\d+)\s+(passed|failed|errors?|skipped)\b", output
    ):
        label = raw_label.decode("ascii")
        key = "errors" if label in {"error", "errors"} else label
        counts[key] = int(number)
    return counts


def _runtime_versions(python: Path, pythonpath: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json,platform,sys; "
        "names={'pytest':'pytest','mlx':'mlx','mlx_lm':'mlx-lm','numpy':'numpy'}; "
        "out={'python':platform.python_version()}; "
        "[out.__setitem__(k,m.version(v)) for k,v in names.items()]; "
        "print(json.dumps(out,sort_keys=True))"
    )
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": str(pythonpath),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [str(python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    return json.loads(result.stdout)


def _validate_candidate(
    workspace: Path,
    expected_base_hashes: dict[str, str],
    *,
    allow_unmodified_baseline: bool,
) -> tuple[tuple[str, ...], dict[str, str]]:
    if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError(
            "macOS sandbox-exec is unavailable; refusing to grade source"
        )
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("candidate workspace must be a real directory")
    if (workspace / ".git").is_symlink() or not (workspace / ".git").is_dir():
        raise ValueError("candidate git metadata is missing or symlinked")
    if _git(workspace, "rev-parse", "HEAD") != BASELINE:
        raise ValueError("candidate workspace HEAD is not the pinned baseline")
    if _git(workspace, "rev-parse", "--is-shallow-repository") != "true":
        raise ValueError("candidate workspace is not shallow")
    if _git(workspace, "remote"):
        raise ValueError("candidate workspace still has a Git remote")
    paths = changed_paths(workspace)
    if not paths and not allow_unmodified_baseline:
        raise ValueError("candidate contains no source changes")
    if not set(paths).issubset(set(EDITABLE)):
        raise ValueError(f"candidate changed paths outside editable allowlist: {paths}")
    if _all_links(workspace):
        raise ValueError("candidate workspace contains symlinks outside Git metadata")
    hashes: dict[str, str] = {}
    for relative in EDITABLE:
        path = workspace / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"candidate source is missing or not a regular file: {relative}"
            )
        hashes[relative] = sha256_file(path)
        if relative not in paths and hashes[relative] != expected_base_hashes[relative]:
            raise ValueError(f"unreported candidate source change: {relative}")
    return paths, hashes


def _write_profile(profile_path: Path, text: str) -> None:
    profile_path.write_text(text, encoding="utf-8")


def grade_candidate(
    *,
    task_root: Path,
    source_manifest_path: Path,
    workspace: Path,
    trusted_baseline: Path,
    grader_test: Path,
    python: Path,
    pythonpath: Path,
    output_dir: Path,
    expected_arm: str = "candidate",
) -> dict[str, Any]:
    task_root = task_root.resolve(strict=True)
    source_manifest_path = source_manifest_path.resolve(strict=True)
    if workspace.expanduser().is_symlink():
        raise ValueError("candidate workspace must not be a symlink")
    workspace = workspace.expanduser().resolve(strict=True)
    trusted_baseline = trusted_baseline.resolve(strict=True)
    grader_test = grader_test.resolve(strict=True)
    python = python.resolve(strict=True)
    pythonpath = pythonpath.resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite grader evidence: {output_dir}")

    lock, lock_hash = _load_lock(task_root)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest["baseline_revision"] != BASELINE:
        raise ValueError("prepared source manifest baseline mismatch")
    if source_manifest["protocol_lock_sha256"] != lock_hash:
        raise ValueError(
            "prepared source manifest is bound to a different grader protocol"
        )
    if (
        source_manifest["task_json_sha256"]
        != lock["task_definition"]["task_json_sha256"]
    ):
        raise ValueError("prepared source manifest task.json hash mismatch")
    if source_manifest["task_md_sha256"] != lock["task_definition"]["task_md_sha256"]:
        raise ValueError("prepared source manifest TASK.md hash mismatch")
    if source_manifest["candidate_visible_files"] != list(EDITABLE):
        raise ValueError("prepared candidate allowlist mismatch")
    if source_manifest["trusted_baseline_source"] != str(trusted_baseline):
        raise ValueError("trusted baseline path does not match prepared manifest")
    if source_manifest["trusted_grader_test"] != str(grader_test):
        raise ValueError("trusted grader test path does not match prepared manifest")
    if source_manifest["grader_test_sha256"] != lock["grader"]["upstream_test_sha256"]:
        raise ValueError("prepared source manifest test hash mismatch")
    if source_manifest["license_sha256"] != lock["source"]["license_file_sha256"]:
        raise ValueError("prepared source manifest license hash mismatch")
    if (
        source_manifest["baseline_file_sha256"]
        != lock["source"]["baseline_file_sha256"]
    ):
        raise ValueError("prepared baseline file hashes differ from the protocol lock")
    if sha256_file(grader_test) != lock["grader"]["upstream_test_sha256"]:
        raise ValueError("trusted grader test hash mismatch")
    if (grader_test.parent / ".git").exists():
        raise ValueError("grader test directory contains Git metadata")
    if _git(trusted_baseline, "rev-parse", "HEAD") != BASELINE:
        raise ValueError("trusted source tree is not the pinned baseline")
    if _git(trusted_baseline, "remote"):
        raise ValueError("trusted source tree still has a Git remote")
    if _git(trusted_baseline, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("trusted source tree is not clean")
    if (
        sha256_file(trusted_baseline / "LICENSE")
        != lock["source"]["license_file_sha256"]
    ):
        raise ValueError("trusted source license hash mismatch")
    if _git(workspace, "rev-parse", "HEAD") != BASELINE:
        raise ValueError("candidate source tree is not the pinned baseline")
    expected_base_hashes = lock["source"]["baseline_file_sha256"]
    for relative, expected_hash in expected_base_hashes.items():
        if sha256_file(trusted_baseline / relative) != expected_hash:
            raise ValueError(f"trusted baseline file hash mismatch: {relative}")

    paths, candidate_hashes = _validate_candidate(
        workspace,
        expected_base_hashes,
        allow_unmodified_baseline=expected_arm == "baseline",
    )
    roots_to_separate = (
        task_root,
        source_manifest_path,
        trusted_baseline,
        grader_test,
        python,
        pythonpath,
    )
    if any(
        path == workspace or workspace in path.parents for path in roots_to_separate
    ):
        raise ValueError(
            "a trusted grader input or runtime is inside the candidate workspace"
        )
    if output_dir == workspace or workspace in output_dir.parents:
        raise ValueError(
            "grader evidence output must be outside the candidate workspace"
        )
    versions = _runtime_versions(python, pythonpath)
    expected_versions = lock["runtime"]
    if (
        versions["python"] != expected_versions["python"]
        or versions["pytest"] != expected_versions["pytest"]
    ):
        raise RuntimeError(f"Python/pytest runtime mismatch: {versions}")
    if (
        versions["mlx"] != expected_versions["mlx"]
        or versions["mlx_lm"] != expected_versions["mlx_lm"]
    ):
        raise RuntimeError(f"MLX runtime mismatch: {versions}")

    output_dir.mkdir(parents=True)
    scratch = output_dir / "sandbox-scratch"
    scratch.mkdir()
    for directory in ("home", "tmp", "pytest-tmp"):
        (scratch / directory).mkdir()
    grader_root = output_dir / "grader-source"
    grader_root.mkdir()
    shutil.copytree(trusted_baseline / "mlx_lm", grader_root / "mlx_lm")
    for relative in EDITABLE:
        destination = grader_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / relative, destination)
    for path in grader_root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    grader_root.chmod(0o555)

    selectors = [f"{grader_test}::{name}" for name in TEST_SELECTORS]
    command = [
        "/usr/bin/sandbox-exec",
        "-f",
        str((output_dir / "sandbox.sb").resolve()),
        str(python),
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:logging",
        "--basetemp",
        str(scratch / "pytest-tmp"),
        *selectors,
    ]
    cache_paths = _metal_cache_paths()
    profile = sandbox_profile(
        candidate=workspace,
        trusted_source=grader_root,
        grader_test=grader_test,
        runtime_root=pythonpath,
        scratch=scratch,
        python_prefix=Path(sys.prefix),
        metal_cache_paths=cache_paths,
    )
    _write_profile(output_dir / "sandbox.sb", profile)
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(scratch / "home"),
        "TMPDIR": str(scratch / "tmp"),
        "PYTHONPATH": os.pathsep.join((str(grader_root), str(pythonpath))),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "PIP_NO_INDEX": "1",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }
    stdout_chunks: list[bytes] = []
    reader_state: dict[str, Any] = {}
    started = time.monotonic()
    peak_rss = 0
    timed_out = False
    rss_exceeded = False
    output_exceeded = False
    monitor_error = None
    proc = subprocess.Popen(
        command,
        cwd=grader_root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert proc.stdout is not None
    reader = threading.Thread(
        target=_read_output,
        args=(proc.stdout, stdout_chunks, reader_state),
        daemon=True,
    )
    reader.start()
    while proc.poll() is None:
        try:
            peak_rss = max(peak_rss, process_tree_rss(proc.pid))
        except Exception as exc:
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
        if time.monotonic() - started > MAX_WALL_SECONDS:
            timed_out = True
            _terminate(proc)
            break
        time.sleep(0.2)
    proc.wait()
    _kill_process_group(proc)
    reader.join(timeout=5)
    output = b"".join(stdout_chunks)
    (output_dir / "pytest.log").write_bytes(output)
    counts = _parse_pytest_counts(output)

    infrastructure_ok = not (
        timed_out or rss_exceeded or output_exceeded or monitor_error
    )
    passed = (
        infrastructure_ok
        and proc.returncode == 0
        and counts == {"passed": len(TEST_SELECTORS)}
    )
    result = {
        "schema_version": 1,
        "task_id": lock["task_id"],
        "arm": expected_arm,
        "passed": passed,
        "grader_status": "PASS"
        if passed
        else ("TEST_FAIL" if infrastructure_ok else "INFRASTRUCTURE_FAIL"),
        "pytest_returncode": proc.returncode,
        "pytest_counts": counts,
        "selected_test_ids": list(TEST_SELECTORS),
        "candidate_revision": _git(workspace, "rev-parse", "HEAD"),
        "changed_paths": list(paths),
        "candidate_source_sha256": candidate_hashes,
        "trusted_baseline_revision": _git(trusted_baseline, "rev-parse", "HEAD"),
        "trusted_test_sha256": sha256_file(grader_test),
        "protocol_lock_sha256": lock_hash,
        "grader_sha256": sha256_file(Path(__file__).resolve()),
        "python_sha256": sha256_file(python),
        "sandbox_profile_sha256": sha256_file(output_dir / "sandbox.sb"),
        "pytest_log_sha256": sha256_file(output_dir / "pytest.log"),
        "runtime_versions": versions,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "sampled_process_tree_peak_rss_bytes": peak_rss,
        "process_tree_rss_limit_bytes": MAX_RSS_BYTES,
        "timed_out": timed_out,
        "rss_limit_exceeded": rss_exceeded,
        "output_limit_exceeded": output_exceeded,
        "monitor_error": monitor_error,
        "network_denied": True,
        "grader_test_outside_candidate_workspace": workspace not in grader_test.parents,
    }
    (output_dir / "grade.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Only the trusted parent-facing summary is printed; raw test output stays in its record.
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "task_id",
                    "arm",
                    "passed",
                    "grader_status",
                    "pytest_returncode",
                    "pytest_counts",
                    "elapsed_seconds",
                    "sampled_process_tree_peak_rss_bytes",
                )
                if key in result
            },
            sort_keys=True,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--trusted-baseline", type=Path, required=True)
    parser.add_argument("--grader-test", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--pythonpath", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="candidate")
    args = parser.parse_args()
    try:
        result = grade_candidate(
            task_root=args.task_root,
            source_manifest_path=args.source_manifest,
            workspace=args.workspace,
            trusted_baseline=args.trusted_baseline,
            grader_test=args.grader_test,
            python=args.python,
            pythonpath=args.pythonpath,
            output_dir=args.output,
            expected_arm=args.arm,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "grader_status": "INFRASTRUCTURE_FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        )
        return 2
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
