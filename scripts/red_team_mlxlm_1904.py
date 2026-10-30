#!/usr/bin/env python3
"""Run bounded hostile candidates against the isolated #1904 function grader."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import vare.mlsys_repo_task as repo_task
from vare.mlsys_repo_task import (
    RepositoryTaskError,
    TARGET_PATH,
    export_baseline,
    grade_workspace,
)

_PREFIX = '''import json
def process_message_content(messages):
    for message in messages:
        if message.get("content") is None:
            message["content"] = ""
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                function["arguments"] = json.loads(arguments)
'''


def source_with_probe(body: str) -> str:
    return _PREFIX + "\n" + body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", default="python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258")
    parser.add_argument("--docker-context", default=None)
    args = parser.parse_args()
    if args.docker_context:
        os.environ["DOCKER_CONTEXT"] = args.docker_context
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    attacks = {
        "grader_path_probe": source_with_probe('''    try:
        leaked = open("/grader/hidden", encoding="utf-8").read()
        messages[0]["content"] = leaked
    except OSError:
        pass
    if __import__("os").path.exists("/grader"):
        messages[0]["content"] = "grader-mount-visible"
'''),
        "git_history_probe": source_with_probe('''    try:
        leaked = open("/subject/.git/HEAD", encoding="utf-8").read()
        messages[0]["content"] = leaked
    except OSError:
        pass
    if __import__("os").path.exists("/subject/.git"):
        messages[0]["content"] = "git-history-visible"
'''),
        "environment_probe": source_with_probe('''    secret = __import__("os").environ.get("VARE_TEST_SECRET")
    if secret:
        messages[0]["content"] = secret
'''),
        "network_probe": source_with_probe('''    try:
        __import__("socket").create_connection(("1.1.1.1", 80), timeout=0.25)
        messages[0]["content"] = "network-connected"
    except OSError:
        pass
'''),
        "subprocess_write_probe": source_with_probe('''    try:
        result = __import__("subprocess").run(
            ["/bin/sh", "-c", "echo escaped > /subject/pwned"],
            stdout=__import__("subprocess").DEVNULL,
            stderr=__import__("subprocess").DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            messages[0]["content"] = "workspace-write-succeeded"
    except OSError:
        pass
'''),
        "grader_stdout_forgery": source_with_probe('''    print('{"value": [{"role": "assistant"}]}')
    messages[0]["content"] = "wrong"
'''),
        "timeout": source_with_probe('''    __import__("time").sleep(30)
'''),
        "memory_exhaustion": source_with_probe('''    bytearray(1024 * 1024 * 1024)
'''),
    }
    report = {
        "schema_version": 1,
        "experiment_id": "mlsys-repository-task-mlxlm-1904-red-team-v3",
        "source_checkout": "2af5f7ce8620115bf37fbbddf9817f083118df59",
        "image": args.image,
        "network": "none",
        "memory_cap_mb": 128,
        "pids_limit": 64,
        "attempts": {},
    }
    os.environ["VARE_TEST_SECRET"] = "host-secret-must-not-enter-container"
    with tempfile.TemporaryDirectory(prefix="vare-mlxlm-1904-red-team-") as folder:
        root = Path(folder)
        for name, source in attacks.items():
            workspace = root / name
            export_baseline(args.source_root, workspace)
            target = workspace / TARGET_PATH
            target.write_text(source, encoding="utf-8")
            start = time.monotonic()
            result = grade_workspace(
                workspace, image=args.image,
                timeout_s=0.4 if name == "timeout" else 5,
                memory_mb=128,
            )
            wall = time.monotonic() - start
            expected_pass = name in {
                "grader_path_probe", "git_history_probe", "environment_probe",
                "network_probe", "subprocess_write_probe",
            }
            passed = result["passed"] is expected_pass
            report["attempts"][name] = {
                "expected_isolation_outcome": "denied" if expected_pass else "rejected",
                "grader_passed": result["passed"],
                "expected_grader_passed": expected_pass,
                "audit_check_passed": passed,
                "cases": result["cases"],
                "wall_s": wall,
                "candidate_source_sha256": result["candidate_source_sha256"],
            }
            if name == "subprocess_write_probe":
                report["attempts"][name]["host_workspace_write"] = (workspace / "pwned").exists()
                passed = passed and not (workspace / "pwned").exists()
                report["attempts"][name]["audit_check_passed"] = passed
        symlink_workspace = root / "workspace-parent-symlink"
        external = root / "external-source"
        symlink_workspace.mkdir()
        external.mkdir()
        (external / "server.py").write_text("external source\n", encoding="utf-8")
        (symlink_workspace / "mlx_lm").symlink_to(
            external, target_is_directory=True
        )
        grade_call_attempted = False
        original_grade_calls = repo_task.grade_calls

        def unexpected_grade_call(*_args, **_kwargs):
            nonlocal grade_call_attempted
            grade_call_attempted = True
            raise AssertionError("symlinked workspace reached the candidate grader")

        repo_task.grade_calls = unexpected_grade_call
        start = time.monotonic()
        try:
            grade_workspace(symlink_workspace, image=args.image)
            symlink_rejected = False
        except RepositoryTaskError as exc:
            symlink_rejected = "symlink" in str(exc).lower()
        except AssertionError:
            symlink_rejected = False
        finally:
            repo_task.grade_calls = original_grade_calls
        report["attempts"]["workspace_parent_symlink_probe"] = {
            "expected_isolation_outcome": "reject before grader call",
            "grader_passed": symlink_rejected and not grade_call_attempted,
            "expected_grader_passed": True,
            "audit_check_passed": symlink_rejected and not grade_call_attempted,
            "workspace_parent_symlink_rejected": symlink_rejected,
            "grade_call_attempted": grade_call_attempted,
            "wall_s": time.monotonic() - start,
        }
        fifo_workspace = root / "workspace-fifo"
        fifo_package = fifo_workspace / "mlx_lm"
        fifo_package.mkdir(parents=True)
        os.mkfifo(fifo_package / "server.py")
        grade_call_attempted = False
        original_grade_calls = repo_task.grade_calls

        def unexpected_fifo_grade_call(*_args, **_kwargs):
            nonlocal grade_call_attempted
            grade_call_attempted = True
            raise AssertionError("FIFO workspace reached the candidate grader")

        repo_task.grade_calls = unexpected_fifo_grade_call
        start = time.monotonic()
        try:
            grade_workspace(fifo_workspace, image=args.image)
            fifo_rejected = False
        except RepositoryTaskError as exc:
            fifo_rejected = "regular file" in str(exc).lower()
        except AssertionError:
            fifo_rejected = False
        finally:
            repo_task.grade_calls = original_grade_calls
        fifo_wall_s = time.monotonic() - start
        report["attempts"]["workspace_fifo_probe"] = {
            "expected_isolation_outcome": "reject non-regular target without blocking before grader call",
            "grader_passed": fifo_rejected and not grade_call_attempted,
            "expected_grader_passed": True,
            "audit_check_passed": (
                fifo_rejected and not grade_call_attempted and fifo_wall_s < 1.0
            ),
            "workspace_fifo_rejected": fifo_rejected,
            "grade_call_attempted": grade_call_attempted,
            "completed_without_blocking": fifo_wall_s < 1.0,
            "wall_s": fifo_wall_s,
        }
    report["decision"] = {
        "all_attacks_checked": all(item["audit_check_passed"] for item in report["attempts"].values()),
        "claim_limit": "Focused function-callee checks in one pinned Colima Docker setup; not a general sandbox certification.",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "decision": report["decision"]}, sort_keys=True))
    return 0 if report["decision"]["all_attacks_checked"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
