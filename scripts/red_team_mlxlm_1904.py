#!/usr/bin/env python3
"""Run bounded hostile candidates against the isolated #1904 function grader."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from vare.mlsys_repo_task import (
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
        "experiment_id": "mlsys-repository-task-mlxlm-1904-red-team-v1",
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
    report["decision"] = {
        "all_attacks_checked": all(item["audit_check_passed"] for item in report["attempts"].values()),
        "claim_limit": "Focused function-callee checks in one pinned Colima Docker setup; not a general sandbox certification.",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "decision": report["decision"]}, sort_keys=True))
    return 0 if report["decision"]["all_attacks_checked"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
