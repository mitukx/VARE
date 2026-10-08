#!/usr/bin/env python3
"""Check bounded command output, overflow classification, and POSIX cleanup."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    # Load only the declared candidate files. A checkout's editable install or
    # namespace-package fallback must not silently substitute the evaluator host's
    # currently installed VARE source for the candidate revision.
    source_root = workspace / "src" / "vare"
    vare_package = ModuleType("vare")
    vare_package.__path__ = [str(source_root)]
    environments_package = ModuleType("vare.environments")
    environments_package.__path__ = [str(source_root / "environments")]
    sys.modules["vare"] = vare_package
    sys.modules["vare.environments"] = environments_package

    def load_candidate(name: str, relative_path: str):
        path = source_root / relative_path
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load candidate module: {relative_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError(f"candidate import escaped workspace: {name}")
        return module

    load_candidate("vare.evidence", "evidence.py")
    spec_module = load_candidate("vare.environments.spec", "environments/spec.py")
    runner_module = load_candidate("vare.environments.runner", "environments/runner.py")
    run_command = runner_module.run_command
    CommandSpec = spec_module.CommandSpec
    ResourceLimits = spec_module.ResourceLimits

    limit = 128
    limits = ResourceLimits(
        wall_time_s=3,
        cpu_time_s=None,
        memory_mb=None,
        max_output_bytes=limit,
    )
    with tempfile.TemporaryDirectory(prefix="vare-output-budget-") as temporary:
        marker = Path(temporary) / "descendant-finished"
        child = (
            "import pathlib,sys,time; time.sleep(.5); "
            "pathlib.Path(sys.argv[1]).write_text('alive')"
        )
        parent = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); "
            "sys.stdout.write('x'*1048576); sys.stdout.flush(); "
            "sys.stderr.write('y'*1048576); sys.stderr.flush()"
        )
        overflow = run_command(
            workspace,
            CommandSpec(
                argv=(sys.executable, "-c", parent, child, str(marker)),
                name="output-flood",
            ),
            limits,
        )
        if os.name == "posix":
            time.sleep(.6)
        normal = run_command(
            workspace,
            CommandSpec(argv=(sys.executable, "-c", "print('ok')"), name="normal"),
            limits,
        )

    overflow_flag = getattr(overflow, "output_limited", False)
    retained_stdout = len(overflow.stdout.encode("utf-8"))
    retained_stderr = len(overflow.stderr.encode("utf-8"))
    checks = {
        "overflow_not_passed": not overflow.passed,
        "overflow_flag": overflow_flag,
        "not_timeout": not overflow.timed_out,
        "stdout_bounded": retained_stdout <= limit,
        "stderr_bounded": retained_stderr <= limit,
        "normal_command_passes": normal.passed and normal.stdout == "ok\n",
        "normal_not_limited": not getattr(normal, "output_limited", False),
        "posix_child_reaped": os.name != "posix" or not marker.exists(),
    }
    result = {
        "schema_version": 1,
        "task_id": "vare-environment-command-output-budget-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "overflow": {
            "returncode": overflow.returncode,
            "timed_out": overflow.timed_out,
            "output_limited": overflow_flag,
            "stdout_bytes_retained": retained_stdout,
            "stderr_bytes_retained": retained_stderr,
        },
        "normal": {
            "returncode": normal.returncode,
            "passed": normal.passed,
            "stdout": normal.stdout,
        },
        "claim_limit": (
            "One deterministic local CPU resource-control regression; not a hostile-code "
            "sandbox or hard memory quota."
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
