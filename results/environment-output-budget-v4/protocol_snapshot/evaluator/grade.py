#!/usr/bin/env python3
"""Check stdout and stderr budgets independently and verify POSIX cleanup."""
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

    # Load only the declared candidate files, avoiding the evaluator host's
    # editable install or other import paths.
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

    def flood_command(stream: str, marker: Path | None = None) -> CommandSpec:
        child = (
            "import pathlib,sys,time; time.sleep(.5); "
            "pathlib.Path(sys.argv[1]).write_text('alive')"
        )
        parent = (
            "import subprocess,sys; "
            "marker=sys.argv[2]; "
            "subprocess.Popen([sys.executable,'-c',sys.argv[3],marker]) if marker != '-' else None; "
            "writer=sys.stdout if sys.argv[1]=='stdout' else sys.stderr; "
            "[(writer.write('z'*8192),writer.flush()) for _ in range(256)]"
        )
        return CommandSpec(
            argv=(sys.executable, "-c", parent, stream, "-" if marker is None else str(marker), child),
            name=f"{stream}-flood",
        )

    descendant_survived = False
    with tempfile.TemporaryDirectory(prefix="vare-output-budget-v3-") as temporary:
        marker = Path(temporary) / "descendant-finished"
        stdout_flood = run_command(workspace, flood_command("stdout", marker), limits)
        if os.name == "posix":
            time.sleep(.6)
            descendant_survived = marker.exists()
        stderr_flood = run_command(workspace, flood_command("stderr"), limits)
        normal = run_command(
            workspace,
            CommandSpec(argv=(sys.executable, "-c", "print('ok')"), name="normal"),
            limits,
        )

    stdout_bytes = len(stdout_flood.stdout.encode("utf-8"))
    stdout_stderr_bytes = len(stdout_flood.stderr.encode("utf-8"))
    stderr_stdout_bytes = len(stderr_flood.stdout.encode("utf-8"))
    stderr_bytes = len(stderr_flood.stderr.encode("utf-8"))
    checks = {
        "stdout_overflow_rejected": getattr(stdout_flood, "output_limited", False) and not stdout_flood.passed,
        "stdout_capture_bounded": stdout_bytes <= limit and stdout_stderr_bytes <= limit,
        "stdout_overflow_not_timeout": not stdout_flood.timed_out,
        "stderr_overflow_rejected": getattr(stderr_flood, "output_limited", False) and not stderr_flood.passed,
        "stderr_capture_bounded": stderr_stdout_bytes <= limit and stderr_bytes <= limit,
        "stderr_overflow_not_timeout": not stderr_flood.timed_out,
        "normal_command_passes": normal.passed and normal.stdout == "ok\n",
        "normal_not_limited": not getattr(normal, "output_limited", False),
        "posix_descendant_stopped": os.name != "posix" or not descendant_survived,
    }
    result = {
        "schema_version": 1,
        "task_id": "vare-environment-command-output-budget-v4",
        "passed": all(checks.values()),
        "checks": checks,
        "limit_bytes_per_stream": limit,
        "stdout_overflow": {
            "returncode": stdout_flood.returncode,
            "timed_out": stdout_flood.timed_out,
            "output_limited": getattr(stdout_flood, "output_limited", False),
            "stdout_bytes_retained": stdout_bytes,
            "stderr_bytes_retained": stdout_stderr_bytes,
        },
        "stderr_overflow": {
            "returncode": stderr_flood.returncode,
            "timed_out": stderr_flood.timed_out,
            "output_limited": getattr(stderr_flood, "output_limited", False),
            "stdout_bytes_retained": stderr_stdout_bytes,
            "stderr_bytes_retained": stderr_bytes,
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
