#!/usr/bin/env python3
"""Generate prospective v2 goldens from the pinned MLX-LM fixed source."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from vare.mlxlm_1971_repo_task import (  # noqa: E402
    FIXED_COMMIT,
    FIXED_SOURCE_SHA256,
    IMAGE,
    _copy_package,
    sha256_file,
)
from vare.mlxlm_1971_repo_task import CASE as CASE_TEMPLATE  # noqa: E402

SCALES = (0.25, 0.5, 0.8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default=IMAGE)
    args = parser.parse_args()
    if os.environ.get("DOCKER_CONTEXT") != "colima":
        raise SystemExit("DOCKER_CONTEXT=colima is required")
    if args.image != IMAGE:
        raise SystemExit("image differs from the pinned MLX-LM v1 runtime")
    source = args.fixed_source.resolve(strict=True)
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    target = source / "mlx_lm/models/mimo_v2_flash.py"
    if (
        head != FIXED_COMMIT
        or status
        or sha256_file(target) != FIXED_SOURCE_SHA256
        or target.is_symlink()
    ):
        raise SystemExit("fixed source checkout or target hash mismatch")
    if args.output.exists():
        raise SystemExit("golden output already exists")

    cache_root = Path.home() / ".cache/vare"
    cache_root.mkdir(parents=True, exist_ok=True)
    cases = []
    for scale in SCALES:
        case = {**CASE_TEMPLATE, "value_scale": scale}
        cases.append(case)

    with tempfile.TemporaryDirectory(prefix="mlxlm-1971-v2-golden-", dir=cache_root) as folder:
        staging = Path(folder)
        package = staging / "subject"
        _copy_package(source, package)
        bridge = ROOT / "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/grader_adapter_v2.py"
        source_code = """import json
from pathlib import Path
import sys
sys.path.insert(0, '/task')
from vare_task_adapter_v2 import run
cases = json.loads(Path('/task/cases.json').read_text())
print(json.dumps({'cases': [run(case) for case in cases]}, sort_keys=True, separators=(',', ':')))
"""
        task_dir = staging / "task"
        task_dir.mkdir()
        (task_dir / "generate.py").write_text(source_code, encoding="utf-8")
        (task_dir / "cases.json").write_text(json.dumps(cases, sort_keys=True), encoding="utf-8")
        shutil.copyfile(bridge, task_dir / "vare_task_adapter_v2.py")
        command = [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64",
            "--memory=1536m", "--cpus=1", "--user=65534:65534",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m",
            "-e", "PYTHONPATH=/subject",
            "-v", f"{package}:/subject:ro",
            "-v", f"{task_dir}:/task:ro",
            args.image, "python", "/task/generate.py",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f"golden generator container failed: {result.stderr}")
    parsed = json.loads(result.stdout)
    if len(parsed.get("cases", [])) != len(SCALES):
        raise SystemExit("reference output count mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(parsed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "scales": SCALES}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
