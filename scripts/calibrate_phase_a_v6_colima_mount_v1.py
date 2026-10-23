#!/usr/bin/env python3
"""Persist a local Docker mount control using fixed synthetic code fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from grade_transformers_phase_a_v1 import _inspect_frozen_image  # noqa: E402
from run_coding_agent_dev_study_v1 import create_docker_shared_workspace  # noqa: E402
from run_local_agent import run_public_tests  # noqa: E402

IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)
TEST_SOURCE = (
    "from pathlib import Path\n"
    "namespace = {}\n"
    "exec(Path('src/module.py').read_text(), namespace)\n"
    "def test_repaired_behavior():\n"
    "    assert namespace['answer']() == 42\n"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _run_fixture(workspace: Path, source: str, test_hash: str) -> dict[str, Any]:
    target = workspace / "src" / "module.py"
    target.write_text(source, encoding="utf-8")
    result = run_public_tests(
        workspace,
        IMAGE,
        test_hash,
        memory_mb=512,
        expected_workspace_files={"src/module.py", "test_public.py"},
    )
    return {
        "target_sha256": sha256_file(target),
        "source": source,
        "result": result,
    }


def calibrate(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite calibration evidence: {output_dir}")
    docker = subprocess.run(
        ["docker", "--context", "colima", "info", "--format", "{{.OperatingSystem}}"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if docker.returncode != 0:
        raise RuntimeError("Colima Docker context is unavailable")
    image_id = _inspect_frozen_image(IMAGE)
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout
    dirty_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True,
        capture_output=True,
    ).stdout
    workspace = create_docker_shared_workspace("colima-public-mount-calibration")
    parent = workspace.parent
    try:
        workspace.mkdir(mode=0o755)
        target = workspace / "src" / "module.py"
        target.parent.mkdir()
        test_file = workspace / "test_public.py"
        test_file.write_text(TEST_SOURCE, encoding="utf-8")
        test_hash = sha256_file(test_file)
        baseline = _run_fixture(workspace, "def answer():\n    return 0\n", test_hash)
        fixed = _run_fixture(workspace, "def answer():\n    return 42\n", test_hash)
        passed = (
            baseline["result"].get("classification") == "assertion_failure"
            and baseline["result"].get("returncode") == 1
            and baseline["result"].get("passed") is False
            and fixed["result"].get("classification") == "passed"
            and fixed["result"].get("returncode") == 0
            and fixed["result"].get("passed") is True
        )
        report = {
            "schema_version": 1,
            "calibration_id": "vare-phase-a-v6-colima-shared-mount-v2",
            "purpose": "Docker workspace mount and public-test execution path only; no model inference",
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "source_dirty_at_start": bool(git_status.strip()),
            "working_tree_diff_sha256": sha256_bytes(dirty_diff),
            "implementation_sha256": {
                "scripts/run_local_agent.py": sha256_file(ROOT / "scripts/run_local_agent.py"),
                "src/vare/agent_structured_edit.py": sha256_file(ROOT / "src/vare/agent_structured_edit.py"),
                "scripts/run_coding_agent_dev_study_v1.py": sha256_file(ROOT / "scripts/run_coding_agent_dev_study_v1.py"),
                "scripts/calibrate_phase_a_v6_colima_mount_v1.py": sha256_file(Path(__file__)),
            },
            "docker_context": "colima",
            "docker_server_os": docker.stdout.strip(),
            "grading_image": IMAGE,
            "container_image_id": image_id,
            "workspace_location_class": "~/.cache/vare/candidate-workspaces/<private-task-dir>/workspace",
            "workspace_permissions": {
                "task_directory_mode": "0700",
                "mounted_workspace_mode": "0755 for uid 65534 read access",
                "mount_readonly": True,
                "network": "none",
            },
            "test_public_sha256": test_hash,
            "baseline": baseline,
            "known_fix": fixed,
            "calibration_passed": passed,
            "model_inference_performed": False,
            "limitations": [
                "Synthetic behavior fixture proves runtime path only; it is not an OSS task or model success.",
                "Task-specific public calibration remains separately required before a new model protocol.",
            ],
        }
        output_dir.mkdir(parents=True)
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        (output_dir / "report.json").write_text(payload, encoding="utf-8")
        (output_dir / "stdout-baseline.txt").write_text(
            baseline["result"].get("stdout", ""), encoding="utf-8"
        )
        (output_dir / "stderr-baseline.txt").write_text(
            baseline["result"].get("stderr", ""), encoding="utf-8"
        )
        (output_dir / "stdout-known-fix.txt").write_text(
            fixed["result"].get("stdout", ""), encoding="utf-8"
        )
        (output_dir / "stderr-known-fix.txt").write_text(
            fixed["result"].get("stderr", ""), encoding="utf-8"
        )
        lines = [
            f"{sha256_file(path)}  {path.name}"
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "SHA256SUMS"
        ]
        (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report
    finally:
        shutil.rmtree(parent)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["DOCKER_CONTEXT"] = "colima"
    report = calibrate(args.output_dir)
    print(json.dumps({
        "calibration_passed": report["calibration_passed"],
        "baseline": report["baseline"]["result"]["classification"],
        "known_fix": report["known_fix"]["result"]["classification"],
        "grading_image": report["grading_image"],
    }, indent=2, sort_keys=True))
    return 0 if report["calibration_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
