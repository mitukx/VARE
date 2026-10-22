#!/usr/bin/env python3
"""Fail-closed host preflight and negative-control runner for MLX-LM #1912 v4.

This is prospective operational tooling, NOT historical v1-v3 evidence.
Runs no models or training. The --execute flag needs macOS/Colima.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vare.mlxlm_1912_repo_task import (  # noqa: E402
    BASELINE_COMMIT,
    BASELINE_SOURCE_SHA256,
    FIXED_COMMIT,
    FIXED_SOURCE_SHA256,
    TARGET_PATH,
    export_candidate,
    grade_workspace,
    sha256_file,
    verify_protocol_v4,
)


def _run(argv: list[str], *, env: dict[str, str] | None = None) -> bytes:
    return subprocess.run(
        argv, check=True, capture_output=True, timeout=120, env=env
    ).stdout


def _pinned_source(repo: Path, revision: str) -> bytes:
    env = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}
    return _run(
        ["git", "-C", str(repo), "show", f"{revision}:{TARGET_PATH}"], env=env
    )


def verify_inputs(source_repo: Path, output: Path) -> dict[str, str]:
    """Read-only checks; never create, clean, or overwrite a run directory."""
    protocol = verify_protocol_v4()
    source_repo = source_repo.resolve(strict=True)
    if not source_repo.is_dir():
        raise ValueError("source repo must be an existing directory")
    origin = _run(
        ["git", "-C", str(source_repo), "rev-parse", "--show-toplevel"]
    ).decode().strip()
    if Path(origin).resolve() != source_repo:
        raise ValueError("source repo must be the exact Git checkout root")
    for revision, expected in (
        (BASELINE_COMMIT, BASELINE_SOURCE_SHA256),
        (FIXED_COMMIT, FIXED_SOURCE_SHA256),
    ):
        actual = hashlib.sha256(_pinned_source(source_repo, revision)).hexdigest()
        if actual != expected:
            raise ValueError(f"pinned source hash mismatch for {revision}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    if not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    return {
        "protocol_id": protocol["protocol_id"],
        "image": protocol["runtime"]["image"],
        "source_repo": str(source_repo),
        "output": str(output),
    }


def verify_host(image: str, context: str) -> None:
    if sys.platform != "darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("live v4 calibration requires the pinned Apple Silicon/macOS host")
    _run(["docker", "--context", context, "version", "--format", "{{.Server.Version}}"])
    _run(["docker", "--context", context, "image", "inspect", image, "--format", "{{.Id}}"])


def run_adversarial_probe(source_repo: Path, output: Path, image: str) -> bool:
    """Test a hardcoded index-0/zero shortcut, without changing upstream inputs."""
    with tempfile.TemporaryDirectory(prefix="vare-1912-v4-negative-") as temporary:
        root = Path(temporary)
        checkout = root / "source-checkout"
        _run([
            "git", "-C", str(source_repo), "worktree", "add", "--detach",
            str(checkout), BASELINE_COMMIT,
        ])
        try:
            workspace = root / "candidate"
            export_candidate(checkout, workspace)
            target = workspace / TARGET_PATH
            original = target.read_text(encoding="utf-8")
            start = original.find("def apply_top_p(")
            end = original.find("\n\n@partial(", start)
            if start < 0 or end <= start:
                raise RuntimeError("pinned top-p function boundaries changed")
            mutant = (
                "def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:\n"
                "    # Intentional incorrect answer: keep only index 0 at value 0.0.\n"
                "    return mx.where(mx.arange(logprobs.shape[-1]) == 0, "
                "mx.zeros_like(logprobs), -float('inf'))\n"
            )
            target.write_text(original[:start] + mutant + original[end:], encoding="utf-8")
            report = grade_workspace(workspace, image=image)
            observed = report["cases"]
            by_id = {item["case_id"]: item for item in observed}
            wrong_controls = {
                "shifted_maximum_nonzero_values",
                "leading_maximum_nonzero_value",
            }
            wrong_by_value = (
                all(
                    by_id.get(name, {}).get("failure") == "wrong_output"
                    for name in wrong_controls
                )
                and len(by_id) == len(observed) == 14
            )
            unchanged_controls_pass = all(
                item["passed"] is True and item.get("failure") is None
                for name, item in by_id.items() if name not in wrong_controls
            )
            rejected = (
                report["passed"] is False
                and wrong_by_value
                and unchanged_controls_pass
            )
            result = {
                "schema_version": 1,
                "protocol_id": "mlxlm-1912-isolated-task-v4",
                "probe": "hardcoded-index-0-zero-value",
                "candidate_source_sha256": sha256_file(target),
                "grader_passed": report["passed"],
                "case_outcomes": [
                    {"case_id": item["case_id"], "passed": item["passed"],
                     "failure": item.get("failure")}
                    for item in observed
                ],
                "adversarial_rejected": rejected,
                "qualification": "live-candidate-negative-control-only",
            }
            path = output / "adversarial-negative-control.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            (output / "adversarial-negative-control.sha256").write_text(
                f"{digest}  adversarial-negative-control.json\n", encoding="utf-8"
            )
            return bool(result["adversarial_rejected"])
        finally:
            _run([
                "git", "-C", str(source_repo), "worktree",
                "remove", "--force", str(checkout),
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-context", default="colima")
    parser.add_argument(
        "--execute", action="store_true",
        help="run the frozen v4 calibration and the adversarial negative control",
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"evidence path already exists: {args.output}")
    try:
        info = verify_inputs(args.source_repo, args.output)
        verify_host(info["image"], args.docker_context)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"BLOCKED preflight: {exc}") from exc
    print(json.dumps({"status": "READY_NOT_RUN", **info}, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    env = {**os.environ, "DOCKER_CONTEXT": args.docker_context}
    command = [
        sys.executable,
        str(ROOT / "scripts/run_mlxlm_1912_calibration.py"),
        "--source-repo", info["source_repo"],
        "--output", info["output"],
        "--image", info["image"],
    ]
    calibration = subprocess.run(command, env=env, check=False)
    if calibration.returncode != 0:
        print("BLOCKED: calibration failed; retained its original evidence", file=sys.stderr)
        return 2
    try:
        passed = run_adversarial_probe(
            Path(info["source_repo"]), Path(info["output"]), info["image"]
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"BLOCKED: adversarial probe error: {exc}", file=sys.stderr)
        return 3
    if not passed:
        print("BLOCKED: index-0 shortcut was not rejected", file=sys.stderr)
        return 4
    print("V4_CALIBRATION_AND_ADVERSARIAL_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
