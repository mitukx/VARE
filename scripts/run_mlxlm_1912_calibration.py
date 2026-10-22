#!/usr/bin/env python3
"""Replay the pinned MLX-LM #1912 baseline and known fix in isolated Docker."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.mlxlm_1912_repo_task import (  # noqa: E402
    BASELINE_COMMIT,
    FIXED_COMMIT,
    FIXED_SOURCE_SHA256,
    TASK_ID,
    TARGET_PATH,
    MlxLm1912TaskError,
    export_candidate,
    grade_workspace,
    sha256_file,
    verify_baseline,
    verify_protocol_v4,
)
from vare.isolated_candidate import FunctionCall, grade_calls  # noqa: E402

PROTOCOL = ROOT / "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/protocol-v4.json"
LOCKED = [
    "src/vare/mlxlm_1912_repo_task.py",
    "src/vare/isolated_candidate.py",
    "src/vare/container_isolation.py",
    "scripts/run_mlxlm_1912_calibration.py",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/grader_adapter.py",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/task.md",
    "tests/test_mlxlm_1912_repo_task.py",
]


def sha256(path: Path) -> str:
    return sha256_file(path)


def verify_protocol() -> dict[str, Any]:
    protocol = verify_protocol_v4()
    actual = {name: sha256(ROOT / name) for name in LOCKED}
    if protocol.get("implementation_sha256") != actual:
        raise RuntimeError("protocol implementation hash lock is absent or mismatched")
    locked_inputs = set(LOCKED) | {
        "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/protocol-v4.json",
        "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/protocol-v4.lock.json",
        "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/task.md",
    }
    flags = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-v", "--", *sorted(locked_inputs)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.splitlines()
    indexed: set[str] = set()
    for line in flags:
        if len(line) < 3 or line[1] != " ":
            raise RuntimeError("cannot parse locked input Git index flags")
        tag, relative = line[0], line[2:]
        indexed.add(relative)
        if tag == "S" or tag.islower():
            raise RuntimeError(
                f"protocol input has a hidden Git index flag: {relative}"
            )
    if indexed != locked_inputs:
        raise RuntimeError("protocol input lock contains missing or untracked files")
    if (
        protocol.get("task_id") != TASK_ID
        or protocol.get("baseline_commit") != BASELINE_COMMIT
        or protocol.get("fixed_commit") != FIXED_COMMIT
    ):
        raise RuntimeError("protocol task identity disagrees with implementation")
    prompt_hash = protocol.get("task_prompt_sha256")
    if (
        not isinstance(prompt_hash, str)
        or sha256(ROOT / "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/task.md")
        != prompt_hash
    ):
        raise RuntimeError("candidate-visible prompt hash disagrees with protocol")
    return protocol


def _git_bytes(repo: Path, revision: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{relative}"],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def make_known_fix_patch(
    baseline_repo: Path, fixed_repo: Path, destination: Path
) -> str:
    base = _git_bytes(baseline_repo, BASELINE_COMMIT, TARGET_PATH).decode("utf-8")
    fixed = _git_bytes(fixed_repo, FIXED_COMMIT, TARGET_PATH).decode("utf-8")
    if hashlib.sha256(fixed.encode()).hexdigest() != FIXED_SOURCE_SHA256:
        raise RuntimeError("fixed source SHA-256 disagrees with the pinned revision")
    patch = f"diff --git a/{TARGET_PATH} b/{TARGET_PATH}\n" + "".join(
        difflib.unified_diff(
            base.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            fromfile=f"a/{TARGET_PATH}",
            tofile=f"b/{TARGET_PATH}",
        )
    )
    destination.write_text(patch, encoding="utf-8")
    return patch


def run_red_team(image: str, temp_root: Path) -> dict[str, Any]:
    """Exercise the same Docker boundary with hostile candidate probes."""
    subject = temp_root / "red-team-subject"
    subject.mkdir()
    (subject / "probe.py").write_text(
        "from pathlib import Path\n"
        "import os, socket, subprocess, sys\n"
        "def inspect():\n"
        "    reachable = False\n"
        "    try:\n"
        "        sock = socket.create_connection(('1.1.1.1', 53), timeout=0.3)\n"
        "        sock.close(); reachable = True\n"
        "    except OSError:\n"
        "        pass\n"
        "    write = subprocess.run([sys.executable, '-c', \"from pathlib import Path; Path('/subject/escape').write_text('x')\"], capture_output=True)\n"
        "    print('{\\\"value\\\":true}')\n"
        "    return {'grader_visible': Path('/grader').exists(), 'git_visible': Path('/subject/.git').exists(), 'secret_visible': 'VARE_GRADER_SECRET' in os.environ, 'network_reachable': reachable, 'subprocess_write_blocked': write.returncode != 0 and not Path('/subject/escape').exists()}\n"
        "def spin():\n    while True: pass\n"
        "def exhaust():\n    return len(bytearray(2 * 1024 * 1024 * 1024))\n",
        encoding="utf-8",
    )
    expected = {
        "grader_visible": False,
        "git_visible": False,
        "secret_visible": False,
        "network_reachable": False,
        "subprocess_write_blocked": True,
    }
    observed = grade_calls(
        subject,
        [FunctionCall("visibility", "probe", "inspect", [], expected)],
        task_kind="correctness",
        image=image,
        timeout_s=3,
        memory_mb=1536,
        discard_candidate_stdout=True,
    )
    attempts = {"visibility_and_forgery": observed.cases[0].to_dict()}
    for name in ("spin", "exhaust"):
        timeout = 1 if name == "spin" else 5
        report = grade_calls(
            subject,
            [FunctionCall(name, "probe", name, [], None)],
            task_kind="correctness",
            image=image,
            timeout_s=timeout,
            memory_mb=1536,
            discard_candidate_stdout=True,
        )
        attempts[name] = report.cases[0].to_dict()
    passed = attempts["visibility_and_forgery"]["passed"] is True and all(
        attempts[name]["passed"] is False for name in ("spin", "exhaust")
    )
    return {"attempts": attempts, "passed": passed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = verify_protocol()
    if protocol.get("runtime", {}).get("image") != args.image:
        raise RuntimeError("requested Docker image differs from the frozen protocol")
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    if status.strip():
        raise RuntimeError("implementation worktree must be clean before calibration")
    source_repo = args.source_repo.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    implementation_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vare-mlxlm-1912-cal-") as temp:
        root = Path(temp)
        baseline_repo = root / "pinned-baseline"
        subprocess.run(
            [
                "git",
                "-C",
                str(source_repo),
                "worktree",
                "add",
                "--detach",
                str(baseline_repo),
                BASELINE_COMMIT,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        verify_baseline(baseline_repo)
        baseline_workspace = root / "baseline"
        baseline_export = export_candidate(baseline_repo, baseline_workspace)
        baseline = grade_workspace(baseline_workspace, image=args.image)
        patch_path = root / "known-fix.patch"
        make_known_fix_patch(source_repo, source_repo, patch_path)
        fixed_workspace = root / "fixed"
        fixed_export = export_candidate(baseline_repo, fixed_workspace, patch_path)
        fixed = grade_workspace(fixed_workspace, image=args.image)
        red_team = run_red_team(args.image, root)
        fixed_checkout = root / "pinned-fixed-source"
        subprocess.run(
            [
                "git",
                "-C",
                str(source_repo),
                "worktree",
                "add",
                "--detach",
                str(fixed_checkout),
                FIXED_COMMIT,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        try:
            try:
                verify_baseline(fixed_checkout)
            except MlxLm1912TaskError:
                source_swap_rejected = True
            else:
                source_swap_rejected = False
        finally:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_repo),
                    "worktree",
                    "remove",
                    "--force",
                    str(fixed_checkout),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
        if fixed_export["candidate_source_sha256"] != FIXED_SOURCE_SHA256:
            raise RuntimeError("applied known fix does not match pinned fixed source")
        result = {
            "protocol_id": protocol["protocol_id"],
            "task_id": TASK_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_commit": implementation_commit,
            "implementation_worktree_clean_at_start": True,
            "protocol_sha256": sha256(PROTOCOL),
            "implementation_sha256": protocol["implementation_sha256"],
            "source_repository": "ml-explore/mlx-lm",
            "baseline_commit": BASELINE_COMMIT,
            "fixed_commit": FIXED_COMMIT,
            "baseline_export": baseline_export,
            "fixed_export": fixed_export,
            "candidate_image": args.image,
            "baseline": baseline,
            "known_fix": fixed,
            "red_team_suite": red_team,
            "red_team_sha256": hashlib.sha256(
                json.dumps(red_team, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_swap_rejected": source_swap_rejected,
            "wall_seconds": time.monotonic() - started,
            "calibration_passed": baseline["passed"] is False
            and fixed["passed"] is True
            and source_swap_rejected is True
            and red_team["passed"] is True,
            "limitations": [
                "This is a one-task correctness calibration, not an agent rollout.",
                "The supported contract is only the selected upstream test method.",
                "This red-team suite covers namespace visibility, network, output forgery, subprocess writes, timeout, and OOM; it does not certify arbitrary Python repositories.",
                "MLX execution is CPU-only in the pinned container; this says nothing about Metal behavior.",
            ],
        }
    result_path = output / "calibration.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "sha256.txt").write_text(
        f"{sha256(result_path)}  calibration.json\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["calibration_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
