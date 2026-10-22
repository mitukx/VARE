#!/usr/bin/env python3
"""Run baseline/fix calibration for the frozen Transformers #45852 task."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.transformers_repo_task import (  # noqa: E402
    BASELINE_COMMIT,
    FIXED_COMMIT,
    PACKAGE_PATH,
    TASK_DESCRIPTION,
    TARGET_PATH,
    TASK_ID,
    apply_patch,
    export_baseline,
    grade_workspace,
    _validate_diff,
    TransformersTaskError,
    verify_baseline,
    verify_protocol,
)
from vare.isolated_candidate import FunctionCall, grade_calls  # noqa: E402

TASK_DIR = ROOT / "benchmarks/mlsys_repo_eval_v2/transformers_45852"
PROTOCOL = TASK_DIR / "protocol-v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    protocol = verify_protocol()
    repo = args.source_repo.resolve(strict=True)
    implementation_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    implementation_worktree_clean = not subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cases: dict[str, Any] = {}
    red_team: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="vare-transformers-45852-cal-") as temp:
        temp_root = Path(temp)
        baseline = temp_root / "source"
        prepare_sparse_checkout(repo, baseline)
        verify_baseline(baseline)
        baseline_workspace = temp_root / "baseline-work"
        baseline_export = export_baseline(baseline, baseline_workspace)
        cases["baseline"] = grade_workspace(
            baseline_workspace,
            image=args.image,
            baseline_files=baseline_export["baseline_files"],
        )
        # Apply the exact source delta as the candidate patch against the pinned baseline.
        patch = temp_root / "known-fix.patch"
        fixed_bytes = subprocess.run(
            ["git", "-C", str(baseline), "show", f"{FIXED_COMMIT}:{TARGET_PATH}"],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        if hashlib.sha256(fixed_bytes).hexdigest() != protocol["fixed_target_sha256"]:
            raise RuntimeError("pinned fixed source SHA-256 does not match protocol")
        baseline_lines = (
            (baseline / TARGET_PATH)
            .read_text(encoding="utf-8")
            .splitlines(keepends=True)
        )
        fixed_lines = fixed_bytes.decode("utf-8").splitlines(keepends=True)
        patch_text = f"diff --git a/{TARGET_PATH} b/{TARGET_PATH}\n" + "".join(
            difflib.unified_diff(
                baseline_lines,
                fixed_lines,
                fromfile=f"a/{TARGET_PATH}",
                tofile=f"b/{TARGET_PATH}",
            )
        )
        patch.write_text(patch_text, encoding="utf-8")
        fixed_workspace = temp_root / "fixed-work"
        fixed_export = apply_patch(baseline, patch, fixed_workspace)
        cases["known_fix"] = grade_workspace(
            fixed_workspace,
            image=args.image,
            baseline_files=fixed_export["baseline_files"],
        )
        if sha256(fixed_workspace / TARGET_PATH) != protocol["fixed_target_sha256"]:
            raise RuntimeError(
                "candidate-applied fix does not match pinned fixed source"
            )
        mutation_checks = run_mutation_checks(
            temp_root, fixed_workspace, fixed_export["baseline_files"], args.image
        )
        red_team = run_red_team(temp_root, args.image)
    calibration_passed = (
        cases["baseline"]["passed"] is False and cases["known_fix"]["passed"] is True
    )
    result = {
        "protocol_id": protocol["protocol_id"],
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(repo),
        "implementation_commit": implementation_commit,
        "implementation_worktree_clean_at_start": implementation_worktree_clean,
        "baseline_commit": BASELINE_COMMIT,
        "fixed_commit": FIXED_COMMIT,
        "known_fix_patch_sha256": fixed_export["patch_sha256"],
        "protocol_sha256": sha256(PROTOCOL),
        "implementation_sha256": protocol["implementation_sha256"],
        "candidate_image": args.image,
        "cases": cases,
        "red_team_suite": red_team,
        "calibration_passed": False,
        "wall_s": time.monotonic() - started,
    }
    result["mutation_checks"] = mutation_checks
    result["calibration_passed"] = (
        calibration_passed and red_team["passed"] and mutation_checks["passed"]
    )
    result_path = output / "calibration.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "sha256.txt").write_text(
        f"{sha256(result_path)}  calibration.json\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["calibration_passed"] else 1


def run_mutation_checks(
    temp_root: Path,
    known_fix_workspace: Path,
    baseline_files: dict[str, str],
    image: str,
) -> dict[str, Any]:
    """Prove the parent-owned cases reject shortcuts to the pinned fix."""
    source_path = known_fix_workspace / TARGET_PATH
    source = source_path.read_text(encoding="utf-8")
    replacements = {
        "constant-mask-output": (
            "torch.stack(preserved_attention_mask)",
            "torch.stack([torch.full_like(mask, 1) for mask in preserved_attention_mask])",
        ),
        "mean-mask-output": (
            "torch.stack(preserved_attention_mask)",
            "torch.stack([torch.full_like(mask, mask.mean()) for mask in preserved_attention_mask])",
        ),
        "constant-numpy-mask-output": (
            "np.stack(preserved_attention_mask)",
            "np.stack([np.full_like(mask, 1) for mask in preserved_attention_mask])",
        ),
        "mean-numpy-mask-output": (
            "np.stack(preserved_attention_mask)",
            "np.stack([np.full_like(mask, mask.mean()) for mask in preserved_attention_mask])",
        ),
        "corrupt-padded-input-ids": (
            "return BatchEncoding(batch_outputs, tensor_type=return_tensors)",
            'batch_outputs["input_ids"] = [[0] * len(row) for row in batch_outputs["input_ids"]]\n'
            "        return BatchEncoding(batch_outputs, tensor_type=return_tensors)",
        ),
        "corrupt-padded-attention-mask": (
            "return BatchEncoding(batch_outputs, tensor_type=return_tensors)",
            'batch_outputs["attention_mask"] = [[1] * len(row) for row in batch_outputs["attention_mask"]]\n'
            "        return BatchEncoding(batch_outputs, tensor_type=return_tensors)",
        ),
    }
    rejected: dict[str, bool] = {}
    for name, (old, new) in replacements.items():
        if old not in source:
            rejected[name] = False
            continue
        mutant = temp_root / f"mutant-{name}"
        shutil.copytree(known_fix_workspace, mutant)
        mutant_source = mutant / TARGET_PATH
        mutant_source.write_text(source.replace(old, new, 1), encoding="utf-8")
        try:
            report = grade_workspace(
                mutant,
                image=image,
                baseline_files=baseline_files,
            )
            rejected[name] = not report["passed"]
        except (TransformersTaskError, OSError):
            rejected[name] = True
    return {"executed": True, "passed": bool(rejected) and all(rejected.values()), "rejected": rejected}


def prepare_sparse_checkout(source_repo: Path, destination: Path) -> None:
    """Fetch exact source revisions with only the allowlisted package materialized."""
    remote = subprocess.run(
        ["git", "-C", str(source_repo), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    subprocess.run(["git", "init", "-q", str(destination)], check=True, timeout=10)
    subprocess.run(
        ["git", "-C", str(destination), "remote", "add", "origin", remote],
        check=True,
        timeout=10,
    )
    subprocess.run(
        ["git", "-C", str(destination), "config", "extensions.partialClone", "origin"],
        check=True,
        timeout=10,
    )
    subprocess.run(
        ["git", "-C", str(destination), "config", "remote.origin.promisor", "true"],
        check=True,
        timeout=10,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "config",
            "remote.origin.partialclonefilter",
            "blob:none",
        ],
        check=True,
        timeout=10,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--filter=blob:none",
            "--no-tags",
            "origin",
            BASELINE_COMMIT,
            FIXED_COMMIT,
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    sparse = destination / ".git/info/sparse-checkout"
    sparse.write_text(f"/{PACKAGE_PATH}/\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(destination), "config", "core.sparseCheckout", "true"],
        check=True,
        timeout=10,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "-q", "--detach", BASELINE_COMMIT],
        check=True,
        capture_output=True,
        timeout=60,
    )


def run_red_team(temp_root: Path, image: str) -> dict[str, Any]:
    """Exercise the candidate/grader process boundary using adversarial code."""
    subject = temp_root / "red-team-subject"
    subject.mkdir()
    (subject / "probe.py").write_text(
        "from pathlib import Path\n"
        "import os, socket, subprocess, sys\n"
        "def inspect():\n"
        "    reachable = False\n"
        "    try:\n"
        "        sock = socket.create_connection(('1.1.1.1', 53), timeout=0.4)\n"
        "        sock.close(); reachable = True\n"
        "    except OSError:\n"
        "        pass\n"
        "    write = subprocess.run([sys.executable, '-c', \"from pathlib import Path; Path('/subject/escape').write_text('x')\"], capture_output=True)\n"
        '    print(\'{\\"value\\":{\\"forged\\":true}}\')\n'
        "    return {'grader_visible': Path('/grader').exists(), 'git_visible': Path('/subject/.git').exists(), 'secret_visible': 'VARE_GRADER_SECRET' in os.environ, 'network_reachable': reachable, 'subprocess_write_blocked': write.returncode != 0 and not Path('/subject/escape').exists()}\n"
        "def spin():\n    while True: pass\n"
        "def exhaust():\n    return len(bytearray(256 * 1024 * 1024))\n",
        encoding="utf-8",
    )
    call = FunctionCall(
        "red_team_visibility",
        "probe",
        "inspect",
        [],
        {
            "grader_visible": False,
            "git_visible": False,
            "secret_visible": False,
            "network_reachable": False,
            "subprocess_write_blocked": True,
        },
    )
    visibility = grade_calls(
        subject,
        [call],
        task_kind="correctness",
        image=image,
        timeout_s=5,
        memory_mb=256,
        discard_candidate_stdout=True,
    )
    # The forged stdout line says success, but the parent must compare the actual return value.
    forged = grade_calls(
        subject,
        [FunctionCall("stdout_forgery", "probe", "inspect", [], {"forged": True})],
        task_kind="correctness",
        image=image,
        timeout_s=5,
        memory_mb=256,
        discard_candidate_stdout=True,
    )
    timeout = grade_calls(
        subject,
        [FunctionCall("timeout", "probe", "spin", [], True)],
        task_kind="correctness",
        image=image,
        timeout_s=2,
        memory_mb=256,
        discard_candidate_stdout=True,
    )
    memory = grade_calls(
        subject,
        [FunctionCall("memory_limit", "probe", "exhaust", [], True)],
        task_kind="correctness",
        image=image,
        timeout_s=5,
        memory_mb=64,
        discard_candidate_stdout=True,
    )
    try:
        _validate_diff(
            "diff --git a/../grader.py b/../grader.py\n--- a/../grader.py\n"
            "+++ b/../grader.py\n@@ -1 +1 @@\n-a\n+b\n"
        )
        path_traversal_rejected = False
    except TransformersTaskError:
        path_traversal_rejected = True

    contaminated = temp_root / "contaminated"
    contaminated.mkdir()
    source_file = contaminated / TARGET_PATH
    source_file.parent.mkdir(parents=True)
    source_file.write_text("candidate\n", encoding="utf-8")
    task_copy = contaminated / "TASK.md"
    task_copy.write_bytes(TASK_DESCRIPTION.read_bytes())
    mini_manifest = {
        "TASK.md": sha256(task_copy),
        TARGET_PATH: "0" * 64,
    }
    (contaminated / "tests").mkdir()
    (contaminated / "tests/leak.py").write_text("answer = 42\n", encoding="utf-8")
    try:
        grade_workspace(contaminated, image=image, baseline_files=mini_manifest)
        test_injection_rejected = False
    except TransformersTaskError:
        test_injection_rejected = True
    link = contaminated / "src/transformers/leak"
    try:
        link.symlink_to(task_copy)
        from vare.transformers_repo_task import sha256_inventory

        try:
            sha256_inventory(contaminated)
            symlink_rejected = False
        except TransformersTaskError:
            symlink_rejected = True
    except OSError:
        symlink_rejected = False
    checks = {
        "grader_git_env_network_isolation": visibility.passed,
        "stdout_forgery_rejected": not forged.passed,
        "timeout_killed": not timeout.passed,
        "memory_limit_enforced": not memory.passed,
        "path_traversal_rejected": path_traversal_rejected,
        "hidden_test_injection_rejected": test_injection_rejected,
        "symlink_rejected": symlink_rejected,
    }
    return {
        "executed": True,
        "passed": all(checks.values()),
        "checks": checks,
        "raw_results": {
            "visibility": visibility.to_dict(),
            "stdout_forgery": forged.to_dict(),
            "timeout": timeout.to_dict(),
            "memory_limit": memory.to_dict(),
            "structural": {
                "path_traversal_rejected": path_traversal_rejected,
                "hidden_test_injection_rejected": test_injection_rejected,
                "symlink_rejected": symlink_rejected,
            },
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
