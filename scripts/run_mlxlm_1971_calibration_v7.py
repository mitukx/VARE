#!/usr/bin/env python3
"""Calibrate MLX-LM #1971 v7 with source and cache-specific controls."""

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
sys.path.insert(0, str(ROOT))
from vare.isolated_candidate import FunctionCall, grade_calls  # noqa: E402
from vare.mlxlm_1971_repo_task import (  # noqa: E402
    BASELINE_COMMIT,
    FIXED_COMMIT,
    FIXED_SOURCE_SHA256,
    IMAGE,
    TASK_ID,
    TARGET_PATH,
    MlxLm1971TaskError,
    sha256_file,
)
from vare.mlxlm_1971_repo_task_v7 import (  # noqa: E402
    IMPLEMENTATION_FILES as V7_IMPLEMENTATION_FILES,
    export_candidate_v7,
    grade_workspace_v7,
    verify_pinned_source_tree,
    verify_protocol_v7,
)

TASK_ROOT = ROOT / "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale"
PROTOCOL = TASK_ROOT / "protocol-v7.json"
LOCK = TASK_ROOT / "protocol-v7.lock.json"
IMPLEMENTATION_FILES = sorted(V7_IMPLEMENTATION_FILES)


def _case_decisions(report: dict[str, Any], scales: list[float]) -> list[tuple[str, bool]] | None:
    cases = report.get("cases")
    expected_ids = [
        f"value-scale-{scale:g}-{phase}"
        for scale in scales
        for phase in ("full", "cached")
    ]
    if not isinstance(cases, list) or [case.get("case_id") for case in cases] != expected_ids:
        return None
    decisions: list[tuple[str, bool]] = []
    for case in cases:
        failure = case.get("failure")
        passed = case.get("passed")
        if not isinstance(passed, bool) or (failure is not None and not isinstance(failure, str)):
            return None
        decisions.append((failure or "", passed))
    return decisions


def _has_semantic_decisions(
    baseline: dict[str, Any], fixed: dict[str, Any], constant: dict[str, Any],
    cache_mutant: dict[str, Any], scales: list[float],
) -> bool:
    baseline_cases = _case_decisions(baseline, scales)
    fixed_cases = _case_decisions(fixed, scales)
    constant_cases = _case_decisions(constant, scales)
    cache_cases = _case_decisions(cache_mutant, scales)
    return (
        baseline.get("passed") is False
        and baseline_cases == [("wrong_output", False)] * (2 * len(scales))
        and fixed.get("passed") is True
        and fixed_cases == [("", True)] * (2 * len(scales))
        and constant.get("passed") is False
        and constant_cases == [
            ("wrong_output", False), ("wrong_output", False),
            ("", True), ("", True),
            ("wrong_output", False), ("wrong_output", False),
        ]
        and cache_mutant.get("passed") is False
        and cache_cases == [
            ("", True), ("wrong_output", False),
            ("", True), ("wrong_output", False),
            ("", True), ("wrong_output", False),
        ]
    )


def _run_red_team(image: str, temp_root: Path) -> dict[str, object]:
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
    visibility = grade_calls(
        subject,
        [FunctionCall("visibility", "probe", "inspect", [], expected)],
        task_kind="correctness",
        image=image,
        timeout_s=3,
        memory_mb=1536,
        discard_candidate_stdout=True,
    )
    controls: dict[str, Any] = {
        "visibility_and_forgery": visibility.cases[0].to_dict()
    }
    for name, timeout in (("spin", 1), ("exhaust", 5)):
        report = grade_calls(
            subject,
            [FunctionCall(name, "probe", name, [], None)],
            task_kind="correctness",
            image=image,
            timeout_s=timeout,
            memory_mb=1536,
            discard_candidate_stdout=True,
        )
        controls[name] = report.cases[0].to_dict()
    return {
        "attempts": controls,
        "passed": visibility.cases[0].passed
        and all(not controls[name]["passed"] for name in ("spin", "exhaust")),
    }


def _git_bytes(repo: Path, revision: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{TARGET_PATH}"],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def _known_fix_patch(repo: Path, destination: Path) -> str:
    baseline = _git_bytes(repo, BASELINE_COMMIT).decode("utf-8")
    fixed = _git_bytes(repo, FIXED_COMMIT).decode("utf-8")
    if hashlib.sha256(fixed.encode()).hexdigest() != FIXED_SOURCE_SHA256:
        raise RuntimeError("fixed source hash mismatch")
    text = f"diff --git a/{TARGET_PATH} b/{TARGET_PATH}\n" + "".join(
        difflib.unified_diff(
            baseline.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            fromfile=f"a/{TARGET_PATH}",
            tofile=f"b/{TARGET_PATH}",
        )
    )
    destination.write_text(text, encoding="utf-8")
    return text


def _mutate_workspace(source: Path, destination: Path, kind: str) -> str:
    shutil.copytree(source, destination)
    target = destination / TARGET_PATH
    original = target.read_text(encoding="utf-8")
    if kind == "constant_scale":
        changed = original.replace(
            "values = values * self.value_scale", "values = values * 0.5", 1
        )
    elif kind == "scale_after_cache":
        expected = """        if self.value_scale is not None:
            values = values * self.value_scale

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
"""
        replacement = """        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)

        if self.value_scale is not None:
            values = values * self.value_scale
"""
        if expected not in original:
            raise RuntimeError("fixed source cache block changed unexpectedly")
        changed = original.replace(expected, replacement, 1)
    else:
        raise ValueError(kind)
    if changed == original:
        raise RuntimeError(f"failed to create {kind} negative control")
    target.write_text(changed, encoding="utf-8")
    return sha256_file(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = verify_protocol_v7()
    if args.image != protocol["runtime"]["image"]:
        raise RuntimeError("requested image differs from v7 protocol")
    implementation_hashes = {
        name: sha256_file(ROOT / name) for name in IMPLEMENTATION_FILES
    }
    if implementation_hashes != protocol["implementation_sha256"]:
        raise RuntimeError("v7 implementation hash lock mismatch")
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    if status:
        raise RuntimeError("implementation worktree must be clean before calibration")
    repo = args.source_repo.resolve(strict=True)
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
    with tempfile.TemporaryDirectory(prefix="vare-mlxlm-1971-v7-") as folder:
        temp = Path(folder)
        baseline_source = temp / "baseline-source"
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(baseline_source), BASELINE_COMMIT],
            check=True,
            capture_output=True,
            timeout=120,
        )
        verify_pinned_source_tree(baseline_source)
        baseline_workspace = temp / "baseline-candidate"
        baseline_export = export_candidate_v7(baseline_source, baseline_workspace)
        baseline = grade_workspace_v7(baseline_workspace, image=args.image)

        patch_path = temp / "known-fix.patch"
        patch_text = _known_fix_patch(repo, patch_path)
        fixed_workspace = temp / "fixed-candidate"
        fixed_export = export_candidate_v7(baseline_source, fixed_workspace, patch_path)
        fixed = grade_workspace_v7(fixed_workspace, image=args.image)

        constant_workspace = temp / "constant-scale-mutant"
        constant_hash = _mutate_workspace(
            fixed_workspace, constant_workspace, "constant_scale"
        )
        constant = grade_workspace_v7(constant_workspace, image=args.image)

        cache_workspace = temp / "post-cache-mutant"
        cache_hash = _mutate_workspace(
            fixed_workspace, cache_workspace, "scale_after_cache"
        )
        cache_mutant = grade_workspace_v7(cache_workspace, image=args.image)
        red_team = _run_red_team(args.image, temp)

        fixed_source = temp / "fixed-source"
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(fixed_source), FIXED_COMMIT],
            check=True,
            capture_output=True,
            timeout=120,
        )
        try:
            try:
                verify_pinned_source_tree(fixed_source)
            except MlxLm1971TaskError:
                source_swap_rejected = True
            else:
                source_swap_rejected = False
        finally:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(fixed_source)],
                check=True,
                capture_output=True,
                timeout=30,
            )
        semantic_decisions_match = _has_semantic_decisions(
            baseline, fixed, constant, cache_mutant, protocol["value_scales"]
        )
        result = {
            "protocol_id": protocol["protocol_id"],
            "task_id": TASK_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_commit": implementation_commit,
            "implementation_worktree_clean_at_start": True,
            "protocol_sha256": sha256_file(PROTOCOL),
            "implementation_sha256": implementation_hashes,
            "source_repository": "ml-explore/mlx-lm",
            "baseline_commit": BASELINE_COMMIT,
            "fixed_commit": FIXED_COMMIT,
            "baseline_export": baseline_export,
            "fixed_export": fixed_export,
            "known_patch_sha256": hashlib.sha256(patch_text.encode()).hexdigest(),
            "candidate_image": args.image,
            "expected_output_sha256": sha256_file(
                ROOT / "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/expected-output-v2.json"
            ),
            "source_manifest_sha256": sha256_file(
                TASK_ROOT / "baseline-python-manifest-v6.json"
            ),
            "baseline": baseline,
            "known_fix": fixed,
            "constant_scale_mutant": {"source_sha256": constant_hash, "result": constant},
            "scale_after_cache_mutant": {"source_sha256": cache_hash, "result": cache_mutant},
            "semantic_negative_controls_match": semantic_decisions_match,
            "red_team_suite": red_team,
            "source_swap_rejected": source_swap_rejected,
            "wall_seconds": time.monotonic() - started,
            "calibration_passed": semantic_decisions_match
            and source_swap_rejected
            and red_team["passed"] is True,
            "limitations": [
                "Synthetic small random model and one Development bug; not a model-generated patch or capability result.",
                "CPU-only pinned MLX container; this does not establish Metal behavior.",
                "Grader checks a narrow task contract and does not certify arbitrary repositories.",
                "Negative controls count only when every required case has the expected wrong_output or pass decision; crashes and timeouts are not accepted as semantic rejection.",
            ],
        }
    result_path = output / "calibration.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        f"{sha256_file(result_path)}  calibration.json\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["calibration_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
