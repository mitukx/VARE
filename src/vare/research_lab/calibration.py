"""Grader calibration. Known-good and incorrect controls are not agent results."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .evaluation import grade
from .spec import ResearchTask, load_catalog, repo_root, suite_manifest
from .workspace import materialize, overlay_control

_EXPECTATIONS = {
    "baseline": False,
    "known_good": True,
    "incorrect": False,
}


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def calibrate_task(task: ResearchTask, work_root: Path) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for name, expect_pass in _EXPECTATIONS.items():
        workspace = work_root / task.task_id / name
        materialize(task, workspace)
        if name != "baseline":
            overlay_control(task, workspace, name)
        result = grade(task, workspace)
        cases[name] = {
            "expected_pass": expect_pass,
            "passed": result.get("passed"),
            "classification": result.get("classification"),
            "correctness_evaluated": result.get("correctness_evaluated"),
            "correctness_passed": result.get("correctness_passed"),
            "performance_evaluated": result.get("performance_evaluated"),
            "performance_passed": result.get("performance_passed"),
            "metrics": result.get("metrics"),
            "failures": result.get("failures"),
            "wall_s": result.get("wall_s"),
            "peak_rss_kb": result.get("peak_rss_kb"),
            "source_hashes": result.get("source_hashes"),
            "matches_expectation": result.get("passed") is expect_pass,
        }
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "spec_sha256": task.spec_sha256,
        "cases": cases,
        "matches_expectation": all(case["matches_expectation"] for case in cases.values()),
    }


def run_calibration(work_root: Path, catalog_root: Path | None = None) -> dict[str, Any]:
    tasks = load_catalog(catalog_root)
    manifest = suite_manifest(catalog_root)
    rows = [calibrate_task(task, work_root) for task in tasks.values()]
    return {
        "schema": "vare.research_lab.calibration.v1",
        "git_revision": _git_revision(),
        "suite_sha256": manifest["manifest_sha256"],
        "expectations_met": all(row["matches_expectation"] for row in rows),
        "tasks": rows,
        "note": (
            "Known-good controls calibrate the grader. "
            "They are not autonomous agent successes and do not authorize training."
        ),
    }
