#!/usr/bin/env python3
"""Versioned strict-output wrapper; refuses the already-consumed v1 cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import run_qwen3_4b_mlsys_development_feasibility_v1 as legacy  # noqa: E402
from vare.qwen_patch_contract_v2 import grade_with_phase_taxonomy  # noqa: E402

PROTOCOL = ROOT / "protocols/qwen3_4b_mlsys_development_feasibility_v2.json"
LOCK = ROOT / "protocols/qwen3_4b_mlsys_development_feasibility_v2.lock.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protocol_v2() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != lock.get("protocol_sha256"):
        raise RuntimeError("v2 protocol hash mismatch")
    parent_protocol = ROOT / "protocols/qwen3_4b_mlsys_development_feasibility_v1.lock.json"
    if _sha256(parent_protocol) != protocol.get("protocol_parent_sha256"):
        raise RuntimeError("v1 parent protocol identity mismatch")
    contract_protocol = ROOT / "protocols/qwen3_patch_output_contract_v2.json"
    if _sha256(contract_protocol) != protocol.get("contract_protocol_sha256"):
        raise RuntimeError("v2 patch contract identity mismatch")
    hashes = protocol.get("implementation_sha256", {})
    if hashes != lock.get("implementation_sha256") or not hashes:
        raise RuntimeError("v2 implementation inventory mismatch")
    actual = {name: _sha256(ROOT / name) for name in hashes}
    if actual != hashes:
        raise RuntimeError("v2 implementation source hash mismatch")
    if protocol.get("task_ids") != list(legacy.TASKS):
        raise RuntimeError("v2 task inventory mismatch")
    if protocol.get("status") != "blocked-consumed-task-cohort":
        raise RuntimeError("v2 task cohort state is not explicitly blocked")
    raise RuntimeError("v2 rollout blocked: every task in this cohort was exposed in v1")


def _validate_task_diff(module: Any, raw: str) -> None:
    validator = getattr(module, "_validate_diff", None)
    if validator is None:
        validator = getattr(module, "_validate_patch", None)
    if validator is None:
        validator = getattr(module, "validate_patch", None)
    if validator is None:
        raise ValueError("task adapter has no pinned diff validator")
    validator(raw)


def _apply_and_grade_v2(
    task_id: str, spec: dict[str, Any], baseline_repo: Path, raw: str, folder: Path
) -> dict[str, Any]:
    module = spec["module"]
    target = spec["target"]
    patch = folder / "candidate.patch"
    workspace = folder / "candidate"

    def apply(candidate_diff: str) -> tuple[Path, dict[str, Any]]:
        patch.write_text(candidate_diff, encoding="utf-8")
        if task_id == legacy.TASKS[0]:
            exported = module.apply_patch(baseline_repo, patch, workspace)
        else:
            exported = module.export_candidate(baseline_repo, workspace, patch)
        return workspace, exported

    def grade(candidate: tuple[Path, dict[str, Any]]) -> dict[str, Any]:
        candidate_workspace, exported = candidate
        if task_id == legacy.TASKS[0]:
            return module.grade_workspace(
                candidate_workspace,
                image=spec["image"],
                baseline_files=exported["baseline_files"],
            )
        return module.grade_workspace(candidate_workspace, image=spec["image"])

    result = grade_with_phase_taxonomy(
        raw,
        target,
        apply_patch=apply,
        grade_candidate=grade,
        validate_patch=lambda text: _validate_task_diff(module, text),
    )
    result["patch_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return result


def main() -> int:
    legacy.PROTOCOL = PROTOCOL
    legacy.verify_protocol = verify_protocol_v2
    legacy._apply_and_grade = _apply_and_grade_v2
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
