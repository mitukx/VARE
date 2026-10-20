#!/usr/bin/env python3
"""Bind the frozen MLSys readiness audit to an exact VARE commit and tree."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/mlsys_actual_model_update_no_go_v2.lock.json"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def verify_source_binding(lock: dict[str, Any]) -> dict[str, str]:
    source = lock["source"]
    commit = source["commit"]
    tree = source["tree"]
    actual_tree = _git("rev-parse", f"{commit}^{{tree}}")
    if actual_tree != tree:
        raise ValueError("pinned VARE source tree does not match the lock")
    _git("cat-file", "-e", f"{commit}^{{commit}}")
    for relative in source["paths_bound_to_tree"]:
        changed = subprocess.run(
            ["git", "diff", "--quiet", commit, "HEAD", "--", relative],
            cwd=ROOT, check=False,
        )
        if changed.returncode != 0:
            raise ValueError(f"audit input differs from the pinned VARE commit: {relative}")
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=ROOT, check=False,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "HEAD", "--", relative],
            cwd=ROOT, check=False,
        )
        if dirty.returncode != 0 or staged.returncode != 0:
            raise ValueError(f"audit input has local changes: {relative}")
    return {"commit": commit, "tree": actual_tree}


def run_audit(lock: dict[str, Any]) -> dict[str, Any]:
    binding = verify_source_binding(lock)
    protocol_path = ROOT / lock["inventory_protocol"]
    spec = importlib.util.spec_from_file_location(
        "audit_mlsys_update_readiness_v1",
        ROOT / "scripts/audit_mlsys_update_readiness_v1.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v1 inventory auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inventory_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base = module.audit_protocol(inventory_protocol, ROOT)
    lock_digest = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    return {
        "protocol_id": lock["protocol_id"],
        "protocol_sha256": lock_digest,
        "audited_vare_commit": binding["commit"],
        "audited_vare_tree": binding["tree"],
        "source_inputs_match_pinned_commit": True,
        "inventory_audit": base,
        "status": lock["decision"]["status"],
        "eligible_unique_heldout_tasks": base["eligible_unique_heldout_tasks"],
        "required_unique_heldout_tasks": lock["readiness_gate"]["minimum_unique_eligible_heldout_tasks"],
        "audit_passed": (
            base["audit_passed"]
            and base["status"] == lock["decision"]["status"]
            and base["eligible_unique_heldout_tasks"]
            == lock["decision"]["eligible_unique_heldout_tasks"]
            and base["unique_task_definitions_by_sha256"]
            == lock["decision"]["unique_task_definitions"]
        ),
        "claim_boundary": lock["decision"]["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    report = run_audit(lock)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
