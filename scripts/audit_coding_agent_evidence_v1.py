#!/usr/bin/env python3
"""Independently verify a coding-agent evidence manifest and task claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise ValueError(message)
    checks.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    root = args.evidence.expanduser().resolve()
    manifest_path = root / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[str] = []
    expected = {item["path"]: item for item in manifest["evidence_files"]}
    require(len(expected) == len(manifest["evidence_files"]), "unique manifest paths", checks)
    observed = set()
    for relative, item in expected.items():
        rel = Path(relative)
        require(not rel.is_absolute() and ".." not in rel.parts, f"safe evidence path: {relative}", checks)
        path = root / rel
        require(path.is_file(), f"evidence file exists: {relative}", checks)
        require(path.stat().st_size == item["size_bytes"], f"evidence size matches: {relative}", checks)
        require(sha256(path) == item["sha256"], f"evidence hash matches: {relative}", checks)
        observed.add(rel.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"evidence_manifest.json", "audit.json"}
    }
    require(actual == observed, "manifest file inventory is complete", checks)

    protocol_path = root / "protocol.lock.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(sha256(protocol_path) == manifest["protocol_sha256"], "protocol hash matches manifest", checks)
    model = protocol["models"]["primary"]
    require(model["revision"] == manifest["model"]["revision"], "model revision matches protocol", checks)
    require(model["repository"] == manifest["model"]["repository"], "model repository matches protocol", checks)
    resolved = json.loads((root / "resolved.json").read_text(encoding="utf-8"))
    require(resolved["runner_source_commit"] == manifest["source"]["git_commit"], "source commit matches resolved record", checks)
    require(resolved["runner_source_dirty_at_start"] is manifest["source"]["git_dirty_at_start"], "dirty-tree state matches resolved record", checks)

    rows = json.loads((root / "primary-tasks.json").read_text(encoding="utf-8"))["rows"]
    checked_tasks = []
    for row in rows:
        task_id = row["task_id"]
        task_root = root / "task-roots" / task_id
        require(task_root.is_dir(), f"candidate task root exists: {task_id}", checks)
        require(not (task_root / "evaluator").exists() and not (task_root / "subject").exists(),
                f"hidden grader and source tree absent from stripped task root: {task_id}", checks)
        task_run = root / "primary" / f"seed-{row['seed']}" / task_id
        agent = json.loads((task_run / "agent.json").read_text(encoding="utf-8"))
        grade = json.loads((task_run / "grade.json").read_text(encoding="utf-8"))
        require(agent["model_repository"] == model["repository"], f"agent model repository matches: {task_id}", checks)
        require(agent["model_snapshot"] == model["revision"], f"agent model revision matches: {task_id}", checks)
        before, after = agent["source_hashes_before"], agent["source_hashes_after"]
        changed = before != after
        public_pass = isinstance(agent.get("public_test_result"), dict) and agent["public_test_result"].get("passed") is True
        grader_pass = grade.get("passed") is True
        verified = changed and public_pass and grader_pass
        require(row["task_passed"] is verified, f"verified task result recomputes: {task_id}", checks)
        require(row.get("public_test_passed") is public_pass, f"public test decision matches trajectory: {task_id}", checks)
        require(row.get("grader_passed") is grader_pass, f"grader decision matches independent record: {task_id}", checks)
        diff = agent["candidate_diff"]
        require(hashlib.sha256(diff.encode()).hexdigest() == agent["candidate_diff_sha256"],
                f"candidate diff hash matches trajectory: {task_id}", checks)
        workspace = task_run / "workspaces" / "trial"
        require(workspace.is_dir(), f"candidate workspace retained: {task_id}", checks)
        for name, digest in after.items():
            source = workspace / name
            require(source.is_file() and sha256(source) == digest,
                    f"workspace source hash matches trajectory: {task_id}/{name}", checks)
        checked_tasks.append({"task_id": task_id, "verified_success": verified,
                              "public_test_passed": public_pass, "grader_passed": grader_pass})

    result = {
        "schema_version": 1,
        "status": "pass",
        "evidence_id": manifest["evidence_id"],
        "manifest_sha256": sha256(manifest_path),
        "protocol_sha256": sha256(protocol_path),
        "checks_passed": len(checks),
        "tasks": checked_tasks,
        "claim_limit": "Evidence integrity and result recomputation only; not held-out generalization or VARE reliability benefit.",
    }
    output = args.audit_output.expanduser().resolve() if args.audit_output else root / "audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
