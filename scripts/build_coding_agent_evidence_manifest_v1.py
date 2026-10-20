#!/usr/bin/env python3
"""Hash every file in one completed coding-agent evidence directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/run_local_agent.py",
    "scripts/run_separated_coding_eval.py",
    "scripts/mlsys_task_isolated_calls.py",
    "scripts/verify_docker_isolation_boundary_v1.py",
    "src/vare/agent_edit_format.py",
    "src/vare/agent_task_contract.py",
    "src/vare/container_isolation.py",
    "src/vare/isolated_candidate.py",
    "src/vare/separated_grading.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    evidence = args.evidence.expanduser().resolve()
    if not evidence.is_dir():
        parser.error("evidence directory does not exist")
    manifest_path = evidence / "evidence_manifest.json"
    if manifest_path.exists():
        parser.error("refusing to overwrite evidence_manifest.json")
    resolved = json.loads((evidence / "resolved.json").read_text(encoding="utf-8"))
    protocol = evidence / "protocol.lock.json"
    model = json.loads(protocol.read_text(encoding="utf-8"))["models"]["primary"]
    source = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    diff = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--binary", "HEAD"],
        check=True, capture_output=True,
    ).stdout
    files = []
    for path in sorted(item for item in evidence.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        files.append({
            "path": path.relative_to(evidence).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    payload = {
        "schema_version": 1,
        "evidence_id": evidence.name,
        "protocol_sha256": sha256(protocol),
        "source": {
            "git_commit": resolved["runner_source_commit"],
            "git_dirty_at_start": resolved["runner_source_dirty_at_start"],
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "runtime_files_sha256": source,
        },
        "model": {
            "repository": model["repository"],
            "revision": model["revision"],
            "license": model["license"],
            "backend": model["backend"],
            "file_hashes": model["file_hashes"],
        },
        "evidence_files": files,
        "claim_limit": "Post-run integrity inventory; not a security certification or a claim of held-out ability.",
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    print(f"files={len(files)} sha256={sha256(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
