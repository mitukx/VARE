"""Audit a transitive evidence chain for the verifier-integrity v4 run.

The v1 manifests are historical artifacts and are intentionally not rewritten.
The v2 chain inventories their retained bytes, verifies the recovered original
nested manifest, and directly binds every scored outcome to a SHA-256 digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MANIFEST_NAME = "evidence-chain-v2.json"
LOCK_NAME = "evidence-chain-v2.lock.json"
REPORT_NAME = "historical-manifest-audit.json"
PUBLISHED_OUTER = "run-evidence-manifest.json"
PUBLISHED_OUTER_COPY = "run-evidence-manifest.as-published.json"
PUBLISHED_INNER = "sha256-manifest.json"
PUBLISHED_INNER_COPY = "sha256-manifest.as-published.json"
RECOVERED_INNER = "sha256-manifest.recovered-original.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path.name}: {exc}") from exc


def _entry(path: Path, root: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(data),
        "sha256": _sha256(data),
    }


def _safe_artifact_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or not relative or ".." in rel.parts:
        raise ValueError(f"unsafe artifact path: {relative!r}")
    path = root / rel
    if path.is_symlink():
        raise ValueError(f"symlink artifact is not allowed: {relative}")
    resolved_root = root.resolve()
    resolved_path = path.resolve(strict=True)
    if not resolved_path.is_relative_to(resolved_root) or not resolved_path.is_file():
        raise ValueError(f"artifact path escapes evidence root or is not a file: {relative}")
    return resolved_path


def _manifest_rows(path: Path, *, bytes_key: str) -> list[dict[str, Any]]:
    value = _json(path)
    if not isinstance(value, list):
        raise ValueError(f"{path.name} must be a list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} contains a non-object row")
        rel = row.get("path")
        digest = row.get("sha256")
        size = row.get(bytes_key)
        if (not isinstance(rel, str) or not isinstance(digest, str)
                or len(digest) != 64 or not isinstance(size, int) or size < 0):
            raise ValueError(f"{path.name} contains a malformed row: {row!r}")
        if rel in seen:
            raise ValueError(f"{path.name} contains duplicate path {rel!r}")
        seen.add(rel)
        rows.append({"path": rel, "sha256": digest, "bytes": size})
    return rows


def historical_audit(root: Path) -> dict[str, Any]:
    """Reproduce and record the original mismatch without changing v1 files."""
    outer_path = root / PUBLISHED_OUTER
    outer_copy_path = root / PUBLISHED_OUTER_COPY
    inner_path = root / PUBLISHED_INNER
    inner_copy_path = root / PUBLISHED_INNER_COPY
    recovered_path = root / RECOVERED_INNER

    if outer_path.read_bytes() != outer_copy_path.read_bytes():
        raise ValueError("as-published outer manifest copy differs from historical bytes")
    if inner_path.read_bytes() != inner_copy_path.read_bytes():
        raise ValueError("as-published nested manifest copy differs from historical bytes")

    outer_rows = _manifest_rows(outer_path, bytes_key="bytes")
    recovered_rows = _manifest_rows(recovered_path, bytes_key="bytes")
    outer_entry = next((row for row in outer_rows if row["path"] == PUBLISHED_INNER), None)
    if outer_entry is None:
        raise ValueError("historical outer manifest does not bind the nested manifest")

    published_inner_bytes = inner_path.read_bytes()
    recovered_inner_bytes = recovered_path.read_bytes()
    published_inner_digest = _sha256(published_inner_bytes)
    recovered_inner_digest = _sha256(recovered_inner_bytes)
    if (len(recovered_inner_bytes) != outer_entry["bytes"]
            or recovered_inner_digest != outer_entry["sha256"]):
        raise ValueError("recovered original nested manifest does not match historical outer entry")

    published_bad: list[str] = []
    recovered_outer_bad: list[str] = []
    for row in outer_rows:
        artifact = root / row["path"]
        data = recovered_inner_bytes if row["path"] == PUBLISHED_INNER else artifact.read_bytes()
        if len(data) != row["bytes"] or _sha256(data) != row["sha256"]:
            recovered_outer_bad.append(row["path"])
        if not artifact.is_file():
            published_bad.append(row["path"])
        else:
            published = artifact.read_bytes()
            if len(published) != row["bytes"] or _sha256(published) != row["sha256"]:
                published_bad.append(row["path"])

    recovered_bad: list[str] = []
    for row in recovered_rows:
        artifact = root / row["path"]
        data = recovered_inner_bytes if row["path"] == RECOVERED_INNER else artifact.read_bytes()
        if len(data) != row["bytes"] or _sha256(data) != row["sha256"]:
            recovered_bad.append(row["path"])

    published_inventory = _json(inner_path)
    if (not isinstance(published_inventory, dict)
            or not isinstance(published_inventory.get("files"), list)):
        raise ValueError("published generic nested inventory has an invalid schema")
    generic_bad: list[str] = []
    for row in published_inventory["files"]:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("published generic nested inventory has a malformed entry")
        artifact = root / row["path"]
        if (not artifact.is_file()
                or artifact.stat().st_size != row.get("size_bytes")
                or _sha256(artifact.read_bytes()) != row.get("sha256")):
            generic_bad.append(row["path"])

    report = {
        "schema": "vare.verifier-integrity-historical-manifest-audit.v1",
        "historical_outer_manifest": PUBLISHED_OUTER,
        "historical_outer_manifest_sha256": _sha256(outer_path.read_bytes()),
        "preserved_outer_copy": PUBLISHED_OUTER_COPY,
        "historical_nested_manifest": PUBLISHED_INNER,
        "preserved_published_nested_copy": PUBLISHED_INNER_COPY,
        "recovered_original_nested_manifest": RECOVERED_INNER,
        "outer_entry_expected_original": outer_entry,
        "published_nested_actual": {
            "bytes": len(published_inner_bytes),
            "sha256": published_inner_digest,
        },
        "recovered_nested_actual": {
            "bytes": len(recovered_inner_bytes),
            "sha256": recovered_inner_digest,
        },
        "outer_entries": len(outer_rows),
        "published_outer_entries_matching": len(outer_rows) - len(published_bad),
        "published_outer_mismatches": sorted(set(published_bad)),
        "reconstructed_outer_entries_matching": len(outer_rows) - len(recovered_outer_bad),
        "reconstructed_outer_mismatches": sorted(set(recovered_outer_bad)),
        "recovered_original_entries": len(recovered_rows),
        "recovered_original_mismatches": sorted(set(recovered_bad)),
        "published_generic_inventory_entries": len(published_inventory["files"]),
        "published_generic_inventory_mismatches": sorted(set(generic_bad)),
        "historical_failure_preserved": bool(published_bad),
        "original_nested_bytes_recovered": not recovered_outer_bad and not recovered_bad,
    }
    if report["published_outer_mismatches"] != [PUBLISHED_INNER]:
        raise ValueError(f"historical mismatch changed unexpectedly: {report['published_outer_mismatches']}")
    if recovered_outer_bad or recovered_bad or generic_bad:
        raise ValueError("a preserved historical inventory does not validate")
    return report


def _build_manifest(root: Path) -> dict[str, Any]:
    report = historical_audit(root)
    report_path = root / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    excluded = {MANIFEST_NAME, LOCK_NAME}
    files = [
        _entry(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]
    outcomes = []
    for path in sorted((root / "outcomes").glob("*.json")):
        record = _json(path)
        if not isinstance(record, dict) or not isinstance(record.get("task_passed"), bool):
            raise ValueError(f"malformed scored outcome: {path.relative_to(root)}")
        if not isinstance(record.get("candidate"), str) or not isinstance(record.get("image"), str):
            raise ValueError(f"scored outcome lacks candidate/image provenance: {path.name}")
        entry = _entry(path, root)
        outcomes.append({
            **entry,
            "candidate": record["candidate"],
            "task_passed": record["task_passed"],
            "target_sha256": record.get("target_sha256"),
            "image": record["image"],
        })
    return {
        "schema": "vare.verifier-integrity-evidence-chain.v2",
        "run_id": "verifier_integrity_pilot_v4/run1",
        "historical_audit": REPORT_NAME,
        "historical_audit_sha256": _sha256(report_path.read_bytes()),
        "artifact_count": len(files),
        "artifacts": files,
        "scored_outcome_count": len(outcomes),
        "scored_outcomes": outcomes,
    }


def write_chain(root: Path, auditor_path: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(root)
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repository_root = auditor_path.resolve().parents[1]
    auditor_relative = auditor_path.resolve().relative_to(repository_root).as_posix()
    lock = {
        "schema": "vare.verifier-integrity-evidence-chain-lock.v1",
        "manifest": MANIFEST_NAME,
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
        "auditor": auditor_relative,
        "auditor_sha256": _sha256(auditor_path.read_bytes()),
    }
    (root / LOCK_NAME).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_chain(root: Path, auditor_path: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    lock = _json(root / LOCK_NAME)
    if not isinstance(lock, dict) or lock.get("schema") != "vare.verifier-integrity-evidence-chain-lock.v1":
        raise ValueError("invalid evidence chain lock")
    if lock.get("manifest") != MANIFEST_NAME or lock.get("manifest_sha256") != _sha256(manifest_path.read_bytes()):
        raise ValueError("top-level evidence manifest does not match lock")
    repository_root = auditor_path.resolve().parents[1]
    auditor_relative = auditor_path.resolve().relative_to(repository_root).as_posix()
    if lock.get("auditor") != auditor_relative:
        raise ValueError("auditor path differs from lock")
    if lock.get("auditor_sha256") != _sha256(auditor_path.read_bytes()):
        raise ValueError("auditor source hash differs from lock")

    manifest = _json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != "vare.verifier-integrity-evidence-chain.v2":
        raise ValueError("invalid evidence chain manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or manifest.get("artifact_count") != len(artifacts):
        raise ValueError("artifact inventory count mismatch")
    excluded = {MANIFEST_NAME, LOCK_NAME}
    symlinks = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"symlink evidence artifacts are not allowed: {sorted(symlinks)}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    } - excluded
    listed_paths = {
        row.get("path") for row in artifacts if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    if actual_files != listed_paths:
        raise ValueError(
            f"artifact inventory is not exact: missing={sorted(actual_files - listed_paths)}, "
            f"unlisted={sorted(listed_paths - actual_files)}"
        )
    recorded: dict[str, dict[str, Any]] = {}
    for row in artifacts:
        if not isinstance(row, dict):
            raise ValueError("artifact entry must be an object")
        rel = row.get("path")
        if not isinstance(rel, str) or rel in recorded:
            raise ValueError(f"invalid or duplicate artifact path: {rel!r}")
        path = _safe_artifact_path(root, rel)
        actual = _entry(path, root)
        if row.get("sha256") != actual["sha256"] or row.get("bytes") != actual["bytes"]:
            raise ValueError(f"artifact hash/size mismatch: {rel}")
        recorded[rel] = row

    outcomes = manifest.get("scored_outcomes")
    if not isinstance(outcomes, list) or manifest.get("scored_outcome_count") != len(outcomes):
        raise ValueError("scored outcome count mismatch")
    actual_outcomes = {p.relative_to(root).as_posix() for p in (root / "outcomes").glob("*.json")}
    linked_outcomes: set[str] = set()
    for row in outcomes:
        if not isinstance(row, dict):
            raise ValueError("scored outcome entry must be an object")
        rel = row.get("path")
        if not isinstance(rel, str) or rel not in actual_outcomes or rel in linked_outcomes:
            raise ValueError(f"missing, non-outcome, or duplicate scored outcome link: {rel!r}")
        linked_outcomes.add(rel)
        if rel not in recorded or row.get("sha256") != recorded[rel]["sha256"] or row.get("bytes") != recorded[rel]["bytes"]:
            raise ValueError(f"scored outcome is not bound to artifact inventory: {rel}")
        record = _json(_safe_artifact_path(root, rel))
        if (row.get("candidate") != record.get("candidate")
                or row.get("task_passed") != record.get("task_passed")
                or row.get("target_sha256") != record.get("target_sha256")
                or row.get("image") != record.get("image")):
            raise ValueError(f"scored outcome metadata mismatch: {rel}")
    if linked_outcomes != actual_outcomes:
        raise ValueError(f"scored outcome links incomplete: missing={sorted(actual_outcomes - linked_outcomes)}")

    report = historical_audit(root)
    if manifest.get("historical_audit") != REPORT_NAME:
        raise ValueError("manifest does not name historical audit report")
    report_path = _safe_artifact_path(root, REPORT_NAME)
    if (_json(report_path) != report
            or manifest.get("historical_audit_sha256") != _sha256(report_path.read_bytes())):
        raise ValueError("historical audit report is stale or does not reproduce")
    return {
        "status": "pass",
        "artifact_count": len(recorded),
        "scored_outcome_count": len(linked_outcomes),
        "published_outer_entries_matching": report["published_outer_entries_matching"],
        "published_outer_entries": report["outer_entries"],
        "reconstructed_outer_entries_matching": report["reconstructed_outer_entries_matching"],
        "historical_mismatch_preserved": report["historical_failure_preserved"],
        "recovered_original_chain_valid": report["original_nested_bytes_recovered"],
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="write prospective v2 report, manifest and lock")
    args = parser.parse_args()
    root = args.root.resolve()
    auditor_path = Path(__file__).resolve()
    if args.write:
        write_chain(root, auditor_path)
    print(json.dumps(audit_chain(root, auditor_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
