#!/usr/bin/env python3
"""Run a frozen two-call tool-format intervention on Development task #1910."""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_mlx_lm_1910_smollm2_feasibility_v1 as base  # noqa: E402
from scripts import run_mlx_lm_1910_smollm2_feasibility_v2 as previous  # noqa: E402

V1_PROTOCOL_ID = "mlsys-repo-task-mlx-lm-1910-smollm2-feasibility-v1"
V2_PROTOCOL_ID = "mlsys-repo-task-mlx-lm-1910-smollm2-tool-call-prompt-v2"
V3_PROTOCOL_ID = "mlsys-repo-task-mlx-lm-1910-smollm2-tool-call-sequence-v3"
# Importing the v2 module patches its shared v1 module object. Retain the
# immutable original verifier exported by v2 instead of the patched global.
ORIGINAL_VERIFY = previous.ORIGINAL_VERIFY


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_sidecar(path: Path) -> None:
    expected = path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
    if _sha256(path) != expected:
        raise ValueError(f"SHA-256 sidecar mismatch: {path.relative_to(ROOT)}")


def _verify_prior_v2_run(lock: dict[str, Any]) -> None:
    prior = lock["prior_development_run"]
    protocol_path = ROOT / prior["protocol_path"]
    if _sha256(protocol_path) != prior["protocol_sha256"]:
        raise ValueError("prior v2 protocol identity mismatch")
    _check_sidecar(protocol_path)
    prior_lock = json.loads(protocol_path.read_text(encoding="utf-8"))
    if prior_lock.get("protocol_id") != V2_PROTOCOL_ID:
        raise ValueError("prior Development attempt is not the locked v2 protocol")
    # Verify v2's retained v1 exposure chain with its original integrity logic.
    previous._verify_prior_run(prior_lock)

    bundle_root = ROOT / prior["bundle_root"]
    bundle_path = ROOT / prior["bundle_manifest_path"]
    if _sha256(bundle_path) != prior["bundle_manifest_sha256"]:
        raise ValueError("prior v2 evidence bundle identity mismatch")
    _check_sidecar(bundle_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("prior v2 bundle does not attest its protocol")
    report_path = ROOT / prior["report_path"]
    if _sha256(report_path) != prior["report_sha256"]:
        raise ValueError("prior v2 report identity mismatch")
    archive_map_path = bundle_root / "workspace-git-archive-map.json"
    _check_sidecar(archive_map_path)
    archive_map = json.loads(archive_map_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / prior["evidence_manifest_path"]
    if _sha256(manifest_path) != prior["evidence_manifest_sha256"]:
        raise ValueError("prior v2 run-manifest identity mismatch")
    _check_sidecar(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if archive_map.get("original_run_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("prior v2 archive map does not attest its run manifest")

    archive_entries = {
        item["run_manifest_path"]: (item, archive["archive_path"])
        for archive in archive_map["archives"]
        for item in archive["entries"]
    }
    for relative, expected in bundle["files"].items():
        path = bundle_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"prior v2 bundled evidence changed: {relative}")
    for relative, expected in manifest["files"].items():
        if "/workspace/.git/" in relative:
            item, archive_relative = archive_entries[relative]
            archive_path = bundle_root / "run-1" / archive_relative
            with tarfile.open(archive_path, "r") as archive:
                content = archive.extractfile(item["archive_member"]).read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"prior v2 archived Git metadata changed: {relative}")
            continue
        path = bundle_root / "run-1" / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"prior v2 raw run evidence changed: {relative}")

    summary_path = ROOT / prior["summary_path"]
    if _sha256(summary_path) != prior["summary_sha256"]:
        raise ValueError("prior v2 summary identity mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_outcome = prior["recorded_outcome"]
    actual = {
        "status": summary.get("status"),
        "independent_successes": summary.get("attempt_successes"),
        "attempts": summary.get("attempt_denominator"),
        "generated_tokens": summary.get("generated_tokens"),
        "optimizer_steps": summary.get("optimizer_steps"),
        "valid_edit_attempts": summary.get("valid_edit_attempts"),
    }
    for key, value in actual.items():
        if expected_outcome.get(key) != value:
            raise ValueError(f"prior v2 outcome changed: {key}")


def verify(protocol_path: Path, model_path: Path, python_exe: Path,
           git_source: Path) -> dict[str, Any]:
    lock = json.loads(protocol_path.read_text(encoding="utf-8"))
    _check_sidecar(protocol_path)
    if (lock.get("protocol_id") != V3_PROTOCOL_ID
            or lock.get("status") != "FROZEN_BEFORE_RUN"):
        raise ValueError("unexpected or unfrozen v3 protocol")
    if lock["task"].get("exposure") != "DEVELOPMENT_ONLY":
        raise ValueError("reused task must remain Development-only")
    _verify_prior_v2_run(lock)

    # Reuse the established task, grader, checkpoint, and resource verifier
    # while preserving strict version identity in the recorded v3 protocol.
    alias = dict(lock)
    alias["protocol_id"] = V1_PROTOCOL_ID
    with tempfile.TemporaryDirectory(prefix="vare-smollm-v3-lock-") as temporary:
        alias_path = Path(temporary) / "protocol.json"
        alias_path.write_text(json.dumps(alias, sort_keys=True) + "\n", encoding="utf-8")
        alias_path.with_suffix(".sha256").write_text(
            _sha256(alias_path) + "\n", encoding="ascii",
        )
        ORIGINAL_VERIFY(alias_path, model_path, python_exe, git_source)
    return lock


base.verify = verify

if __name__ == "__main__":
    raise SystemExit(base.main())
