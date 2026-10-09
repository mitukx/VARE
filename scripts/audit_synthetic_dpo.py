#!/usr/bin/env python3
"""Verify a synthetic DPO bundle's protocol, source, and complete file manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/synthetic_dpo_cpu_v1.json"
LOCK_PATH = ROOT / "protocols/synthetic_dpo_cpu_v1.lock.json"
RUNNER_PATH = ROOT / "scripts/run_synthetic_dpo.py"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def audit(bundle: Path) -> dict[str, object]:
    protocol = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock_body = dict(lock)
    lock_hash = lock_body.pop("sha256", None)
    if lock_body != protocol or lock_hash != canonical_hash(protocol):
        raise ValueError("checked-in protocol and lock do not agree")
    if not bundle.is_dir():
        raise ValueError(f"bundle directory missing: {bundle}")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("manifest files field is not a mapping")
    expected = set()
    for name, digest in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"unsafe manifest path: {name}")
        if not SHA256_RE.fullmatch(str(digest)):
            raise ValueError(f"invalid SHA-256 for {name}")
        path = bundle.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(bundle.resolve()) or not path.is_file():
            raise ValueError(f"missing or escaping manifest file: {name}")
        if sha256_file(path) != digest:
            raise ValueError(f"hash mismatch: {name}")
        expected.add(relative.as_posix())
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*")
              if path.is_file() and path != manifest_path}
    if actual != expected:
        raise ValueError(f"manifest inventory mismatch; missing={sorted(expected-actual)}, unlisted={sorted(actual-expected)}")
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    if summary.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("summary protocol ID mismatch")
    if summary.get("protocol_canonical_sha256") != canonical_hash(protocol):
        raise ValueError("summary protocol hash mismatch")
    snapshot_protocol = bundle / "protocol.json"
    snapshot_lock = bundle / "protocol.lock.json"
    if not snapshot_protocol.is_file() or json.loads(snapshot_protocol.read_text(encoding="utf-8")) != protocol:
        raise ValueError("bundle protocol snapshot differs from current frozen protocol")
    if not snapshot_lock.is_file() or json.loads(snapshot_lock.read_text(encoding="utf-8")) != lock:
        raise ValueError("bundle lock snapshot differs from current frozen lock")
    if summary.get("protocol_file_sha256") != sha256_file(snapshot_protocol):
        raise ValueError("summary protocol file hash mismatch")
    if summary.get("protocol_lock_sha256") != sha256_file(snapshot_lock):
        raise ValueError("summary protocol lock hash mismatch")
    source_snapshot = bundle / "source/run_synthetic_dpo.py"
    if not source_snapshot.is_file() or summary.get("source_script_sha256") != sha256_file(source_snapshot):
        raise ValueError("run source snapshot hash mismatch")
    gradient_check = json.loads((bundle / "gradient-check.json").read_text(encoding="utf-8"))
    if not gradient_check.get("passed") or summary.get("gradient_check") != gradient_check:
        raise ValueError("gradient-check evidence missing, failed, or inconsistent")
    if summary.get("seed_count") != len(protocol["seeds"]):
        raise ValueError("summary seed count mismatch")
    for seed in protocol["seeds"]:
        raw = json.loads((bundle / "seeds" / f"seed-{seed}.json").read_text(encoding="utf-8"))
        if raw.get("seed") != seed:
            raise ValueError(f"seed artifact mismatch: {seed}")
        if len(raw.get("train_examples", [])) != 512 or len(raw.get("heldout_examples", [])) != 256:
            raise ValueError(f"example count mismatch: {seed}")
    return {"status": "verified", "protocol_id": protocol["protocol_id"],
            "seed_count": len(protocol["seeds"]), "manifest_file_count": len(files),
            "acceptance_passed": summary["acceptance_passed"],
            "mean_clean_nll_improvement": summary["clean_dpo_vs_reference_nll_improvement_mean"],
            "paired_bootstrap_95pct": summary["clean_dpo_vs_reference_nll_improvement_paired_bootstrap_95pct"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.bundle.expanduser().resolve())
    except Exception as exc:
        print(f"audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
