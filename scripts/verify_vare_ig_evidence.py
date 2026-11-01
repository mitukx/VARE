#!/usr/bin/env python3
"""Verify committed VARE-IG evidence hashes, source snapshot, and trace ledgers."""
from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vare.runner import digest, encode  # noqa: E402

RESEARCH = ROOT / "research/vare_ig_flagship"


def sha256_file(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest_value.update(block)
    return digest_value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_evidence() -> dict[str, int]:
    summary = read_json(RESEARCH / "summary.json")
    evidence_manifest_path = RESEARCH / "evidence/evidence-file-manifest.json"
    evidence_manifest = read_json(evidence_manifest_path)
    if sha256_file(evidence_manifest_path) != summary["evidence"]["file_manifest_sha256"]:
        raise ValueError("evidence file manifest hash differs from summary.json")

    files = evidence_manifest["files"]
    for item in files:
        path = RESEARCH / item["path"]
        if not path.is_file() or sha256_file(path) != item["public_sha256"]:
            raise ValueError(f"public evidence hash mismatch: {item['path']}")
        if item.get("compression") == "gzip":
            with gzip.open(path, "rb") as stream:
                content = stream.read()
            if sha256_bytes(content) != item["uncompressed_public_sha256"]:
                raise ValueError(f"uncompressed evidence hash mismatch: {item['path']}")

    redaction_path = RESEARCH / "evidence/protocol-redaction-manifest.json"
    redaction = read_json(redaction_path)
    if sha256_file(redaction_path) != summary["evidence"]["protocol_redaction_manifest_sha256"]:
        raise ValueError("protocol redaction manifest hash differs from summary.json")
    for item in redaction["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha256_file(path) != item["public_sha256"]:
            raise ValueError(f"public protocol hash mismatch: {item['path']}")
        if len(item["original_sha256"]) != 64:
            raise ValueError(f"missing original protocol hash: {item['path']}")

    source_manifest_path = RESEARCH / "run_source_snapshot/manifest.json"
    source_manifest = read_json(source_manifest_path)
    if sha256_file(source_manifest_path) != summary["evidence"]["run_source_manifest_sha256"]:
        raise ValueError("run-source manifest hash differs from summary.json")
    source_files = source_manifest["locked_source_files"]
    for item in source_files:
        path = RESEARCH / "run_source_snapshot" / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"run-source snapshot hash mismatch: {item['path']}")

    trace_count = 0
    for item in files:
        if not item["path"].endswith("-events.jsonl.gz"):
            continue
        path = RESEARCH / item["path"]
        previous = "0" * 64
        expected_sequence = 0
        with gzip.open(path, mode="rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                event = record["event"]
                if record["sequence"] != expected_sequence:
                    raise ValueError(f"trace sequence mismatch: {item['path']}:{expected_sequence}")
                if event.get("sequence") != expected_sequence or event.get("previous_hash") != previous:
                    raise ValueError(f"trace previous hash mismatch: {item['path']}:{expected_sequence}")
                event_hash = record["event_hash"]
                if digest(encode(event)) != event_hash:
                    raise ValueError(f"trace event hash mismatch: {item['path']}:{expected_sequence}")
                previous = event_hash
                expected_sequence += 1
        if previous != item["ledger_head_sha256"] or expected_sequence != item["events"]:
            raise ValueError(f"trace ledger head/count mismatch: {item['path']}")
        trace_count += expected_sequence

    return {
        "evidence_files": len(files),
        "protocol_files": len(redaction["files"]),
        "run_source_files": len(source_files),
        "trace_events": trace_count,
    }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    result = verify_evidence()
    print(json.dumps({"verified": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
