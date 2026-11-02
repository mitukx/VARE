#!/usr/bin/env python3
"""Verify the published Sprint IV protocol and hash-chained raw trajectories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify_event_chain(path: Path) -> int:
    previous = "0" * 64
    count = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        event = row["event"]
        if row.get("previous_event_sha256") != previous:
            raise SystemExit(f"{path}: previous event hash mismatch at line {line_no}")
        actual = sha(canonical(event))
        if row.get("event_sha256") != actual:
            raise SystemExit(f"{path}: event hash mismatch at line {line_no}")
        previous = actual
        count += 1
    return count


def main() -> None:
    failures = []
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or sha(path.read_bytes()) != expected:
            failures.append(relative)
    if failures:
        raise SystemExit("bundle SHA-256 mismatch: " + ", ".join(failures))

    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    expected = protocol.pop("sha256")
    if sha(canonical(protocol)) != expected:
        raise SystemExit("frozen protocol hash mismatch")
    recorded = (ROOT / "protocol.sha256").read_text().split()[0]
    if expected != recorded:
        raise SystemExit("protocol.sha256 does not match protocol.json")

    event_files = sorted((ROOT / "attempts").glob("*/events.jsonl"))
    total_events = sum(verify_event_chain(path) for path in event_files)
    print(f"verified_files={len((ROOT / 'SHA256SUMS').read_text().splitlines())}")
    print(f"verified_attempt_event_chains={len(event_files)}")
    print(f"verified_events={total_events}")
    print(f"protocol_sha256={expected}")


if __name__ == "__main__":
    main()
