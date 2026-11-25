#!/usr/bin/env python3
"""Verify and optionally materialize byte-exact Sprint XI optimizer checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(manifest_path: Path, evidence_root: Path,
           materialize_root: Path | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for checkpoint in manifest["checkpoints"]:
        checkpoint_manifest_path = (
            evidence_root / checkpoint["original_path"]
        ).with_name("manifest.json")
        checkpoint_manifest = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8"),
        )
        declared_hash = checkpoint_manifest["files_sha256"].get("optimizer.pt")
        if declared_hash != checkpoint["original_sha256"]:
            raise ValueError(
                f"optimizer part manifest differs from checkpoint manifest: "
                f"{checkpoint['original_path']}"
            )
        chunks = []
        for part in checkpoint["parts"]:
            data = (evidence_root / part["path"]).read_bytes()
            if len(data) != part["size_bytes"] or sha256_bytes(data) != part["sha256"]:
                raise ValueError(f"checkpoint part integrity failure: {part['path']}")
            chunks.append(data)
        reconstructed = b"".join(chunks)
        if (len(reconstructed) != checkpoint["original_size_bytes"]
                or sha256_bytes(reconstructed) != checkpoint["original_sha256"]):
            raise ValueError(f"reconstructed checkpoint integrity failure: {checkpoint['original_path']}")
        if materialize_root is not None:
            target = materialize_root / checkpoint["original_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(reconstructed)
                stream.flush()
        records.append({
            "path": checkpoint["original_path"],
            "bytes": len(reconstructed),
            "sha256": sha256_bytes(reconstructed),
            "parts": len(chunks),
        })
    return {
        "status": "PASS",
        "manifest_id": manifest["manifest_id"],
        "checkpoint_count": len(records),
        "checkpoints": records,
        "materialized_to": str(materialize_root) if materialize_root is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("results/sprint_xi_20261028/checkpoint-optimizer-parts-v1.json"),
    )
    parser.add_argument("--evidence-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--materialize-root", type=Path,
        help="optional new directory outside the evidence checkout for reconstructed .pt files",
    )
    args = parser.parse_args()
    result = verify(args.manifest.resolve(), args.evidence_root.resolve(),
                    args.materialize_root.resolve() if args.materialize_root else None)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
