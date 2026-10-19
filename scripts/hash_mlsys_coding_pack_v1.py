"""Write or check the MLSys coding pack manifest.

The manifest lists every pack file except itself. The evaluation protocol
stores the SHA-256 of the manifest bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks" / "mlsys_coding_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path = PACK) -> dict:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        files.append({
            "bytes": path.stat().st_size,
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        })
    return {"files": files, "pack_id": "mlsys-coding-v1"}


def dump_manifest(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = dump_manifest(build_manifest())
    target = PACK / "manifest.json"
    if args.write:
        target.write_text(text, encoding="utf-8")
    stale = not target.is_file() or target.read_text(encoding="utf-8") != text
    if args.check and stale:
        print("manifest mismatch")
        return 1
    print(hashlib.sha256(text.encode("utf-8")).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
