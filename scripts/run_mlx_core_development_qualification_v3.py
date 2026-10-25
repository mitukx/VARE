#!/usr/bin/env python3
"""Run qualification v3 with an explicitly checked protocol binding."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/mlx_core_development_qualification_v3.json"
ENGINE = runpy.run_path(
    str(ROOT / "scripts/run_mlx_core_development_qualification_v1.py"),
    run_name="vare_mlx_qualification_engine_v1",
)
ENGINE["main"].__globals__["PROTOCOL"] = PROTOCOL


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args()
    if Path(ENGINE["main"].__globals__["PROTOCOL"]).resolve() != PROTOCOL.resolve():
        raise RuntimeError("v3 engine protocol binding assertion failed")
    actual_protocol_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if actual_protocol_hash != args.protocol_sha256:
        raise RuntimeError("v3 protocol hash assertion failed before execution")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["build_python_freeze"] != protocol["python_build_environment"]["build_python_freeze"]:
        raise RuntimeError("v3 protocol package freeze mirror is inconsistent")
    try:
        return ENGINE["main"]()
    except Exception as exc:
        if args.output and args.output.is_dir() and not (args.output / "manifest.json").exists():
            ENGINE["write_json_new"](args.output / "partial-failure.json", {
                "schema_version": 1,
                "status": "INCOMPLETE_FAILURE",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            ENGINE["write_manifest"](args.output)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
