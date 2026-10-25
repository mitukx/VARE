#!/usr/bin/env python3
"""Run qualification v2 through the hash-pinned v1 execution engine."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
ENGINE = runpy.run_path(
    str(ROOT / "scripts/run_mlx_core_development_qualification_v1.py"),
    run_name="vare_mlx_qualification_engine_v1",
)
ENGINE["PROTOCOL"] = ROOT / "protocols/mlx_core_development_qualification_v2.json"


def main() -> int:
    protocol = json.loads(ENGINE["PROTOCOL"].read_text(encoding="utf-8"))
    if protocol["build_python_freeze"] != protocol["python_build_environment"]["build_python_freeze"]:
        raise RuntimeError("v2 protocol package freeze mirror is inconsistent")
    try:
        return ENGINE["main"]()
    except Exception as exc:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--output", type=Path)
        args, _ = parser.parse_known_args()
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
