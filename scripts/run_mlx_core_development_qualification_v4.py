#!/usr/bin/env python3
"""Run qualification v4 with checked protocol and venv interpreter binding."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/mlx_core_development_qualification_v4.json"
ENGINE = runpy.run_path(
    str(ROOT / "scripts/run_mlx_core_development_qualification_v1.py"),
    run_name="vare_mlx_qualification_engine_v1",
)
ENGINE["main"].__globals__["PROTOCOL"] = PROTOCOL


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-python", type=Path, required=True)
    args, _ = parser.parse_known_args()
    engine_protocol = Path(ENGINE["main"].__globals__["PROTOCOL"]).resolve()
    if engine_protocol != PROTOCOL.resolve():
        raise RuntimeError("v4 engine protocol binding assertion failed")
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != args.protocol_sha256:
        raise RuntimeError("v4 protocol hash assertion failed before execution")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["build_python_freeze"] != protocol["python_build_environment"]["build_python_freeze"]:
        raise RuntimeError("v4 protocol package freeze mirror is inconsistent")
    requested_python = args.build_python.expanduser().absolute()
    if args.build_python.is_symlink() and not requested_python.is_symlink():
        raise RuntimeError("v4 interpreter path normalization discarded the venv symlink")
    actual_freeze = subprocess.check_output(
        [str(requested_python), "-m", "pip", "freeze", "--all"], text=True
    ).splitlines()
    if actual_freeze != protocol["build_python_freeze"]:
        raise RuntimeError("v4 venv package inventory differs before source checkout")
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
