#!/usr/bin/env python3
"""Audit qualification v5 through the hash-pinned evidence auditor."""
from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/mlx_core_development_qualification_v5.json"
AUDITOR = runpy.run_path(
    str(ROOT / "scripts/audit_mlx_core_development_qualification_v5.py"),
    run_name="vare_mlx_qualification_auditor_v5",
)
AUDITOR["main"].__globals__["PROTOCOL"] = PROTOCOL


if __name__ == "__main__":
    raise SystemExit(AUDITOR["main"]())
