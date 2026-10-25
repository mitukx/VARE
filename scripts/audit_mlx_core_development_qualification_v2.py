#!/usr/bin/env python3
"""Audit qualification v2 with the hash-pinned v1 evidence auditor."""
from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = runpy.run_path(
    str(ROOT / "scripts/audit_mlx_core_development_qualification_v1.py"),
    run_name="vare_mlx_qualification_auditor_v1",
)
AUDITOR["PROTOCOL"] = ROOT / "protocols/mlx_core_development_qualification_v2.json"


if __name__ == "__main__":
    raise SystemExit(AUDITOR["main"]())
