"""Artifact rejection must survive Python's optimized execution mode."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("fault", ["modified", "missing", "extra"])
def test_audit_rejects_artifact_fault_before_reporting_success(tmp_path, optimized, fault):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"artifact_sha256": {"raw.json": "0" * 64}}))
    if fault != "missing":
        (bundle / "raw.json").write_text("{}")
    if fault == "extra":
        (bundle / "unexpected.json").write_text("{}")
    script = Path(__file__).resolve().parents[1] / "scripts/audit_checkpoint_read_bounds_v1.py"
    result = subprocess.run([sys.executable, *(["-O"] if optimized else []), str(script), str(bundle)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0 and "ValueError" in result.stderr
    assert result.stdout == ""
