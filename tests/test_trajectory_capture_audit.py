"""Optimized Python must not disable capture artifact checks."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("fault", ["missing", "modified", "extra"])
def test_capture_auditor_rejects_tampered_inventory(tmp_path, optimized, fault):
    (tmp_path / "manifest.json").write_text(json.dumps({"artifact_sha256": {"trace.jsonl": "0" * 64}}))
    if fault != "missing":
        (tmp_path / "trace.jsonl").write_text("{}\n")
    if fault == "extra":
        (tmp_path / "extra").write_text("unexpected")
    script = Path(__file__).resolve().parents[1] / "scripts/audit_trajectory_capture_boundary_v1.py"
    run = subprocess.run([sys.executable, *(["-O"] if optimized else []), str(script), str(tmp_path)],
                         capture_output=True, text=True, timeout=5)
    assert run.returncode != 0 and "ValueError" in run.stderr and not run.stdout
