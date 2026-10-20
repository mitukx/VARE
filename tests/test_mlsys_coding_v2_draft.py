"""Draft v2 pack checks. Fixture grades are not a held-out policy result."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks" / "mlsys_coding_v2"
V1 = ROOT / "benchmarks" / "mlsys_coding_v1"
PROTOCOL = ROOT / "protocols" / "mlsys_coding_policy_comparison_v2.json"


def _manifest_text() -> str:
    files = []
    for path in sorted(item for item in PACK.rglob("*") if item.is_file()):
        if path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(PACK).as_posix(),
                "sha256": digest,
            }
        )
    payload = {"files": files, "pack_id": "mlsys-coding-v2", "status": "draft"}
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_draft_split_replaces_causal_attention_and_matches_the_protocol():
    split = json.loads((PACK / "split.json").read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    v1_split = json.loads((V1 / "split.json").read_text(encoding="utf-8"))
    assert split["status"] == "draft"
    assert protocol["status"] == "draft"
    assert protocol["heldout_evaluation_run"] is False
    assert split["train"] == v1_split["train"]
    assert "mlsys-v1-causal-attention" in v1_split["heldout"]
    assert split["heldout"] is None
    assert split["heldout_membership_frozen"] is False
    assert protocol["split"]["heldout"] is None
    assert split["excluded_from_future_heldout"] == ["mlsys-v1-causal-attention"]
    assert "mlsys-v1-causal-attention" not in split["pool_candidates_unassigned"]
    assert set(split["train"]).isdisjoint(split["pool_candidates_unassigned"])
    text = _manifest_text()
    assert (PACK / "manifest.json").read_text(encoding="utf-8") == text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert protocol["draft_pack"]["manifest_sha256"] == digest
    hasher_path = ROOT / "scripts" / "hash_mlsys_coding_pack_v1.py"
    spec = importlib.util.spec_from_file_location("hash_v1", hasher_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    v1_text = module.dump_manifest(module.build_manifest(V1))
    v1_digest = hashlib.sha256(v1_text.encode("utf-8")).hexdigest()
    assert protocol["base_pack"]["manifest_sha256"] == v1_digest
    assert protocol["base_pack"]["previous_manifest_sha256"] == (
        "18c5999f823f9a550a561a913023b87f1fc7bf5a86f481d319e4778212d97499"
    )


def test_power_script_reproduces_the_recorded_table():
    pytest.importorskip("numpy")
    completed = subprocess.run(
        [sys.executable, str(PACK / "power.py")],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    proposal = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    recorded = {
        (row["heldout_tasks"], row["k"]): row
        for row in proposal["power"]["minimum_detectable_effect_at_80_percent_power"]
    }
    assert "baseline mean success 35.4%" in completed.stdout
    parsed = 0
    for line in completed.stdout.splitlines():
        if not line.startswith("heldout_tasks="):
            continue
        matched = re.fullmatch(
            r"heldout_tasks=\s*(\d+)\s+k=\s*(\d+)\s+seeds=\s*(\d+)\s+"
            r"MDE_logit=(\S+)\s+MDE_pp=([0-9.]+)\s+type1=([0-9.]+)",
            line,
        )
        assert matched is not None, line
        key = (int(matched.group(1)), int(matched.group(2)))
        row = recorded[key]
        assert float(matched.group(4)) == pytest.approx(row["mde_logit"])
        assert float(matched.group(5)) == pytest.approx(row["mde_percentage_points"])
        assert round(float(matched.group(6)), 3) == row["type_i"]
        parsed += 1
    assert parsed == len(recorded)
