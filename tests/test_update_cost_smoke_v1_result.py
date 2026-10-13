from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vare.update_cost_smoke import accept_smoke_record

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/heldout-exact-integer-update-cost-smoke-v1/m1-mlx-f22b3b9"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_retained_smoke_bytes_and_acceptance():
    manifest = json.loads((RESULT / "manifest.json").read_text(encoding="utf-8"))
    assert _sha(RESULT / "manifest.json") == "8926c6c6383201bb3f1a067707210db7abd94c90a10f40ae0686784bc4ce511e"
    blob = (RESULT / "smoke_result.json").read_bytes()
    assert len(blob) == 834 < 1_048_576
    assert _sha(RESULT / "smoke_result.json") == "cf953a5c00cd7cc41a577ca364568e3d9c84fa85062ab8cc99504956af54d97c"
    assert manifest["files"]["smoke_result.json"]["bytes"] == 834
    assert manifest["files"]["smoke_result.json"]["sha256"] == _sha(RESULT / "smoke_result.json")
    assert manifest["adapter_files_not_tracked"]["adapters.safetensors"]["sha256"] == (
        "6cf385ec4224674bfafe7baa6cd8ce7185bdc06a9deba2fa404f4c3699c019ef"
    )
    assert manifest["adapter_files_not_tracked"]["adapter_config.json"]["sha256"] == (
        "4815e87d8fd0d18a5b660f7fb33e3ea98841e5dd0f5d792fd95876c1f7625f78"
    )
    payload = json.loads(blob.decode("utf-8"))
    decision = accept_smoke_record(payload["record"])
    assert decision["accepted"] is True
    assert decision["reasons"] == []
    assert payload["record"]["optimizer_steps"] == 1
    assert payload["record"]["loss"] == 3.004885196685791
    assert payload["record"]["reload_matches_saved"] is True
    assert payload["record"]["heldout_pack_used_for_update"] is False
    assert payload["record"]["resources"]["peak_memory_bytes"] == 1959434978.0
    assert payload["record"]["resources"]["wall_seconds"] == 8.762955541998963
    assert payload["code_revision"] == "f22b3b9455c2ed54bebfe82f90fa47aabbf7479c"
    assert payload["record"]["model_id"] == "mlx-community/Qwen2.5-3B-Instruct-4bit"
