"""Check retained Metal path measurements without running MLX."""
import hashlib
import json
import math
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results/grpo-metal-path-isolation-v4"
PROTOCOL = "720b923c0199f407fdc2a1c70bc28c248da4b9f93aae75069ec475aab1e1723d"
SOURCE = "faad1de0acd7c20c24d187f89d5d93cdd91d3c8b"
ARMS = {
    "bf16": (
        "0993994e1038089a3de046b629b3c2cb448f4e8584620a85c5573235d610eefe",
        "bddb576fdbf8d52bbbd46ee0d6fd354a8efa000bf6587310cf73c5d7ceafb337",
        ["explicit_selected", "explicit_full"],
        [0.0078125, 0.015625],
    ),
    "fp32": (
        "b753353560341d49123ff43f4bb7395a92419843d0cfe4f2c32a12d036699d47",
        "efbb5014a906c2396fe5deaee6c4174f03bebe6f9670dc8307d307af777b5755",
        [],
        [3.5762786865234375e-7, 7.152557373046875e-7],
    ),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("precision", sorted(ARMS))
def test_retained_metal_path_evidence(precision):
    zip_hash, result_hash, passing, stock = ARMS[precision]
    proto = (ROOT / "protocols/grpo_metal_path_isolation_v4.json").read_bytes()
    assert digest(proto) == PROTOCOL
    p = json.loads(proto)
    manifest = json.loads((FOLDER / "manifest.json").read_bytes())
    assert manifest["measured_commit"] == SOURCE
    assert manifest["protocol_sha256"] == PROTOCOL
    assert sorted(r["precision"] for r in manifest["records"]) == sorted(ARMS)
    row = next(r for r in manifest["records"] if r["precision"] == precision)
    path = FOLDER / f"{precision}.zip"
    assert digest(path.read_bytes()) == zip_hash == row["zip"]
    assert path.stat().st_size == row["size"]
    with zipfile.ZipFile(path) as z:
        assert sorted(z.namelist()) == ["path-environment.txt", "path-preflight.json", "path-result.json"]
        raw = z.read("path-result.json")
        preflight = json.loads(z.read("path-preflight.json"))
    assert digest(raw) == result_hash == row["result"]
    result = json.loads(raw)
    assert result["source_commit"] == SOURCE and result["protocol_sha256"] == PROTOCOL
    assert result["precision"] == precision and result["status"] == "MEASURED"
    assert result["backend"] == preflight == row["preflight"]
    assert preflight["device"] == "Device(gpu, 0)"
    assert result["parameter_digest_before"] == result["parameter_digest_after"] == row["param"]
    assert result["parameter_bytes_unchanged"] is True
    assert result["model_generated_tokens"] == 0
    assert result["optimizer_updated"] is result["heldout_accessed"] is result["task_success_measured"] is False
    assert result["attention"] == row["attention"]
    assert [a["length"] for a in result["attention"]] == p["attention_fixture"]["lengths"]
    assert [a["results"]["stock"]["plain_vs_differentiated_max"] for a in result["attention"]] == stock
    assert all(a["results"]["explicit"]["plain_vs_differentiated_max"] == 0 for a in result["attention"])
    assert result["paths"] == row["paths"]
    assert [(a["context"], a["mode"]) for a in result["paths"]] == [
        (s["id"], mode) for s in p["sequences"] for mode in p["modes"]
    ]
    for r in result["paths"]:
        assert r["gradient"]["finite"] is True
        assert math.isfinite(r["gradient"]["l2"]) and r["gradient"]["l2"] > 0
        plain = abs(r["plain_logprob"] - r["differentiated_logprob"])
        cached = abs(r["cached_prefill_logprob"] - r["differentiated_logprob"])
        assert plain == r["plain_vs_differentiated_max"]
        assert cached == r["cached_prefill_vs_differentiated_max"]
        assert r["within_tolerance"] is (max(plain, cached) <= p["tolerance"])
    assert result["passing_paths"] == passing == row["passing"]
    assert result["passing_paths"] == [
        mode for mode in p["modes"]
        if all(r["within_tolerance"] for r in result["paths"] if r["mode"] == mode)
    ]
