"""Offline integrity checks for the retained v3 diagnostic archives."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results/grpo-pretrained-sampling-alignment-v3"
SOURCE = "639d7415664033dfba20bfefdc083d0324882ae3"
PROTOCOL = "5e54f7d1bb2b3f8098ca0c7acab45c123e2c33e7c1e56057ab04bc10d7d3aba9"
ARMS = {
    "cpu-q4": (
        "3841c0af320816b1f3abb32386240211460627d72799b072a7168083322594f1",
        "2c4beb9ff3a228ad011bb3cbc33ba714ee61cc3d518f0637a5701da9b064468e",
        "PASS", 0, 0.0, "cpu", 211,
    ),
    "metal-bf16": (
        "581e951197d52e43fd6f63e095d7e3aadb92b0a11a45fd64e6ef9d7fe3e8ef2a",
        "68e8727aeeec98ee73fd3ffbd48b07416cce85884591d27356df628a53d40ece",
        "NO_GO_NUMERICAL_MISMATCH", 4, 0.07882946729660034, "gpu", 0,
    ),
    "metal-q4": (
        "e1200c0132fc80a06b1e677fe72de6ce147cbd860d8c16a0766f4bd88b0c6b95",
        "f1cef47ac27f4be5961e4d522559e48c2daa14a0a1a2eafe074570af99bb92fb",
        "NO_GO_NUMERICAL_MISMATCH", 4, 0.13085639476776123, "gpu", 211,
    ),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_retained_backend_evidence(arm):
    zip_hash, result_hash, status, count, maximum, device, modules = ARMS[arm]
    protocol = ROOT / "protocols/grpo_pretrained_sampling_alignment_v3.json"
    assert digest(protocol.read_bytes()) == PROTOCOL
    manifest = json.loads((FOLDER / "manifest.json").read_bytes())
    assert manifest["measured_commit"] == SOURCE
    assert manifest["protocol_sha256"] == PROTOCOL
    assert sorted(x["arm"] for x in manifest["records"]) == sorted(ARMS)
    row = next(x for x in manifest["records"] if x["arm"] == arm)
    path = FOLDER / f"{arm}.zip"
    assert digest(path.read_bytes()) == zip_hash == row["zip"]
    assert path.stat().st_size == row["size"]
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == [
            "alignment-environment.txt", "alignment-result.json", "backend-preflight.json",
        ]
        raw = archive.read("alignment-result.json")
        preflight = json.loads(archive.read("backend-preflight.json"))
        environment = archive.read("alignment-environment.txt").decode()
    assert digest(raw) == result_hash == row["result"]
    assert "mlx==0.32.2" in environment and "mlx-lm==0.32.2" in environment
    result = json.loads(raw)
    assert result["arm"] == arm
    assert result["source_commit"] == SOURCE
    assert result["protocol_sha256"] == PROTOCOL
    assert result["backend"] == preflight == row["preflight"]
    assert preflight["device"] == f"Device({device}, 0)"
    assert preflight["requested_device"] == device
    assert preflight["status"] == "READY"
    assert result["status"] == row["final"]["status"] == status
    assert result["parameter_digest_before"] == result["parameter_digest_after"] == row["param"]
    assert result["parameter_bytes_unchanged"] is True
    assert result["optimizer_updated"] is result["task_success_measured"] is result["heldout_accessed"] is False
    assert result["quantized_modules"] == modules
    assert result["generated_tokens"] == 4
    group, = result["groups"]
    assert len(group["actions"]) == 4
    assert sorted((a["member"], a["turn"]) for a in group["actions"]) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all(len(a["completion_ids"]) == len(a["sampled_logprobs"]) == 1 for a in group["actions"])
    alignment = group["alignment"]
    assert alignment["n_completion_tokens"] == alignment["n_action_segments"] == 4
    assert alignment["tolerance"] == 1e-6
    assert alignment["metrics"] == row["group"]["metrics"]
    assert alignment["metrics"]["sampled_vs_differentiated"]["n_over_tolerance"] == count
    assert alignment["metrics"]["sampled_vs_differentiated"]["max"] == maximum
    assert alignment["would_reject_lag_zero_update"] is (count > 0)
