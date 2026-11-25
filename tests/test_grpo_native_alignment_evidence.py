"""Verify retained v2 result bytes and claims without MLX or network access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/grpo-pretrained-sampling-alignment-v2"
SOURCE_COMMIT = "45dfb554b3e4d203a6af9c622411eba4f103a010"
PROTOCOL_HASH = "848574edb5d1c07b83bf5bf7e46e20d5b2d3541bf3e41c5f0ace101c4bc909fb"
ARCHIVES = {
    "bf16": (
        "d28065f1d7ecfd3920d057a1d12557e7ac46be13e581c158195778ab897f896e",
        "de019a87c232d26fa7e17d22c962018bfb3aef31f4356a01c6e9839723f9ef86",
        [8, 8],
    ),
    "fp32": (
        "cf4216cf927540daa929bc5afab8345723110140c2a4f15fbd896519fb32f0f5",
        "00a573b241a9425ea5c3ff8ddff5fd7b46559badf59d7583855f3193146c33c6",
        [3, 4],
    ),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("arm", sorted(ARCHIVES))
def test_retained_native_alignment(arm):
    zip_hash, result_hash, counts = ARCHIVES[arm]
    protocol_bytes = (ROOT / "protocols/grpo_pretrained_sampling_alignment_v2.json").read_bytes()
    assert sha(protocol_bytes) == PROTOCOL_HASH
    protocol = json.loads(protocol_bytes)
    manifest = json.loads((RESULTS / "manifest.json").read_bytes())
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["protocol_sha256"] == PROTOCOL_HASH
    assert sorted(row["arm"] for row in manifest["records"]) == sorted(ARCHIVES)
    row = next(row for row in manifest["records"] if row["arm"] == arm)
    archive = RESULTS / f"{arm}.zip"
    assert sha(archive.read_bytes()) == zip_hash == row["zip_sha256"]
    assert archive.stat().st_size == row["size"]
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == ["alignment-environment.txt", "alignment-result.json"]
        result_bytes = z.read("alignment-result.json")
        env = z.read("alignment-environment.txt").decode()
    assert sha(result_bytes) == result_hash == row["result_sha256"]
    assert "mlx==0.32.2" in env and "mlx-lm==0.32.2" in env
    result = json.loads(result_bytes)
    assert result["source_commit"] == SOURCE_COMMIT
    assert result["protocol_sha256"] == PROTOCOL_HASH
    assert result["status"] == row["final"]["status"] == "PASS"
    assert result["arm"] == arm
    assert result["parameter_digest_before"] == result["parameter_digest_after"] == row["parameter_digest"]
    assert result["parameter_bytes_unchanged"] is True
    assert result["optimizer_updated"] is result["heldout_accessed"] is result["task_success_measured"] is False
    assert result["generated_tokens"] == row["final"]["generated_tokens"] == 16
    assert [group["context"] for group in result["groups"]] == ["short", "long"]
    assert len(row["groups"]) == 2
    for group, logged, expected_count in zip(result["groups"], row["groups"], counts, strict=True):
        alignment = group["alignment"]
        assert group["context"] == logged["context"]
        assert alignment["sampling_score_mode"] == protocol["sampling_score_mode"]
        assert alignment["tolerance"] == protocol["tolerance"] == 1e-6
        assert alignment["would_reject_lag_zero_update"] is logged["would_reject_if_selected"] is False
        assert alignment["n_action_segments"] == len(group["actions"]) == 4
        assert alignment["n_completion_tokens"] == logged["tokens"] == 8
        assert alignment["metrics"] == logged["metrics"]
        assert all(metric["max"] == metric["n_over_tolerance"] == 0
                   for metric in alignment["metrics"].values())
        differences = []
        for action, control in zip(group["actions"], group["legacy_arithmetic_control"], strict=True):
            assert (action["member"], action["turn"]) == (control["member"], control["turn"])
            assert len(action["completion_ids"]) == len(action["sampled_logprobs"]) == 2
            differences.extend(abs(a - b) for a, b in zip(
                action["sampled_logprobs"], control["uncached_trainer_logprobs"], strict=True,
            ))
        legacy = group["legacy_sampled_vs_plain"]
        assert legacy == logged["legacy_sampled_vs_plain"]
        assert max(differences) == legacy["max"]
        assert sum(value > 1e-6 for value in differences) == legacy["n_over_tolerance"] == expected_count
