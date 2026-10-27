import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_mlx_lm_1910_smollm2_feasibility_v3 as runner  # noqa: E402


PROTOCOL = ROOT / (
    "results/mlsys-repo-task-mlx-lm-1910-agent-v11-smollm2-360m-toolcall-sequence/protocol.json"
)


def test_v3_verifies_v2_and_its_prior_exposure_chain():
    lock = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    runner._verify_prior_v2_run(lock)

    tampered = dict(lock)
    tampered["prior_development_run"] = dict(lock["prior_development_run"])
    tampered["prior_development_run"]["summary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="summary identity"):
        runner._verify_prior_v2_run(tampered)


def test_v3_runner_keeps_v3_identity_and_supplies_a_sealed_v1_alias(
    monkeypatch, tmp_path
):
    path = tmp_path / "protocol.json"
    lock = {
        "protocol_id": runner.V3_PROTOCOL_ID,
        "status": "FROZEN_BEFORE_RUN",
        "task": {"exposure": "DEVELOPMENT_ONLY"},
    }
    path.write_text(json.dumps(lock), encoding="utf-8")
    path.with_suffix(".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii",
    )
    monkeypatch.setattr(runner, "_verify_prior_v2_run", lambda _: None)
    seen = {}

    def capture_alias(alias_path, model_path, python_exe, git_source):
        seen["lock"] = json.loads(alias_path.read_text(encoding="utf-8"))
        seen["sidecar_matches"] = (
            hashlib.sha256(alias_path.read_bytes()).hexdigest()
            == alias_path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
        )

    monkeypatch.setattr(runner, "ORIGINAL_VERIFY", capture_alias)
    result = runner.verify(path, tmp_path, tmp_path, tmp_path)

    assert result["protocol_id"] == runner.V3_PROTOCOL_ID
    assert seen["lock"]["protocol_id"] == runner.V1_PROTOCOL_ID
    assert seen["sidecar_matches"] is True
