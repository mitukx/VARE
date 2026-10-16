"""Optional pinned pretrained-model check; the ordinary suite needs no model."""

import hashlib
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("optimized", [False, True])
def test_preflight_failure_is_retained_without_model_dependencies(tmp_path, optimized):
    output = tmp_path / "preflight-failure"
    command = [
        sys.executable,
        *(["-O"] if optimized else []),
        "scripts/validate_grpo_e2e_cpu.py",
        "--model-dir",
        str(tmp_path / "missing-model"),
        "--rvl-source",
        str(tmp_path / "missing-rvl"),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    failure = json.loads((output / "failure.json").read_text())
    assert failure["type"] == "FileNotFoundError"
    assert "missing-model" in failure["error"]


def test_preflight_cannot_overwrite_an_existing_output_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "failure.json"
    sentinel.write_bytes(b"previous evidence\n")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_grpo_e2e_cpu.py",
            "--model-dir",
            "missing",
            "--rvl-source",
            "missing",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert sentinel.read_bytes() == b"previous evidence\n"
    assert {p.name for p in output.iterdir()} == {"failure.json"}


def test_actual_rvl_grpo_path_and_integrity_controls(tmp_path):
    model = os.environ.get("VARE_GRPO_MODEL_DIR")
    rvl = os.environ.get("VARE_E2E_RVL_DIR")
    if not model or not rvl:
        pytest.skip(
            "set VARE_GRPO_MODEL_DIR and VARE_E2E_RVL_DIR for actual-model checks"
        )
    output = tmp_path / "actual-grpo"
    command = [
        sys.executable,
        "scripts/validate_grpo_e2e_cpu.py",
        "--model-dir",
        model,
        "--rvl-source",
        rvl,
        "--output",
        str(output),
        "--rounds",
        "4",
        "--seed",
        "61041",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["protocol_conformant"] is False
    assert summary["resume_next_update_exact"]
    assert summary["promotions"] == 3
    assert summary["rejections"] == 1
    assert len(summary["faults"]) == 8
    assert summary["candidate_updates"] == 4
    assert summary["actual_optimizer_calls_including_fault_and_resume"] == 7
    assert all(x["rejected"] and x["state_unchanged"] for x in summary["faults"])
    rounds = json.loads((output / "rounds.json").read_text())
    assert len(rounds) == 4
    assert [x["round"] for x in rounds] == list(range(4))
    assert all(x["changed_state_tensors"] > 0 for x in rounds)
    assert rounds[-1]["metrics"]["grad_norm"] == 0  # Warm Adam still changes weights.
    assert all(x["advantages"] == [0.0] * len(x["advantages"]) for x in rounds[-1:])
    assert rounds[0]["metrics"]["samples"] == 8
    assert (
        len(
            {x["attempt"]["metadata"]["vare_rollout_group"] for x in rounds[0]["batch"]}
        )
        == 2
    )
    assert len(json.loads((output / "evaluations.json").read_text())) == 8
    manifest = json.loads((output / "SHA256.json").read_text())
    assert set(manifest) == {p.name for p in output.iterdir()} - {"SHA256.json"}
    assert all(
        hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
        for name, digest in manifest.items()
    )
    before = {name: (output / name).read_bytes() for name in manifest}
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert repeated.returncode != 0
    assert before == {name: (output / name).read_bytes() for name in manifest}
    assert not (output / "failure.json").exists()
