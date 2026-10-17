"""Optional pinned pretrained-model check; the ordinary suite needs no model."""

import hashlib
import json
import os
import shutil
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
    assert summary["checkpoint_load_through_next_candidate_s"] > 0
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
    samples = json.loads((output / "stage_samples_s.json").read_text())
    assert len(samples["rollout"]) == 32  # Fault canaries are outside main timing.
    for key, values in samples.items():
        metrics = summary["stages"][key]
        assert metrics["n"] == len(values)
        assert metrics["total_s"] == sum(values)
        assert metrics["p50_s"] == sorted(values)[len(values) // 2]
        assert (
            metrics["p95_s"]
            == sorted(values)[__import__("math").ceil(len(values) * 0.95) - 1]
        )
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
    audited = subprocess.run(
        [
            sys.executable,
            "scripts/audit_grpo_e2e_cpu.py",
            str(output),
            "--model-dir",
            model,
            "--rvl-source",
            rvl,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert audited.returncode == 0, audited.stderr
    assert json.loads(audited.stdout)["candidate_updates"] == 4
    replay = json.loads(audited.stdout)["unclipped_reference_replay"]
    assert replay["pass"] is True
    assert replay["nonzero_batch_scaling_controls"] == 3
    # A batch-normalization error can vanish after clipping. Retain the same
    # clipped vectors/parameters but corrupt and re-seal the recorded raw norm.
    wrong_scale = tmp_path / "rehashed-wrong-batch-scale"
    shutil.copytree(output, wrong_scale, copy_function=os.link)
    changed = json.loads((wrong_scale / "rounds.json").read_text())
    changed[0]["metrics"]["grad_norm"] *= len(changed[0]["batch"])
    (wrong_scale / "rounds.json").unlink()
    (wrong_scale / "rounds.json").write_text(json.dumps(changed))
    resealed = dict(manifest)
    resealed["rounds.json"] = hashlib.sha256(
        (wrong_scale / "rounds.json").read_bytes()
    ).hexdigest()
    (wrong_scale / "SHA256.json").unlink()
    (wrong_scale / "SHA256.json").write_text(json.dumps(resealed))
    guarded = subprocess.run(
        [
            sys.executable,
            "scripts/audit_grpo_e2e_cpu.py",
            str(wrong_scale),
            "--model-dir",
            model,
            "--rvl-source",
            rvl,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        guarded.returncode != 0 and "unclipped gradient norm mismatch" in guarded.stderr
    )
    # Re-seal the bytes: this control targets a semantic missing-round error,
    # rather than a trivially broken file checksum.
    bad = tmp_path / "rehashed-missing-round"
    shutil.copytree(output, bad, copy_function=os.link)
    rounds.pop()
    (bad / "rounds.json").unlink()
    (bad / "rounds.json").write_text(json.dumps(rounds))
    manifest["rounds.json"] = hashlib.sha256(
        (bad / "rounds.json").read_bytes()
    ).hexdigest()
    (bad / "SHA256.json").unlink()
    (bad / "SHA256.json").write_text(json.dumps(manifest))
    denied = subprocess.run(
        [sys.executable, "scripts/audit_grpo_e2e_cpu.py", str(bad)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert denied.returncode != 0 and "missing candidate update" in denied.stderr
