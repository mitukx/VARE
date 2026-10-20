"""Runner gates for the MLSys comparison. Synthetic policies only."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vare.mlsys_coding_comparison import (
    FROZEN_MANIFEST_SHA256,
    GROUP_SIZE,
    HeldOutAccessError,
    RealGradingRefused,
    assess_execution_phases,
    grade_real_model_patch,
    grpo_update,
    holm_rejects,
    paired_task_t_test,
    probability_arm_beats_baseline,
    prove_hidden_test_confidentiality,
    sft_update,
    student_t_sf,
    task_cluster_accepts,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_V1 = ROOT / "protocols" / "mlsys_coding_policy_comparison_v1.json"
PROTOCOL_V2 = ROOT / "protocols" / "mlsys_coding_policy_comparison_v2.json"


def test_v1_protocol_is_unchanged_and_refuses():
    protocol = json.loads(PROTOCOL_V1.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "mlsys-coding-policy-comparison-v1"
    assert protocol["authorization"]["status"] == "not_authorized"
    assert protocol["runner"]["implemented"] is False
    assert protocol["metrics"]["acceptance"] == (
        "For each of the two primary contrasts, all three seeds have a strictly "
        "positive paired difference and the 2.5 percentile of the 10000-resample "
        "seed bootstrap of the mean paired difference is greater than 0. Report "
        "both contrasts. A point estimate alone is not acceptance."
    )
    assert protocol["task_pack_manifest_sha256"] == FROZEN_MANIFEST_SHA256
    assert protocol["budgets"]["evaluation_identical_across_arms"]["seeds"] == [
        101,
        211,
        307,
    ]
    assert (
        protocol["budgets"]["evaluation_identical_across_arms"]["decoding"][
            "temperature"
        ]
        == 0
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_mlsys_coding_policy_comparison_v1.py"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 3
    payload = json.loads(completed.stdout)
    assert payload["status"] == "not_run"
    assert any("not implemented" in reason for reason in payload["reasons"])
    assert "torch" not in completed.stdout


def test_live_canary_is_not_confidential(tmp_path: Path):
    proof = prove_hidden_test_confidentiality(tmp_path)
    assert proof["proven"] is False
    assert proof["candidate_read_canary"] is True
    assert proof["separate_containers"] is False
    assert proof["docker_started"] is False
    assert proof["phase_roles"] == ["combined"]


def test_separate_grader_phase_passes_canary(tmp_path: Path):
    token = "CANARY-separate-phase"
    task = tmp_path / "task"
    (task / "evaluator").mkdir(parents=True)
    (task / "evaluator" / "hidden_canary.txt").write_text(
        token + "\n", encoding="utf-8"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text("VALUE = 0\n", encoding="utf-8")
    separated = assess_execution_phases(
        [
            {
                "role": "candidate",
                "mounts": [{"source": str(workspace), "target": "/workspace"}],
            },
            {"role": "grader", "mounts": [{"source": str(task), "target": "/grader"}]},
        ],
        canary_token=token,
        work=tmp_path / "separated",
    )
    assert separated["proven"] is True
    assert separated["candidate_read_canary"] is False
    assert separated["grader_read_canary"] is True
    assert separated["candidate_exited_before_grader"] is True
    combined = assess_execution_phases(
        [
            {
                "role": "combined",
                "mounts": [
                    {"source": str(workspace), "target": "/workspace"},
                    {"source": str(task), "target": "/grader"},
                ],
            }
        ],
        canary_token=token,
        work=tmp_path / "combined",
    )
    assert combined["proven"] is False
    assert combined["candidate_read_canary"] is True


def test_real_model_patch_is_not_graded():
    called = False

    def grader(patch: bytes) -> dict:
        nonlocal called
        called = True
        return {"patch": patch}

    with pytest.raises(RealGradingRefused) as raised:
        grade_real_model_patch(b"diff --git a/x b/x\n", grader)
    assert called is False
    assert raised.value.decision["allowed"] is False
    assert raised.value.decision["canary"]["candidate_read_canary"] is True


def test_heldout_files_stay_closed(monkeypatch: pytest.MonkeyPatch):
    from vare.mlsys_coding_comparison import FrozenPack

    opened: list[str] = []
    original = Path.read_text

    def spy(self: Path, *args, **kwargs):
        opened.append(str(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    pack = FrozenPack()
    heldout = pack.heldout_ids[0]
    before = len(opened)
    with pytest.raises(HeldOutAccessError):
        pack.task_directory(heldout, role="train")
    with pytest.raises(HeldOutAccessError):
        pack.oracle_file(heldout, role="tune")
    with pytest.raises(HeldOutAccessError):
        pack.reject_unless_train(pack.train_ids[0], role="eval")
    leaked = [
        path
        for path in opened[before:]
        if heldout.removeprefix("mlsys-v1-").replace("-", "_") in path
    ]
    assert leaked == []


def test_synthetic_updates_use_minigrpo_and_keep_the_group():
    with pytest.raises(RuntimeError, match="truncated"):
        grpo_update([True, False], [1.0, 0.0])
    grpo = grpo_update([True, False, True, False], [1.0, 0.0, 1.0, 0.0])
    assert grpo["group_size"] == GROUP_SIZE
    assert len(grpo["advantages"]) == GROUP_SIZE
    assert grpo["theta"] > 0.0
    assert grpo["loss_after"] < grpo["loss_before"]
    sft = sft_update()
    assert sft["theta"] > 0.0
    assert sft["loss_after"] < sft["loss_before"]
    assert sft["objective"] == "bernoulli_nll_oracle_patch"


def test_power_records_match_the_closed_form():
    proposal = json.loads(PROTOCOL_V2.read_text(encoding="utf-8"))
    assert proposal["protocol_id"] == "mlsys-coding-policy-comparison-v2"
    assert proposal["status"] == "draft"
    assert proposal["heldout_evaluation_run"] is False
    assert proposal["authorization"]["status"] == "not_authorized"
    assert proposal["runner"]["implemented"] is False
    assert proposal["split"]["heldout"] is None
    assert proposal["split"]["heldout_membership_frozen"] is False
    assert proposal["split"]["heldout_manifest_sha256"] is None
    assert proposal["split"]["current_v1_heldout_pool_size"] == 4
    assert "20-30 task expansion" in proposal["split"]["depends_on"]
    assert (
        "mlsys-v1-causal-attention" in proposal["split"]["excluded_from_future_heldout"]
    )
    assert "mlsys-v2-gqa-repeat" in proposal["split"]["pool_candidates_unassigned"]
    assert proposal["split"]["train"] == [
        "mlsys-v1-grad-accum-tail",
        "mlsys-v1-paired-shuffle-loader",
        "mlsys-v1-range-sums",
        "mlsys-v1-row-log-softmax",
        "mlsys-v1-warmup-cosine",
    ]
    acceptance = proposal["metrics"]["acceptance"]
    assert "paired t-test" in acceptance
    assert "Holm" in acceptance
    assert "does not accept or reject" in acceptance
    assert proposal["metrics"]["primary_contrasts"] == [
        "grpo minus no_update",
        "sft minus no_update",
    ]
    assert proposal["metrics"]["alpha"] == 0.05
    exact = proposal["power"]["v1_exact"]
    realistic = probability_arm_beats_baseline(4, 0.05, 0.30)
    assert exact["greedy_temperature_0_power_at_realistic_effect"] == pytest.approx(
        realistic
    )
    assert exact["independent_seed_power_at_realistic_effect"] == pytest.approx(
        realistic**3
    )
    assert exact["independent_seed_power_at_realistic_effect"] < 0.8
    assert exact["greedy_temperature_0_power_at_realistic_effect"] < 0.8
    assert proposal["power"]["v1_detects_realistic_effect"] is False
    assert proposal["power"]["v1_at_baseline_0_5"]["absolute_lift_for_power_0_80"] == (
        0.491
    )
    assert proposal["power"]["holm_adjusted_mde"].startswith("not recomputed")
    assert "near 0" in proposal["power"]["floor_effect"]
    assert proposal["power"]["script"] == "benchmarks/mlsys_coding_v2/power.py"
    rows = proposal["power"]["minimum_detectable_effect_at_80_percent_power"]
    by_key = {(row["heldout_tasks"], row["k"]): row for row in rows}
    assert by_key[(10, 8)]["mde_percentage_points"] == 15.5
    assert by_key[(10, 32)]["mde_percentage_points"] == 8.5
    assert by_key[(20, 8)]["mde_percentage_points"] == 12.0
    assert by_key[(20, 32)]["mde_percentage_points"] == 6.8
    assert by_key[(30, 8)]["mde_percentage_points"] == 8.5
    assert by_key[(30, 32)]["mde_percentage_points"] == 6.8
    assert by_key[(15, 32)]["mde_percentage_points"] == 8.5
    assert "about 8 percentage points" in proposal["power"]["recommendation"]
    assert student_t_sf(2.446911851144167, 6) == pytest.approx(0.025, abs=1e-9)
    assert proposal["training_gate"]["prerequisite_for_heldout_power"]
    assert "cannot reject" in proposal["v1_seed_bootstrap"]
    assert "does not keep that conjunction" in proposal["v1_seed_bootstrap"]
    assert "all three seeds" not in proposal["metrics"]["acceptance"]


def test_paired_t_and_holm_match_the_draft_rule():
    positive = paired_task_t_test([0.2, 0.2, 0.2, 0.2])
    assert positive["p"] == 0.0
    flat = paired_task_t_test([0.0, 0.0, 0.0, 0.0])
    assert flat["p"] == 0.5
    assert holm_rejects([0.02, 0.04]) == [True, True]
    assert holm_rejects([0.03, 0.04]) == [False, False]
    assert holm_rejects([0.01, 0.06]) == [True, False]
    assert holm_rejects([0.06, 0.01]) == [False, True]


def test_task_cluster_interval_is_not_the_acceptance_rule():
    proposal = json.loads(PROTOCOL_V2.read_text(encoding="utf-8"))
    assert "only acceptance" not in proposal["metrics"]["acceptance"]
    assert task_cluster_accepts([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], resamples=99, seed=1)
    assert (
        task_cluster_accepts([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], resamples=99, seed=1)
        is False
    )
    # One negative seed can sit inside a positive task mean. The rule looks at
    # the task means, not at a requirement that every seed difference is positive.
    assert task_cluster_accepts([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], resamples=99, seed=2)


def test_synthetic_train_dry_run(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_mlsys_coding_policy_comparison_v1.py"),
            "--synthetic-train-dry-run",
            "--skip-performance-proxy",
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "synthetic_dry_run"
    assert summary["capability_claim"] is False
    assert summary["protocol_budget_consumed"] is False
    assert summary["heldout_tasks_read"] == []
    assert summary["heldout_probe"]["rejected"] is True
    assert summary["real_grading"]["allowed"] is False
    assert summary["arms"]["no_update"]["gradient_updates"] == 0
    assert summary["arms"]["no_update"]["split"] == "train"
    assert summary["arms"]["sft"]["split"] == "train"
    assert summary["arms"]["sft"]["theta"] > 0.0
    assert summary["arms"]["grpo"]["admitted_group"] is True
    assert summary["arms"]["grpo"]["trajectory_count"] == GROUP_SIZE
    assert summary["arms"]["grpo"]["theta"] > 0.0
    assert (
        summary["arms"]["grpo"]["loss_after"] < summary["arms"]["grpo"]["loss_before"]
    )
    assert summary["cpu_policy_step"]["theta"] > 0.0
    stdout = json.loads(completed.stdout)
    assert stdout["capability_claim"] is False
    assert stdout["real_grading_allowed"] is False
    assert "torch" not in completed.stdout
