"""Synthetic outputs only: no confirmation-cohort task generator is called."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from vare import format_vs_correctness_v2 as legacy
from scripts import summarize_format_vs_correctness_seeded_v3 as command
from reproducers.seeded_summary_fixture_v1 import synthetic_grid
from vare.format_vs_correctness_seeded_summary_v3 import (
    ANALYSIS_ID, ANALYSIS_PATH, ANALYSIS_SHA256, LEGACY_MODULE_SHA256,
    bind_analysis, canonical_analysis_bytes, summarize_seeded_comparison,
)


def curve(summary, arm, k=4):
    return next(row[f"pass_at_{k}"] for row in summary["curves"]
                if row["arm"] == arm and row["split"] == "confirm" and row["checkpoint_step"] == 16)


def test_actual_summary_reverses_pool_ranking_and_keeps_primary_exact():
    confirm, pack, rows = synthetic_grid()
    before = copy.deepcopy(rows)
    old = legacy.summarize_comparison(confirm, pack, rows)
    new = summarize_seeded_comparison(confirm, pack, rows, prospective=False)
    assert curve(old, "format_sft") > curve(old, "correctness_reward_rl")
    assert curve(new, "format_sft") == pytest.approx(0.2)
    assert curve(new, "correctness_reward_rl") == pytest.approx(0.6)
    for field in ("paired_bootstrap", "decision", "decision_split", "by_seed_confirm_final", "promotes_weights"):
        assert json.dumps(new[field], sort_keys=True) == json.dumps(old[field], sort_keys=True)
    assert new["legacy_pooled_curves"] == old["curves"]
    assert new["legacy_any20_mcnemar"] == old["mcnemar"]
    assert "mcnemar" not in new
    assert new["analysis_id"] == ANALYSIS_ID
    assert new["analysis_status"] == "retrospective_analysis"
    assert rows == before


def test_pass_at_one_agrees_and_run_variation_is_not_a_population_interval():
    confirm, pack, rows = synthetic_grid({"format_sft": [4, 0, 0, 0, 0],
                                         "correctness_reward_rl": [4, 4, 0, 0, 0]})
    old = legacy.summarize_comparison(confirm, pack, rows)
    for row in rows:
        row["analysis_sha256"] = ANALYSIS_SHA256
    new = summarize_seeded_comparison(confirm, pack, rows, prospective=True)
    assert new["analysis_file_pin_verified"] is False  # Pure API does not inspect a run directory.
    for old_point, new_point in zip(old["curves"], new["curves"], strict=True):
        assert old_point["pass_at_1"] == pytest.approx(new_point["pass_at_1"], abs=1e-15)
    assert new["seed_contrast_description"]["sample_sd"] > 0
    assert new["seed_contrast_description"]["population_confidence_interval"] is None
    assert new["seed_contrast_description"]["includes_evaluation_sampling_noise"]
    assert new["paired_bootstrap"]["low"] == new["paired_bootstrap"]["high"]
    assert "fixed fitted runs" in new["primary_bootstrap_scope"]
    assert all(row["seed_role"] == "decoding_block" for row in new["by_seed_curves"] if row["arm"] == "base")


@pytest.mark.parametrize("case", ["missing", "duplicate", "unknown_seed", "bool_sample", "bool_checkpoint", "unknown_task"])
def test_incomplete_or_misbound_grid_fails_closed(case):
    confirm, pack, rows = synthetic_grid()
    first = next(i for i, row in enumerate(rows) if row["kind"] == "eval")
    if case == "missing":
        rows.pop(first)
    elif case == "duplicate":
        rows.append(dict(rows[first]))
    else:
        key, value = {"unknown_seed": ("run_seed", -1), "bool_sample": ("sample_index", False),
                      "bool_checkpoint": ("checkpoint_step", False), "unknown_task": ("task_id", "other")}[case]
        rows[first][key] = value
    with pytest.raises(ValueError):
        summarize_seeded_comparison(confirm, pack, rows, prospective=False)


@pytest.mark.parametrize("case", ["duplicate_within", "collision_across", "split_mismatch"])
def test_task_identity_is_not_silently_reweighted(case):
    confirm, pack, rows = synthetic_grid()
    if case == "duplicate_within":
        confirm.append(dict(confirm[0]))
    elif case == "collision_across":
        pack[0]["task_id"] = confirm[0]["task_id"]
    else:
        confirm[0]["split"] = "pack"
    with pytest.raises(ValueError, match="task|split"):
        summarize_seeded_comparison(confirm, pack, rows, prospective=False)


def test_analysis_pin_and_frozen_legacy_files():
    assert ANALYSIS_PATH.read_bytes() == canonical_analysis_bytes()
    assert hashlib.sha256(canonical_analysis_bytes()).hexdigest() == ANALYSIS_SHA256
    assert hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest() == LEGACY_MODULE_SHA256
    training_protocol = ANALYSIS_PATH.with_name("format_vs_correctness_comparison_v2.json")
    assert hashlib.sha256(training_protocol.read_bytes()).hexdigest() == legacy.PROTOCOL_SHA256


def test_bind_analysis_before_outputs_and_reject_changed_pin(tmp_path):
    output = tmp_path / "new"
    bind_analysis(output)
    pin = output / "analysis-protocol.seeded-v3.json"
    assert pin.read_bytes() == canonical_analysis_bytes()
    (output / "progress.jsonl").write_text('{"kind":"train"}\n')
    bind_analysis(output)
    pin.write_text("{}")
    with pytest.raises(ValueError, match="different analysis"):
        bind_analysis(output)
    assert pin.read_text() == "{}"


def test_old_arm_output_cannot_be_silently_declared_prospective(tmp_path):
    (tmp_path / "progress.jsonl").write_text('{"kind":"train"}\n')
    with pytest.raises(ValueError, match="retrospective"):
        bind_analysis(tmp_path)
    assert not (tmp_path / "analysis-protocol.seeded-v3.json").exists()


def synthetic_run(tmp_path, monkeypatch, *, prospective):
    confirm, pack, rows = synthetic_grid()
    run = tmp_path / "run"
    run.mkdir()
    if prospective:
        bind_analysis(run)
    for row in rows:
        row.update({"backend": "mlx", "model_id": "synthetic-only-model",
                    "protocol_sha256": legacy.PROTOCOL_SHA256})
        if prospective:
            row["analysis_sha256"] = ANALYSIS_SHA256
    progress = run / "progress.jsonl"
    progress.write_text("".join(json.dumps(row) + "\n" for row in rows))
    root = tmp_path / "synthetic-root"
    task_path = root / "benchmarks/heldout/exact_integer_v1/tasks.jsonl"
    task_path.parent.mkdir(parents=True)
    task_path.write_text("synthetic fixture only")
    monkeypatch.setattr(command, "ROOT", root)
    monkeypatch.setattr(command, "_protocol", lambda: {"models": {"mlx": {"repo_id": "synthetic-only-model"}}})
    monkeypatch.setattr(command, "confirm_eval_tasks", lambda: confirm)
    monkeypatch.setattr(command, "pack_eval_tasks", lambda _: pack)
    return run


def test_retrospective_command_preserves_raw_and_old_summary(tmp_path, monkeypatch):
    run = synthetic_run(tmp_path, monkeypatch, prospective=False)
    old = run / "summary.json"
    old.write_text("legacy output must remain byte-identical")
    before = (run / "progress.jsonl").read_bytes()
    output = run / "summary.seeded-v3.json"
    with pytest.raises(ValueError, match="retrospective"):
        command.analyze(run, output, retrospective=False)
    command.analyze(run, output, retrospective=True)
    result = json.loads(output.read_text())
    assert result["analysis_status"] == "retrospective_analysis"
    assert result["progress_sha256"] == hashlib.sha256(before).hexdigest()
    assert (run / "progress.jsonl").read_bytes() == before
    assert old.read_text() == "legacy output must remain byte-identical"
    with pytest.raises(FileExistsError):
        command.analyze(run, output, retrospective=True)


def test_prospective_command_checks_every_row_binding(tmp_path, monkeypatch):
    run = synthetic_run(tmp_path, monkeypatch, prospective=True)
    output = run / "summary.seeded-v3.json"
    command.analyze(run, output, retrospective=False)
    assert json.loads(output.read_text())["analysis_status"] == "prospective_rows_bound"
    assert json.loads(output.read_text())["analysis_file_pin_verified"] is True
    progress = run / "progress.jsonl"
    rows = [json.loads(line) for line in progress.read_text().splitlines()]
    del rows[-1]["analysis_sha256"]
    progress.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(SystemExit, match="prospective analysis"):
        command.analyze(run, run / "invalid.json", retrospective=False)
    assert not (run / "invalid.json").exists()


@pytest.mark.parametrize("binding", [None, "wrong-analysis"])
def test_public_api_cannot_claim_prospective_rows_without_binding(binding):
    confirm, pack, rows = synthetic_grid()
    for row in rows:
        if binding is not None:
            row["analysis_sha256"] = binding
    with pytest.raises(ValueError, match="prospective rows"):
        summarize_seeded_comparison(confirm, pack, rows, prospective=True)


@pytest.mark.parametrize("flag", [1, "true"])
def test_public_api_requires_a_boolean_provenance_flag(flag):
    confirm, pack, rows = synthetic_grid()
    with pytest.raises(TypeError, match="Boolean"):
        summarize_seeded_comparison(confirm, pack, rows, prospective=flag)
