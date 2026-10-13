"""Versioned seed-aware analysis of the unchanged v2 learning protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

from . import format_vs_correctness_v2 as legacy

ANALYSIS_ID = "format-vs-correctness-seeded-summary-v3"
ANALYSIS_SHA256 = "4729ad5294f896f4d6e30fef1535fd28006a7b194b7080d6fe659915f060cc1f"
LEGACY_MODULE_SHA256 = "fca8d5174747c1f3c1a8207d179dde83e9ee7b84ccb7767ddd5a33ae2fb96c15"
ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_PATH = ROOT / "protocols/format_vs_correctness_seeded_summary_v3.json"


def analysis_document() -> dict[str, object]:
    return {
        "analysis_id": ANALYSIS_ID,
        "training_protocol_id": legacy.PROTOCOL_ID,
        "training_protocol_sha256": legacy.PROTOCOL_SHA256,
        "legacy_module_sha256": LEGACY_MODULE_SHA256,
        "primary_decision": "preserve the v2 pass@1 task-bootstrap and decision exactly",
        "secondary_pass_at_k": "equal mean over fitted runs of equal task means; k draws from one fitted model",
        "sample_count_per_task_per_run": legacy.EVAL_SAMPLES,
        "run_seeds": list(legacy.RUN_SEEDS),
        "curve_ks": list(legacy.CURVE_KS),
        "bootstrap_resampling_unit": "task; conditional on the fixed fitted runs and observed generations",
        "seed_variation": "descriptive sample SD of per-run primary contrasts; includes evaluation sampling noise",
        "base_run_seed_role": "decoding block only, not independent retraining",
        "legacy_pooled_curves": "retained as a different model-pool estimand",
        "legacy_mcnemar": "descriptive any-success among 20 pooled outputs, not primary pass@1 inference",
        "promotion": False,
    }


def canonical_analysis_bytes() -> bytes:
    return (json.dumps(analysis_document(), indent=2, sort_keys=True) + "\n").encode()


def validate_analysis() -> None:
    blob = ANALYSIS_PATH.read_bytes()
    if blob != canonical_analysis_bytes() or hashlib.sha256(blob).hexdigest() != ANALYSIS_SHA256:
        raise ValueError("seeded analysis protocol bytes changed")
    module = Path(legacy.__file__).read_bytes()
    if hashlib.sha256(module).hexdigest() != LEGACY_MODULE_SHA256:
        raise ValueError("frozen legacy analysis implementation changed")


def bind_analysis(output: Path) -> None:
    """Pin prospective analysis before any output; never relabel an old run."""
    validate_analysis()
    pin = output / "analysis-protocol.seeded-v3.json"
    expected = canonical_analysis_bytes()
    if pin.exists():
        if pin.read_bytes() != expected:
            raise ValueError("run has a different analysis protocol")
        return
    progress = output / "progress.jsonl"
    if progress.exists() and progress.stat().st_size:
        raise ValueError("existing arm output has no seeded analysis pin; use retrospective analysis")
    output.mkdir(parents=True, exist_ok=True)
    with pin.open("xb") as handle:
        handle.write(expected)


def read_progress_snapshot(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Parse and hash one captured blob, without reopening a changing file.

    This binds metrics to the analyzed bytes, not to the latest on-disk version.
    It is not a writer lock or an atomic filesystem snapshot.
    """
    blob = path.read_bytes()
    rows = []
    for line_no, line in enumerate(blob.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_no} is not valid JSON; keep completed records") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {line_no} is not an object")
        rows.append(row)
    return rows, hashlib.sha256(blob).hexdigest()


def _seeded_pass_at(
    grouped: Mapping[tuple[object, ...], Sequence[Mapping[str, object]]],
    tasks: Sequence[Mapping[str, str]],
    *, arm: str, checkpoint_step: int, split: str, k: int,
) -> float:
    # Reuse the frozen scorer and per-run estimator. Never mix fitted models
    # before applying the nonlinear pass@k transformation.
    return fmean(legacy._pass_at(grouped, tasks, arm=arm, checkpoint_step=checkpoint_step,
                                split=split, k=k, run_seed=seed) for seed in legacy.RUN_SEEDS)


def _validate_inputs(confirm: Sequence[Mapping[str, str]], pack: Sequence[Mapping[str, str]],
                     rows: Sequence[Mapping[str, object]]) -> None:
    all_ids: set[str] = set()
    for split, tasks in (("confirm", confirm), ("pack", pack)):
        if not tasks:
            raise ValueError("seeded analysis requires nonempty task splits")
        for task in tasks:
            task_id = task.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id in all_ids:
                raise ValueError("task IDs must be nonempty and unique across splits")
            all_ids.add(task_id)
            if task.get("split", split) != split:
                raise ValueError("task split identity mismatch")
    for row in rows:
        if type(row.get("run_seed")) is not int or row["run_seed"] not in legacy.RUN_SEEDS:
            raise ValueError("row has an unknown or invalid run seed")
        field = "sample_index" if row.get("kind") == "eval" else "step"
        if type(row.get(field)) is not int:
            raise ValueError(f"{field} must be an integer, not a Boolean or coercible value")
        if row.get("kind") == "eval" and type(row.get("checkpoint_step")) is not int:
            raise ValueError("checkpoint_step must be an integer")


def summarize_seeded_comparison(
    confirm: Sequence[Mapping[str, str]], pack: Sequence[Mapping[str, str]],
    rows: Sequence[Mapping[str, object]], *, prospective: bool,
) -> dict[str, Any]:
    validate_analysis()
    if type(prospective) is not bool:
        raise TypeError("prospective must be a Boolean")
    if prospective and any(row.get("analysis_sha256") != ANALYSIS_SHA256 for row in rows):
        raise ValueError("prospective rows must bind the seeded analysis hash")
    _validate_inputs(confirm, pack, rows)
    previous = legacy.summarize_comparison(confirm, pack, rows)
    # The legacy function checks the complete grid and uniqueness before this
    # code averages anything. Missing seed/task/sample rows never get dropped.
    grouped = legacy._index_eval(rows)
    curves: list[dict[str, object]] = []
    by_seed: list[dict[str, object]] = []
    for split, tasks in (("confirm", confirm), ("pack", pack)):
        for arm in legacy.ARMS:
            for step in legacy.checkpoints_for(arm):
                if split == "pack" and step != legacy.final_checkpoint(arm):
                    continue
                point: dict[str, object] = {"arm": arm, "checkpoint_step": step, "split": split}
                for k in legacy.CURVE_KS:
                    point[f"pass_at_{k}"] = _seeded_pass_at(
                        grouped, tasks, arm=arm, checkpoint_step=step, split=split, k=k)
                curves.append(point)
                for seed in legacy.RUN_SEEDS:
                    run_point = {"arm": arm, "checkpoint_step": step, "split": split,
                                 "run_seed": seed, "seed_role": "decoding_block" if arm == "base" else "training_and_decoding_run"}
                    for k in legacy.CURVE_KS:
                        run_point[f"pass_at_{k}"] = legacy._pass_at(
                            grouped, tasks, arm=arm, checkpoint_step=step, split=split, k=k, run_seed=seed)
                    by_seed.append(run_point)
    primary_runs = previous["by_seed_confirm_final"]
    assert isinstance(primary_runs, list)
    contrasts = [float(row["correctness_reward_rl_pass_at_1"]) - float(row["format_sft_pass_at_1"])
                 for row in primary_runs]
    result: dict[str, Any] = dict(previous)
    result["legacy_pooled_curves"] = result.pop("curves")
    result["legacy_any20_mcnemar"] = result.pop("mcnemar")
    result.update({
        "analysis_id": ANALYSIS_ID, "analysis_sha256": ANALYSIS_SHA256,
        "analysis_status": "prospective_rows_bound" if prospective else "retrospective_analysis",
        "analysis_file_pin_verified": False,
        "curves": curves, "by_seed_curves": by_seed,
        "primary_bootstrap_scope": "task-resampling conditional on fixed fitted runs and observed generations; no new-training-seed coverage",
        "seed_contrast_description": {"n_runs": len(contrasts), "mean": fmean(contrasts),
            "sample_sd": stdev(contrasts), "min": min(contrasts), "max": max(contrasts),
            "includes_evaluation_sampling_noise": True, "population_confidence_interval": None},
    })
    return result
