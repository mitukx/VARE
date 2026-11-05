"""Public probes and hidden graders.

Public probes return measurements the controller may see. Hidden grading uses
different seeds, recomputes metrics in the parent, and ignores any JSON the
candidate writes into the workspace. Expected values are never sent to the worker.

The worker is a same-user subprocess, not a trusted OS sandbox. A passing
grade does not mean memory or network access was confined.

The performance gate is a paired Student-t interval: the lower 95% bound on
time saved must exceed the upper 95% bound of absolute A/A differences. A
fixed speedup ratio is recorded and is not the acceptance rule.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

from vare.isolated_candidate import CandidateHarnessError, paired_difference_ci

from .execute import (
    CLASS_MALFORMED,
    CLASS_MEMORY,
    CLASS_OK,
    CLASS_OUTPUT_LIMIT,
    CLASS_TIMEOUT,
    limits_from_task,
    run_worker,
)
from .reference import (
    accumulated_gradient,
    affine_series,
    descent_step,
    grad_case,
    max_abs_diff,
    median,
    recomputed_mse,
    score_case,
    score_matrix,
)
from .spec import ResearchTask
from .workspace import IntegrityError, source_hashes, stage_subject

PUBLIC_SEED = 11011
# Hidden seeds are intentionally absent from task packs and public observations.
_HIDDEN_GRAD_SEEDS = (44011, 44029, 44087)
_HIDDEN_AFFINE = (
    (44011, -0.7, 1.4),
    (44029, 1.1, 0.3),
    (44087, -1.5, -0.4),
)
_SCORE_PASSES = 6
_BENCH_ROWS = 48
_BENCH_COLS = 48
_BENCH_WIDTH = 16
# Index-aligned samples are only paired if they are taken in one interleaved
# pass. A single burst of five repeats, then a later baseline burst, lets a
# busy host inflate the A/A bound above the time-saved bound even when the
# median ratio is large. The predicate remains gain_low > noise_floor.
_BENCH_REPEATS = 12
_PUBLIC_BENCH_ROWS = 36
_PUBLIC_BENCH_COLS = 36
_PUBLIC_BENCH_WIDTH = 12
_PUBLIC_BENCH_PASSES = 8
_PUBLIC_BENCH_REPEATS = 3
_MSE_LIMIT = 1e-6
_PARAM_LIMIT = 1e-3
_GRAD_ATOL = 1e-8
_SCORE_ATOL = 1e-8


def paired_speed_gate(
    baseline_a: list[float],
    baseline_b: list[float],
    candidate: list[float],
) -> dict[str, float | bool | str]:
    """Accept a speedup only when its confidence bound clears A/A noise.

    ``baseline_a`` and ``baseline_b`` are two timings of the untouched
    implementation. ``candidate`` is timed the same way. The descriptive
    median ratio is not the gate.
    """
    if len(baseline_a) != len(baseline_b) or len(baseline_a) != len(candidate) or len(baseline_a) < 2:
        raise ValueError("paired timing needs at least two matched samples")
    samples = (*baseline_a, *baseline_b, *candidate)
    if not all(math.isfinite(value) and value > 0.0 for value in samples):
        raise ValueError("timing samples must be finite and positive")
    aa_abs = [abs(left - right) for left, right in zip(baseline_a, baseline_b)]
    gain = [left - right for left, right in zip(baseline_a, candidate)]
    try:
        aa_mean, aa_low, aa_high, _, _ = paired_difference_ci(aa_abs)
        gain_mean, gain_low, gain_high, _, _ = paired_difference_ci(gain)
    except CandidateHarnessError as exc:
        raise ValueError("paired interval is undefined") from exc
    noise_floor = max(aa_high, 0.0)
    med_a = median(baseline_a)
    med_b = median(baseline_b)
    med_c = median(candidate)
    return {
        "performance_passed": gain_low > noise_floor,
        "gain_mean_s": gain_mean,
        "gain_low_s": gain_low,
        "gain_high_s": gain_high,
        "aa_abs_mean_s": aa_mean,
        "aa_abs_low_s": aa_low,
        "aa_abs_high_s": aa_high,
        "noise_floor_s": noise_floor,
        "baseline_median_s": med_a,
        "aa_median_s": med_b,
        "candidate_median_s": med_c,
        "aa_relative": abs(med_a - med_b) / max(med_a, 1e-12),
        "speedup": med_a / max(med_c, 1e-12),
        "performance_gate": "paired_ci_exceeds_aa_noise",
    }


def _failure(
    classification: str,
    *,
    failures: list[str],
    hashes: dict[str, str],
    wall_s: float = 0.0,
    peak_rss_kb: int | None = None,
    metrics: dict[str, Any] | None = None,
    correctness_evaluated: bool = False,
) -> dict[str, Any]:
    return {
        "passed": False,
        "classification": classification,
        "correctness_evaluated": correctness_evaluated,
        "correctness_passed": False if correctness_evaluated else None,
        "performance_evaluated": False,
        "performance_passed": None,
        "metrics": metrics or {},
        "failures": failures,
        "source_hashes": hashes,
        "wall_s": wall_s,
        "peak_rss_kb": peak_rss_kb,
    }


def _invoke(
    task: ResearchTask,
    workspace: Path,
    calls: list[dict[str, Any]],
    *,
    limits=None,
    expected_hashes: dict[str, str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        hashes = source_hashes(workspace, task.editable_files)
    except (IntegrityError, OSError, ValueError) as exc:
        return None, _failure("integrity", failures=[str(exc)], hashes={})
    if expected_hashes is not None and expected_hashes != hashes:
        return None, _failure(
            "stale_source",
            failures=["workspace hashes do not match the committed snapshot"],
            hashes=hashes,
        )
    active_limits = limits or limits_from_task(task.compute)
    with tempfile.TemporaryDirectory(prefix="vare-rg-subject-") as tmp:
        subject = Path(tmp) / "subject"
        try:
            stage_subject(workspace, task.editable_files, subject)
        except (IntegrityError, OSError, ValueError) as exc:
            return None, _failure("integrity", failures=[str(exc)], hashes=hashes)
        execution = run_worker(subject, calls, active_limits)
    summary = {
        "classification": execution.classification,
        "wall_s": execution.wall_s,
        "peak_rss_kb": execution.peak_rss_kb,
        "returncode": execution.returncode,
    }
    if execution.classification in {CLASS_TIMEOUT, CLASS_OUTPUT_LIMIT, CLASS_MEMORY}:
        return None, _failure(
            execution.classification,
            failures=[execution.classification],
            hashes=hashes,
            wall_s=execution.wall_s,
            peak_rss_kb=execution.peak_rss_kb,
        )
    if execution.classification != CLASS_OK or not execution.payload or not execution.payload.get("ok"):
        kind = execution.classification if execution.classification != CLASS_OK else CLASS_MALFORMED
        return None, _failure(
            kind,
            failures=[kind],
            hashes=hashes,
            wall_s=execution.wall_s,
            peak_rss_kb=execution.peak_rss_kb,
        )
    results = execution.payload.get("results")
    if not isinstance(results, list):
        return None, _failure(
            CLASS_MALFORMED,
            failures=["worker results missing"],
            hashes=hashes,
            wall_s=execution.wall_s,
            peak_rss_kb=execution.peak_rss_kb,
        )
    by_id = {}
    for row in results:
        if isinstance(row, dict) and "id" in row:
            by_id[row["id"]] = row
    return {
        "results": by_id,
        "hashes": hashes,
        "wall_s": execution.wall_s,
        "peak_rss_kb": execution.peak_rss_kb,
        "execution": summary,
    }, {}


def _result_value(bundle: dict[str, Any], call_id: str) -> tuple[Any, str | None]:
    row = bundle["results"].get(call_id)
    if not isinstance(row, dict):
        return None, "missing worker result"
    if row.get("ok") is not True:
        return None, str(row.get("error_type") or "candidate_error")
    return row.get("value"), None


def public_metrics(task: ResearchTask, workspace: Path) -> dict[str, Any]:
    if task.task_type == "training_correctness":
        return _public_grad(task, workspace)
    if task.task_type == "performance":
        return _public_score(task, workspace)
    if task.task_type == "experiment_debug":
        return _public_affine(task, workspace)
    return _failure("integrity", failures=[f"unsupported task type {task.task_type}"], hashes={})


def grade(
    task: ResearchTask,
    workspace: Path,
    *,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if task.task_type == "training_correctness":
        return _grade_grad(task, workspace, expected_hashes)
    if task.task_type == "performance":
        return _grade_score(task, workspace, expected_hashes)
    if task.task_type == "experiment_debug":
        return _grade_affine(task, workspace, expected_hashes)
    return _failure("integrity", failures=[f"unsupported task type {task.task_type}"], hashes={})


def _public_grad(task: ResearchTask, workspace: Path) -> dict[str, Any]:
    batches, params = grad_case(int(task.reproducibility["public_seed"]))
    bundle, failed = _invoke(task, workspace, [{
        "id": "grad",
        "op": "call",
        "module": "accum",
        "function": "accumulated_gradient",
        "args": [batches, params],
    }])
    if failed:
        failed["metrics"] = {}
        return failed
    assert bundle is not None
    value, error = _result_value(bundle, "grad")
    reference = accumulated_gradient(batches, params)
    if error is not None or not isinstance(value, list):
        return _failure(
            "candidate_error",
            failures=[error or "non-list gradient"],
            hashes=bundle["hashes"],
            wall_s=bundle["wall_s"],
            peak_rss_kb=bundle["peak_rss_kb"],
            correctness_evaluated=True,
        )
    error_abs = max_abs_diff(value, reference)
    denom = sum(item * item for item in reference) ** 0.5
    numer = sum(float(item) * float(item) for item in value) ** 0.5 if all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ) else float("inf")
    scale = numer / denom if denom else float("inf")
    return {
        "passed": None,
        "classification": "public_measurement",
        "correctness_evaluated": False,
        "metrics": {"max_abs_error": error_abs, "scale_ratio": scale},
        "failures": [],
        "source_hashes": bundle["hashes"],
        "wall_s": bundle["wall_s"],
        "peak_rss_kb": bundle["peak_rss_kb"],
    }


def _grade_grad(
    task: ResearchTask,
    workspace: Path,
    expected_hashes: dict[str, str] | None,
) -> dict[str, Any]:
    calls = []
    references = []
    for index, seed in enumerate(_HIDDEN_GRAD_SEEDS):
        batches, params = grad_case(seed)
        references.append(accumulated_gradient(batches, params))
        calls.append({
            "id": f"grad-{index}",
            "op": "call",
            "module": "accum",
            "function": "accumulated_gradient",
            "args": [batches, params],
        })
    calls.append({
        "id": "empty",
        "op": "call",
        "module": "accum",
        "function": "accumulated_gradient",
        "args": [[], [0.0, 0.0, 0.0]],
    })
    bundle, failed = _invoke(task, workspace, calls, expected_hashes=expected_hashes)
    if failed:
        return failed
    assert bundle is not None
    worst = 0.0
    failures: list[str] = []
    for index, reference in enumerate(references):
        value, error = _result_value(bundle, f"grad-{index}")
        if error is not None:
            failures.append(error)
            continue
        worst = max(worst, max_abs_diff(value, reference))
    if worst > _GRAD_ATOL:
        failures.append("hidden gradient residual exceeds tolerance")
    empty = bundle["results"].get("empty")
    if not isinstance(empty, dict) or empty.get("ok") is True or empty.get("error_type") != "ValueError":
        failures.append("empty batch must raise ValueError")
    passed = not failures
    return {
        "passed": passed,
        "classification": "pass" if passed else "fail_correctness",
        "correctness_evaluated": True,
        "correctness_passed": passed,
        "performance_evaluated": False,
        "performance_passed": None,
        "metrics": {"max_abs_error": worst},
        "failures": failures,
        "source_hashes": bundle["hashes"],
        "wall_s": bundle["wall_s"],
        "peak_rss_kb": bundle["peak_rss_kb"],
    }


def _public_score(task: ResearchTask, workspace: Path) -> dict[str, Any]:
    rows, cols = score_case(PUBLIC_SEED, 4, 3, 2)
    bench_rows, bench_cols = score_case(
        PUBLIC_SEED,
        _PUBLIC_BENCH_ROWS,
        _PUBLIC_BENCH_COLS,
        _PUBLIC_BENCH_WIDTH,
    )
    bundle, failed = _invoke(task, workspace, [
        {
            "id": "correct",
            "op": "call",
            "module": "project",
            "function": "score_matrix",
            "args": [rows, cols, 4],
        },
        {
            "id": "bench",
            "op": "bench",
            "module": "project",
            "function": "score_matrix",
            "args": [bench_rows, bench_cols, _PUBLIC_BENCH_PASSES],
            "warmup": 1,
            "repeats": _PUBLIC_BENCH_REPEATS,
        },
    ])
    if failed:
        return failed
    assert bundle is not None
    value, error = _result_value(bundle, "correct")
    reference = score_matrix(rows, cols, 4)
    bench = bundle["results"].get("bench")
    elapsed = bench.get("elapsed_s") if isinstance(bench, dict) else None
    if error is not None or not isinstance(elapsed, list) or not elapsed:
        return _failure(
            "candidate_error",
            failures=[error or "missing public timing"],
            hashes=bundle["hashes"],
            wall_s=bundle["wall_s"],
            peak_rss_kb=bundle["peak_rss_kb"],
            correctness_evaluated=True,
        )
    return {
        "passed": None,
        "classification": "public_measurement",
        "correctness_evaluated": False,
        "metrics": {
            "max_abs_error": max_abs_diff(value, reference),
            "median_elapsed_s": median([float(item) for item in elapsed]),
        },
        "failures": [],
        "source_hashes": bundle["hashes"],
        "wall_s": bundle["wall_s"],
        "peak_rss_kb": bundle["peak_rss_kb"],
    }


def _one_bench_sample(
    task: ResearchTask,
    workspace: Path,
    rows: list,
    cols: list,
) -> tuple[float | None, dict[str, Any] | None, float, int | None]:
    """Time one repetition. Returns sample, failure, wall seconds, peak rss."""
    bundle, failed = _invoke(task, workspace, [{
        "id": "bench",
        "op": "bench",
        "module": "project",
        "function": "score_matrix",
        "args": [rows, cols, _SCORE_PASSES],
        "warmup": 1,
        "repeats": 1,
    }])
    if failed:
        return None, failed, float(failed.get("wall_s") or 0.0), failed.get("peak_rss_kb")
    assert bundle is not None
    row = bundle["results"].get("bench")
    elapsed = row.get("elapsed_s") if isinstance(row, dict) else None
    wall = float(bundle["wall_s"])
    peak = bundle["peak_rss_kb"]
    if not isinstance(elapsed, list) or len(elapsed) != 1:
        return None, _failure(
            "malformed_output",
            failures=["timing sample missing"],
            hashes=bundle["hashes"],
            wall_s=wall,
            peak_rss_kb=peak,
        ), wall, peak
    try:
        sample = float(elapsed[0])
    except (TypeError, ValueError):
        sample = float("nan")
    if not math.isfinite(sample) or sample <= 0.0:
        return None, _failure(
            "unstable_measurement",
            failures=["timing sample is not a positive finite duration"],
            hashes=bundle["hashes"],
            wall_s=wall,
            peak_rss_kb=peak,
        ), wall, peak
    return sample, None, wall, peak


def _grade_score(
    task: ResearchTask,
    workspace: Path,
    expected_hashes: dict[str, str] | None,
) -> dict[str, Any]:
    small_rows, small_cols = score_case(44011, 5, 4, 3)
    bench_rows, bench_cols = score_case(44029, _BENCH_ROWS, _BENCH_COLS, _BENCH_WIDTH)
    candidate, failed = _invoke(
        task,
        workspace,
        [
            {
                "id": "correct",
                "op": "call",
                "module": "project",
                "function": "score_matrix",
                "args": [small_rows, small_cols, 4],
            },
            {
                "id": "passes-one",
                "op": "call",
                "module": "project",
                "function": "score_matrix",
                "args": [small_rows, small_cols, 1],
            },
        ],
        expected_hashes=expected_hashes,
    )
    if failed:
        return failed
    assert candidate is not None
    failures: list[str] = []
    worst = 0.0
    for call_id, passes in (("correct", 4), ("passes-one", 1)):
        value, error = _result_value(candidate, call_id)
        if error is not None:
            failures.append(error)
            continue
        worst = max(worst, max_abs_diff(value, score_matrix(small_rows, small_cols, passes)))
    if worst > _SCORE_ATOL:
        failures.append("score residual exceeds tolerance")
    correctness_passed = not failures
    if not correctness_passed:
        return {
            "passed": False,
            "classification": "fail_correctness",
            "correctness_evaluated": True,
            "correctness_passed": False,
            "performance_evaluated": False,
            "performance_passed": None,
            "metrics": {"max_abs_error": worst},
            "failures": failures,
            "source_hashes": candidate["hashes"],
            "wall_s": candidate["wall_s"],
            "peak_rss_kb": candidate["peak_rss_kb"],
        }
    baseline_workspace = task.root / "workspace"
    series_a: list[float] = []
    series_b: list[float] = []
    elapsed: list[float] = []
    timing_wall = float(candidate["wall_s"])
    peak = candidate["peak_rss_kb"]
    # One index is baseline, candidate, baseline again, so the A/A pair and
    # the saved-time pair see the same host interval.
    for _index in range(_BENCH_REPEATS):
        for dest, timed_workspace in (
            (series_a, baseline_workspace),
            (elapsed, workspace),
            (series_b, baseline_workspace),
        ):
            sample, sample_failed, sample_wall, sample_peak = _one_bench_sample(
                task, timed_workspace, bench_rows, bench_cols,
            )
            timing_wall += sample_wall
            if isinstance(sample_peak, int):
                peak = sample_peak if peak is None else max(peak, sample_peak)
            if sample_failed is not None:
                sample_failed["source_hashes"] = candidate["hashes"]
                sample_failed["wall_s"] = timing_wall
                sample_failed["correctness_evaluated"] = True
                sample_failed["metrics"] = {"max_abs_error": worst}
                return sample_failed
            assert sample is not None
            dest.append(sample)
    try:
        gate = paired_speed_gate(series_a, series_b, elapsed)
    except (TypeError, ValueError):
        return _failure(
            "unstable_measurement",
            failures=["paired timing interval is undefined"],
            hashes=candidate["hashes"],
            wall_s=timing_wall,
            peak_rss_kb=peak,
            correctness_evaluated=True,
            metrics={"max_abs_error": worst},
        )
    performance_passed = gate["performance_passed"] is True
    metrics = {"max_abs_error": worst, **gate}
    failures = [] if performance_passed else [
        "paired time-saved lower bound does not exceed the A/A noise upper bound",
    ]
    return {
        "passed": performance_passed,
        "classification": "pass" if performance_passed else "fail_performance",
        "correctness_evaluated": True,
        "correctness_passed": True,
        "performance_evaluated": True,
        "performance_passed": performance_passed,
        "metrics": metrics,
        "failures": failures,
        "source_hashes": candidate["hashes"],
        "wall_s": timing_wall,
        "peak_rss_kb": peak,
    }


def _public_affine(task: ResearchTask, workspace: Path) -> dict[str, Any]:
    raw_x, raw_y = affine_series(int(task.reproducibility["public_seed"]), 2.0, -1.0)
    step_a, step_b = descent_step(raw_x, raw_y, 0.2, -0.1, 0.25)
    bundle, failed = _invoke(task, workspace, [
        {
            "id": "fit",
            "op": "call",
            "module": "experiment",
            "function": "fit_affine",
            "args": [raw_x, raw_y],
        },
        {
            "id": "step",
            "op": "call",
            "module": "experiment",
            "function": "gradient_step",
            "args": [raw_x, raw_y, 0.2, -0.1, 0.25],
        },
    ])
    if failed:
        return failed
    assert bundle is not None
    fitted, fit_error = _result_value(bundle, "fit")
    stepped, step_error = _result_value(bundle, "step")
    if fit_error or step_error or not isinstance(fitted, dict) or not isinstance(stepped, list):
        return _failure(
            "candidate_error",
            failures=[fit_error or step_error or "unexpected fit payload"],
            hashes=bundle["hashes"],
            wall_s=bundle["wall_s"],
            peak_rss_kb=bundle["peak_rss_kb"],
            correctness_evaluated=True,
        )
    try:
        a = float(fitted["a"])
        b = float(fitted["b"])
        diverged = bool(fitted["diverged"])
    except (KeyError, TypeError, ValueError):
        return _failure(
            "malformed_output",
            failures=["fit payload missing numeric fields"],
            hashes=bundle["hashes"],
            wall_s=bundle["wall_s"],
            peak_rss_kb=bundle["peak_rss_kb"],
        )
    mse = recomputed_mse(raw_x, raw_y, a, b)
    step_residual = max_abs_diff(stepped, [step_a, step_b])
    return {
        "passed": None,
        "classification": "public_measurement",
        "correctness_evaluated": False,
        "metrics": {
            "recomputed_mse": mse,
            "diverged": diverged,
            "gradient_step_abs_error": step_residual,
        },
        "failures": [],
        "source_hashes": bundle["hashes"],
        "wall_s": bundle["wall_s"],
        "peak_rss_kb": bundle["peak_rss_kb"],
    }


def _grade_affine(
    task: ResearchTask,
    workspace: Path,
    expected_hashes: dict[str, str] | None,
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    expectations = []
    for index, (seed, true_a, true_b) in enumerate(_HIDDEN_AFFINE):
        raw_x, raw_y = affine_series(seed, true_a, true_b)
        calls.append({
            "id": f"fit-{index}",
            "op": "call",
            "module": "experiment",
            "function": "fit_affine",
            "args": [raw_x, raw_y],
        })
        expectations.append((raw_x, raw_y, true_a, true_b))
    step_points = ((0.0, 0.0, 0.25), (0.4, -0.2, 0.1), (-0.3, 0.3, 0.05))
    step_refs = []
    raw_x, raw_y = affine_series(44087, -1.5, -0.4)
    for index, (a, b, lr) in enumerate(step_points):
        step_refs.append(descent_step(raw_x, raw_y, a, b, lr))
        calls.append({
            "id": f"step-{index}",
            "op": "call",
            "module": "experiment",
            "function": "gradient_step",
            "args": [raw_x, raw_y, a, b, lr],
        })
    bundle, failed = _invoke(task, workspace, calls, expected_hashes=expected_hashes)
    if failed:
        return failed
    assert bundle is not None
    failures: list[str] = []
    worst_mse = 0.0
    worst_param = 0.0
    worst_step = 0.0
    for index, (series_x, series_y, true_a, true_b) in enumerate(expectations):
        fitted, error = _result_value(bundle, f"fit-{index}")
        if error is not None or not isinstance(fitted, dict):
            failures.append(error or "fit payload missing")
            continue
        try:
            a = float(fitted["a"])
            b = float(fitted["b"])
            diverged = bool(fitted["diverged"])
        except (KeyError, TypeError, ValueError):
            failures.append("fit payload is not numeric")
            continue
        mse = recomputed_mse(series_x, series_y, a, b)
        worst_mse = max(worst_mse, mse)
        param_error = abs(a - true_a) + abs(b - true_b)
        worst_param = max(worst_param, param_error)
        missed = diverged or mse > _MSE_LIMIT or param_error > _PARAM_LIMIT
        if missed and "held-out affine fit missed the acceptance region" not in failures:
            failures.append("held-out affine fit missed the acceptance region")
    for index, reference in enumerate(step_refs):
        value, error = _result_value(bundle, f"step-{index}")
        if error is not None:
            failures.append(error)
            continue
        residual = max_abs_diff(value, list(reference))
        worst_step = max(worst_step, residual)
        if residual > 1e-8 and "gradient step does not match the mean-reduced descent step" not in failures:
            failures.append("gradient step does not match the mean-reduced descent step")
    passed = not failures
    return {
        "passed": passed,
        "classification": "pass" if passed else "fail_correctness",
        "correctness_evaluated": True,
        "correctness_passed": passed,
        "performance_evaluated": False,
        "performance_passed": None,
        "metrics": {
            "max_recomputed_mse": worst_mse,
            "max_param_abs_error": worst_param,
            "max_step_abs_error": worst_step,
        },
        "failures": failures,
        "source_hashes": bundle["hashes"],
        "wall_s": bundle["wall_s"],
        "peak_rss_kb": bundle["peak_rss_kb"],
    }
