"""Materialize parent-owned isolated calls from the frozen MLSys task grader.

The task evaluator stays outside the agent workspace. It is imported only by
this operator-side adapter; candidate execution uses VARE's existing isolated
subject grader and receives arguments, never expected outputs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import Any


TASKS = {
    "mlsys-v1-aging-scheduler": ("test_hidden.py", "aging_scheduler", "schedule", 4, "correctness"),
    "mlsys-v1-causal-attention": ("test_hidden.py", "causal_attention", "causal_attention_weights", 5, "correctness"),
    "mlsys-v1-grad-accum-tail": ("test_hidden.py", "grad_accum", "average_microbatch_grads", 3, "correctness"),
    "mlsys-v1-global-grad-clip": ("test_hidden.py", "global_grad_clip", "clip_by_global_norm", 4, "correctness"),
    "mlsys-v1-paired-shuffle-loader": ("test_hidden.py", "paired_loader", "iter_batches", 6, "correctness"),
    "mlsys-v1-warmup-cosine": ("test_hidden.py", "warmup_cosine", "learning_rate", 12, "correctness"),
    "mlsys-v1-range-sums": ("grade.py", "range_sums", "range_sums", 9, "performance"),
    "mlsys-v3-ema-bias": ("test_hidden.py", "ema_bias", "bias_corrected_ema", 5, "correctness"),
    "mlsys-v3-label-smooth": ("test_hidden.py", "label_smooth", "smooth_targets", 7, "correctness"),
}

ROUTABLE_TASKS = frozenset({
    "mlsys-v1-causal-attention",
    "mlsys-v1-warmup-cosine",
    "mlsys-v3-ema-bias",
})


class IsolatedTaskAdapterMissing(ValueError):
    """No trusted operator-side adapter exists for this task id."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_evaluator(path: Path, pack_root: Path, task_id: str):
    candidates = [pack_root, path.parent]
    if task_id == "mlsys-v1-warmup-cosine" and pack_root.name == "mlsys_coding_v3":
        # The v3 copy freezes the same generic adapter import as v1, but does
        # not duplicate that helper at the pack root.
        candidates.append(pack_root.parent / "mlsys_coding_v1")
    for candidate in candidates:
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    name = "vare_frozen_grader_" + task_id.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trusted evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))


def materialize_isolated_calls(task_id: str, task_dir: Path) -> tuple[dict, dict]:
    """Run the frozen reference generator, retaining only operator-side expectations."""
    if task_id not in TASKS:
        raise IsolatedTaskAdapterMissing(f"task has no isolated adapter: {task_id}")
    grader_name, module_name, function_name, count, task_kind = TASKS[task_id]
    evaluator = Path(task_dir) / "evaluator" / grader_name
    pack_root = Path(task_dir).parents[1]
    module = _load_evaluator(evaluator, pack_root, task_id)
    generator = getattr(module, "_generator", None)
    reference = getattr(module, "_reference", None)
    if not callable(generator) or not callable(reference):
        raise ValueError(f"frozen evaluator lacks generator/reference: {task_id}")

    indices = range(2, count) if task_id == "mlsys-v1-range-sums" else range(count)
    cases = []
    skipped = []
    for index in indices:
        try:
            drawn = generator(index)
            if type(drawn) is not tuple or len(drawn) != 2:
                raise ValueError("evaluator generator returned an invalid shape")
            args, kwargs = drawn
        except ValueError as exc:
            raise ValueError(f"frozen evaluator could not generate case {index}: {task_id}") from exc
        try:
            expected = reference(*args, **kwargs)
        except ValueError:
            if task_kind != "correctness":
                skipped.append(index)
                continue
            try:
                args_json = _json_value(list(args))
                kwargs_json = _json_value(kwargs)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"frozen evaluator case {index} is not JSON serializable: {task_id}"
                ) from exc
            row = {
                "case_id": f"{task_id}-grader-{index}",
                "args": args_json,
                "kwargs": kwargs_json,
                "expected": None,
                "expected_exception": "ValueError",
                "expected_exception_prefix": "ValueError",
            }
            cases.append(row)
            continue
        row = {
            "case_id": f"{task_id}-grader-{index}",
            "args": _json_value(list(args)),
            "kwargs": _json_value(kwargs),
            "expected": _json_value(expected),
        }
        checker = getattr(module, "_checker", None)
        if callable(checker):
            checker(expected, expected, list(args), dict(kwargs), args, kwargs)
        cases.append(row)

    if not cases:
        raise ValueError(f"frozen evaluator produced no isolated calls: {task_id}")
    spec = {
        "schema_version": 1,
        "task_id": task_id,
        "task_kind": task_kind,
        "module": module_name,
        "function": function_name,
        "rtol": 0.0,
        "atol": 1e-9 if task_id == "mlsys-v1-causal-attention" else 1e-12,
        "timeout_s": (
            60 if task_id == "mlsys-v1-range-sums"
            else 20 if task_id.startswith("mlsys-v3-")
            else 10
        ),
        "memory_mb": int(json.loads((Path(task_dir) / "task.json").read_text())["limits"]["memory_mb"]),
        "preserve_session": task_kind == "correctness",
        "discard_candidate_stdout": task_id in ROUTABLE_TASKS,
    }
    if task_id in {"mlsys-v1-causal-attention", "mlsys-v1-warmup-cosine"}:
        spec["bool_as_number"] = True
    if task_id == "mlsys-v1-range-sums":
        spec["tuple_paths"] = [[1, "*"]]
    if task_kind == "correctness":
        spec["calls"] = cases
    else:
        source = inspect.getsource(reference)
        # Frozen evaluators call their oracle ``_reference`` while the
        # isolated grader imports the baseline through the public task API.
        # Keep the oracle body unchanged and expose it under that API name.
        source = source.replace("def _reference(", f"def {function_name}(", 1)
        if f"def {function_name}(" not in source:
            raise ValueError(f"frozen evaluator reference has an unexpected definition: {task_id}")
        correctness_cases = []
        for index in range(2):
            args, kwargs = generator(index)
            expected_exception: str | None = None
            try:
                expected = reference(*args, **kwargs)
            except ValueError:
                expected = None
                expected_exception = "ValueError"
            correctness_cases.append({
                "case_id": f"{task_id}-grader-correctness-{index}",
                "args": _json_value(list(args)),
                "kwargs": _json_value(kwargs),
                "expected": _json_value(expected),
                **({
                    "expected_exception": expected_exception,
                    "expected_exception_prefix": "ValueError",
                } if expected_exception else {}),
            })
        spec.update({
            "cases": cases,
            "correctness_calls": correctness_cases,
            "repeats": len(cases),
            "seed": int(module.WORKLOAD_SEED),
            "baseline_source": source,
            "performance_metric": {
                "name": "paired_ratio_student_t_lower",
                "t_critical": float(module.timing.T_CRIT_95_DF6),
                "success_threshold": float(module.timing.THRESHOLD),
                "reward_saturation": 4.0,
            },
        })
    provenance = {
        "task_id": task_id,
        "evaluator_path": str(evaluator),
        "evaluator_sha256": sha256_file(evaluator),
        "case_count": len(cases),
        "skipped_invalid_generator_indices": skipped,
        "task_kind": task_kind,
    }
    return spec, provenance
