from pathlib import Path
from typing import Any

from vare.research_lab.calibration import run_calibration
from vare.research_lab.evaluation import grade
from vare.research_lab.spec import load_catalog
from vare.research_lab.workspace import materialize

EQUIVALENT_ACCUM = '''
def _mse_grad(features, targets, params):
    count = len(targets)
    width = len(params)
    grad = [0.0] * width
    for row, target in zip(features, targets):
        pred = sum(value * params[index] for index, value in enumerate(row))
        resid = pred - target
        for index, value in enumerate(row):
            grad[index] += value * resid
    return [value / count for value in grad]


def accumulated_gradient(batches, params):
    if not batches:
        raise ValueError("need batches")
    total = [0.0] * len(params)
    for batch in batches:
        grad = _mse_grad(batch["x"], batch["y"], params)
        total = [left + right for left, right in zip(total, grad)]
    scale = 1.0 / len(batches)
    return [item * scale for item in total]
'''


def _expectation_mismatches(payload: dict[str, Any]) -> list[str]:
    """Name each control that disagreed. Does not change the acceptance rule."""
    lines: list[str] = []
    for row in payload.get("tasks") or []:
        task_id = row.get("task_id")
        cases = row.get("cases") or {}
        for name, case in cases.items():
            if case.get("matches_expectation") is True:
                continue
            metrics = case.get("metrics") or {}
            line = (
                f"{task_id} {name}: expected_pass={case.get('expected_pass')} "
                f"passed={case.get('passed')} classification={case.get('classification')}"
            )
            if task_id == "loop_invariant_score_v1" or "gain_low_s" in metrics or "noise_floor_s" in metrics:
                line += (
                    f" gain_low_s={metrics.get('gain_low_s')} "
                    f"noise_floor_s={metrics.get('noise_floor_s')}"
                )
            failures = case.get("failures") or []
            if failures:
                line += f" failures={failures}"
            lines.append(line)
    return lines


def test_calibration_distinguishes_baseline_known_good_and_incorrect(tmp_path: Path):
    payload = run_calibration(tmp_path / "work")
    mismatches = _expectation_mismatches(payload)
    cases = [
        case
        for row in payload["tasks"]
        for case in row["cases"].values()
    ]
    if any(case["classification"] == "isolation_unverified" for case in cases):
        # macOS and unsupported Linux hosts must not fall back to the legacy
        # same-user worker. Preserve the result as an isolation failure, not a
        # candidate correctness failure or a calibration pass.
        assert payload["expectations_met"] is False
        assert all(case["classification"] == "isolation_unverified" for case in cases)
        assert all(case["passed"] is False for case in cases)
        assert all(case["correctness_evaluated"] is False for case in cases)
        assert all(case["performance_evaluated"] is False for case in cases)
        return
    assert payload["expectations_met"] is True, (
        "calibration expectations failed:\n" + "\n".join(mismatches)
    )
    by_id = {row["task_id"]: row for row in payload["tasks"]}
    assert set(by_id) == {
        "grad_accum_scale_v1",
        "loop_invariant_score_v1",
        "descent_sign_debug_v1",
    }
    score = by_id["loop_invariant_score_v1"]["cases"]
    assert score["baseline"]["correctness_passed"] is True
    assert score["baseline"]["performance_passed"] is False
    assert score["known_good"]["performance_passed"] is True
    good = score["known_good"]["metrics"]
    assert good["performance_gate"] == "paired_ci_exceeds_aa_noise"
    assert good["gain_low_s"] > good["noise_floor_s"]
    assert score["incorrect"]["correctness_passed"] is False


def test_expectation_mismatch_message_names_the_timing_bounds() -> None:
    payload = {
        "expectations_met": False,
        "tasks": [{
            "task_id": "loop_invariant_score_v1",
            "cases": {
                "baseline": {
                    "matches_expectation": True,
                    "expected_pass": False,
                    "passed": False,
                },
                "known_good": {
                    "matches_expectation": False,
                    "expected_pass": True,
                    "passed": False,
                    "classification": "fail_performance",
                    "metrics": {"gain_low_s": 0.001, "noise_floor_s": 0.002},
                    "failures": [
                        "paired time-saved lower bound does not exceed the A/A noise upper bound",
                    ],
                },
            },
        }],
    }
    text = "\n".join(_expectation_mismatches(payload))
    assert "loop_invariant_score_v1 known_good" in text
    assert "expected_pass=True" in text
    assert "passed=False" in text
    assert "gain_low_s=0.001" in text
    assert "noise_floor_s=0.002" in text
    assert "baseline" not in text


def test_behavior_equivalent_gradient_passes_without_source_match(tmp_path: Path):
    task = load_catalog()["grad_accum_scale_v1"]
    workspace = tmp_path / "workspace"
    materialize(task, workspace)
    known = (task.root / "controls" / "known_good" / "accum.py").read_text(encoding="utf-8")
    assert EQUIVALENT_ACCUM.strip() != known.strip()
    (workspace / "accum.py").write_text(EQUIVALENT_ACCUM, encoding="utf-8")
    result = grade(task, workspace)
    if result["classification"] == "isolation_unverified":
        assert result["passed"] is False
        assert result["correctness_evaluated"] is False
        assert result["correctness_passed"] is None
        return
    assert result["passed"] is True
    assert result["classification"] == "pass"
