from pathlib import Path

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


def test_calibration_distinguishes_baseline_known_good_and_incorrect(tmp_path: Path):
    payload = run_calibration(tmp_path / "work")
    assert payload["expectations_met"] is True
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
    assert score["known_good"]["metrics"]["speedup"] >= 2.0
    assert score["incorrect"]["correctness_passed"] is False


def test_behavior_equivalent_gradient_passes_without_source_match(tmp_path: Path):
    task = load_catalog()["grad_accum_scale_v1"]
    workspace = tmp_path / "workspace"
    materialize(task, workspace)
    known = (task.root / "controls" / "known_good" / "accum.py").read_text(encoding="utf-8")
    assert EQUIVALENT_ACCUM.strip() != known.strip()
    (workspace / "accum.py").write_text(EQUIVALENT_ACCUM, encoding="utf-8")
    result = grade(task, workspace)
    assert result["passed"] is True
    assert result["classification"] == "pass"
