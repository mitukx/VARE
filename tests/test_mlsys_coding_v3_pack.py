import json
from pathlib import Path

import pytest

from scripts.mlsys_task_isolated_calls import materialize_isolated_calls
from scripts.run_separated_coding_eval import select_train_tasks


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks" / "mlsys_coding_v3"


def protocol():
    split = json.loads((PACK / "split.json").read_text())
    return {
        "pack": {
            "train": split["train"],
            "heldout": split["heldout"],
            "evaluation_split": "heldout",
        }
    }


def test_heldout_selector_exposes_only_frozen_ids(tmp_path):
    selected = select_train_tasks(PACK, protocol(), tmp_path / "task-roots")

    assert [task["task_id"] for task in selected] == [
        "mlsys-v3-ema-bias",
        "mlsys-v3-label-smooth",
    ]
    assert all(task["split"] == "heldout" for task in selected)
    for task in selected:
        assert not (task["task_root"] / "evaluator").exists()
        assert not (task["task_root"] / "subject").exists()
        assert (task["task_root"] / "task.json").is_file()


@pytest.mark.parametrize(
    ("task_id", "function", "case_count"),
    [
        ("mlsys-v3-ema-bias", "bias_corrected_ema", 5),
        ("mlsys-v3-label-smooth", "smooth_targets", 7),
    ],
)
def test_v3_hidden_cases_materialize_without_running_candidate(task_id, function, case_count):
    task_dir = PACK / "tasks" / task_id

    calls, provenance = materialize_isolated_calls(task_id, task_dir)

    assert calls["task_id"] == task_id
    assert calls["function"] == function
    assert len(calls["calls"]) == case_count
    assert provenance["case_count"] == case_count
    assert "expected" in calls["calls"][0]
