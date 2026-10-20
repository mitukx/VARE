import json
import hashlib
from pathlib import Path

import pytest

from scripts.mlsys_task_isolated_calls import materialize_isolated_calls
from scripts.run_separated_coding_eval import rebuild_manifest_text, select_train_tasks


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


def test_pack_manifest_rebuild_ignores_git_metadata(tmp_path):
    pack = tmp_path / "pack"
    (pack / ".git").mkdir(parents=True)
    (pack / ".git" / "config").write_text("[core]\n")
    (pack / "README.md").write_text("frozen task pack\n")
    (pack / "manifest.json").write_text('{"pack_id":"fixture-pack"}\n')

    expected = {
        "files": [{
            "bytes": len(b"frozen task pack\n"),
            "path": "README.md",
            "sha256": hashlib.sha256(b"frozen task pack\n").hexdigest(),
        }],
        "pack_id": "fixture-pack",
    }

    assert rebuild_manifest_text(pack) == json.dumps(expected, indent=2, sort_keys=True) + "\n"


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
    if task_id == "mlsys-v3-ema-bias":
        assert calls["timeout_s"] == 20
    assert provenance["case_count"] == case_count
    assert "expected" in calls["calls"][0]


def test_v3_warmup_copy_materializes_with_its_shared_v1_adapter():
    task_dir = PACK / "tasks" / "warmup_cosine"

    calls, provenance = materialize_isolated_calls("mlsys-v1-warmup-cosine", task_dir)

    assert calls["task_id"] == "mlsys-v1-warmup-cosine"
    assert calls["function"] == "learning_rate"
    assert len(calls["calls"]) == 12
    assert calls["timeout_s"] == 10
    assert provenance["case_count"] == 12


def test_label_smoothing_materializes_but_is_not_routed_without_type_fidelity():
    from vare.separated_candidate_harness import SeparatedHarnessError, _task_adapter

    task_dir = PACK / "tasks" / "mlsys-v3-label-smooth"
    materialize_isolated_calls("mlsys-v3-label-smooth", task_dir)
    with pytest.raises(SeparatedHarnessError, match="no isolated call materializer"):
        _task_adapter("mlsys-v3-label-smooth", task_dir / "task.json")
