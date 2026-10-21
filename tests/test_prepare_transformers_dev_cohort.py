from __future__ import annotations

import pytest

from scripts import prepare_transformers_dev_cohort_v1 as prepare_cohort


def test_baseline_preparation_refuses_to_overwrite_frozen_task_files(tmp_path, monkeypatch) -> None:
    task_id = "transformers-49481-bloom-static-cache-v1"
    task_root = tmp_path / prepare_cohort.TASKS[task_id]["directory"]
    task_root.mkdir()
    (task_root / "task.json").write_text("frozen\n", encoding="utf-8")
    output = tmp_path / "candidate-baseline"
    monkeypatch.setattr(prepare_cohort, "PACK", tmp_path)

    with pytest.raises(FileExistsError, match="refusing to rewrite frozen task artifacts"):
        prepare_cohort.prepare(task_id, "baseline", tmp_path / "unused-repository", output)

    assert not output.exists()
    assert (task_root / "task.json").read_text(encoding="utf-8") == "frozen\n"
