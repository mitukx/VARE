import json
from pathlib import Path

import pytest

from vare.research_lab.controller import create_run
from vare.research_lab.export_adapter import ExportError, export_run
from vare.research_lab.isolation import isolation_summary
from vare.research_lab.spec import load_catalog, suite_manifest
from vare.research_lab.store import _atomic_json


def test_export_does_not_authorize_training(tmp_path: Path):
    task = load_catalog()["grad_accum_scale_v1"]
    run_dir = tmp_path / "run"
    controller = create_run(
        run_dir,
        task,
        run_id="export-me",
        proposer_name="diagnostic",
        proposer_kind="diagnostic_policy",
        suite_sha256=suite_manifest()["manifest_sha256"],
    )
    controller.run()
    header = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    header["counts_as_autonomous_model_success"] = True
    header["training_authorized"] = True
    _atomic_json(run_dir / "run.json", header)
    payload = export_run(run_dir)
    assert payload["training_authorized"] is False
    assert payload["reward_recomputed"] is False
    assert payload["held_out_split_accessed"] is False
    assert payload["counts_as_autonomous_model_success"] is False
    assert "reward" not in payload
    if not isolation_summary()["attestation_ok"]:
        assert payload["actions"][-1]["verification_passed"] is None
        assert payload["actions"][-1]["verification_classification"] is None
    else:
        assert payload["actions"][-1]["verification_passed"] is True


def test_export_rejects_a_forged_reward_and_frozen_ids(tmp_path: Path):
    task = load_catalog()["grad_accum_scale_v1"]
    run_dir = tmp_path / "run"
    controller = create_run(
        run_dir,
        task,
        run_id="forged",
        proposer_name="external",
        proposer_kind="external_agent",
        suite_sha256=suite_manifest()["manifest_sha256"],
    )
    controller.run(max_actions=0)
    journal = run_dir / "journal.jsonl"
    journal.write_text(
        json.dumps({
            "seq": 1,
            "experiment_id": "forged:1",
            "hypothesis_id": "baseline",
            "metrics": {"reward": 1},
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ExportError):
        export_run(run_dir)

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    _atomic_json(frozen / "run.json", {
        "schema": "vare.research_lab.run.v1",
        "run_id": "held",
        "task_id": "heldout_secret",
    })
    (frozen / "journal.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ExportError):
        export_run(frozen)
