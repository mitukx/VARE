import json
from pathlib import Path

from vare.research_lab.controller import create_run, open_run
from vare.research_lab.spec import load_catalog, suite_manifest
from vare.research_lab.store import RunStore


def _start(tmp_path: Path):
    task = load_catalog()["descent_sign_debug_v1"]
    return task, create_run(
        tmp_path / "run",
        task,
        run_id="resume-me",
        proposer_name="diagnostic",
        proposer_kind="diagnostic_policy",
        suite_sha256=suite_manifest()["manifest_sha256"],
    )


def test_resume_does_not_duplicate_committed_experiments(tmp_path: Path):
    task, controller = _start(tmp_path)
    controller.run(max_actions=1)
    first = controller.store.read_journal()
    assert [row["hypothesis_id"] for row in first] == ["baseline"]
    baseline_id = first[0]["experiment_id"]
    again = open_run(tmp_path / "run", task)
    again.run(max_actions=1)
    ids = [row["experiment_id"] for row in again.store.read_journal()]
    assert ids.count(baseline_id) == 1
    assert len(ids) == 2


def test_completed_resume_is_a_no_op(tmp_path: Path):
    task, controller = _start(tmp_path)
    controller.run()
    count = len(controller.store.read_journal())
    open_run(tmp_path / "run", task).run()
    assert len(RunStore(tmp_path / "run").read_journal()) == count


def test_duplicate_experiment_id_does_not_append(tmp_path: Path):
    _task, controller = _start(tmp_path)
    controller.run(max_actions=1)
    existing = controller.store.read_journal()[0]
    returned = controller.store.commit(dict(existing))
    assert returned["experiment_id"] == existing["experiment_id"]
    assert len(controller.store.read_journal()) == 1


def test_truncated_journal_tail_is_ignored(tmp_path: Path):
    _task, controller = _start(tmp_path)
    controller.run(max_actions=1)
    path = tmp_path / "run" / "journal.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{\"seq\":", encoding="utf-8")
    records = RunStore(tmp_path / "run").read_journal()
    assert len(records) == 1
    assert records[0]["hypothesis_id"] == "baseline"


def test_cancel_file_stops_before_a_new_experiment(tmp_path: Path):
    _task, controller = _start(tmp_path)
    (tmp_path / "run" / "CANCEL").write_text("stop\n", encoding="utf-8")
    summary = controller.run()
    assert summary["status"] == "cancelled"
    rows = controller.store.read_journal()
    assert [row["action_type"] for row in rows] == ["cancel"]
    assert all(row.get("verification") is None for row in rows)


def test_state_cache_can_be_rebuilt_from_the_journal(tmp_path: Path):
    _task, controller = _start(tmp_path)
    controller.run(max_actions=1)
    state_path = tmp_path / "run" / "state.json"
    state_path.unlink()
    state = RunStore(tmp_path / "run").write_state_cache()
    assert state["seq"] == 1
    assert json.loads(state_path.read_text(encoding="utf-8"))["seq"] == 1
