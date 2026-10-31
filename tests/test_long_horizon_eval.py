"""CPU tests for the fixed-controller long-horizon evaluation harness."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts import long_horizon_eval as lh
from vare.trajectory_store import PolicyCheckpoint, TrajectoryStore

ROOT = Path(__file__).resolve().parents[1]


def stub_arm(name: str = "stub", script=lh.FIXTURE_SCRIPT) -> lh.PolicyArm:
    policy = lh.ScriptedPolicy(script)
    return lh.PolicyArm(name, policy, PolicyCheckpoint(name, 0, policy.checkpoint_sha256()))


class FakeGrader:
    """Stand-in for an isolated trusted grader, returning a chosen raw result."""

    name = "fake_isolated_grader"
    version = 1

    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def config_sha256(self) -> str:
        return "a" * 64

    def grade(self, *, task, files):
        self.calls.append(dict(files))
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        passed = "return a + b" in files.get("calc/ops.py", "")
        return trusted_result(passed)


def trusted_result(passed: bool, **overrides):
    result = {"schema": "learning_pipeline_repo_env_grade_v2", "executed": True, "passed": passed,
              "reason": "ok" if passed else "fail_to_pass_failed",
              "isolation": {"mode": "required", "attested": True, "verdict_channel": "hmac_frames_v1"}}
    result.update(overrides)
    return result


def harness(root: Path, *, controller: str = "full", grader=None, arm=None, **kwargs) -> lh.EpisodeHarness:
    task = lh.fixture_task()
    h = lh.EpisodeHarness(root, arm or stub_arm(), lh.CONTROLLER_PRESETS[controller],
                          factories={task.task_id: lh.write_fixture},
                          public_tests={task.task_id: lh.fixture_public_tests},
                          grader_for=(lambda _t: grader) if grader is not None else None, **kwargs)
    h.enroll(lh.EpisodeSpec(task, seed) for seed in range(2))
    return h


def run_once(root: Path, **kwargs) -> dict:
    harness(root, **kwargs).run()
    return lh.run_metrics(root)


def canonical(root: Path) -> list:
    store = TrajectoryStore(root / "trajectories.sqlite", audit_only=True)
    with store.connection() as db:
        ids = [r[0] for r in db.execute("SELECT id FROM trajectories ORDER BY id")]
    return [lh.canonical_trajectory(store, i) for i in ids]


# ---------------------------------------------------------------- metrics


def test_full_controller_metrics_on_fixture(tmp_path):
    result = run_once(tmp_path / "run", grader=FakeGrader())
    assert result["audit"]["pass"] is True
    episode = result["episodes"][0]
    assert episode["stop_reason"] == "agent_finished"
    assert episode["turns"] == 8 and episode["padding_steps"] == 8
    assert episode["tool_invocations"] == 8
    assert episode["valid_tool_calls"] == 7  # the malformed JSON call is invalid
    assert episode["navigation"] == {"target_read": True, "target_read_before_first_edit": True,
                                     "edits_only_targets": True, "correct": True,
                                     "calls_before_target_read": 2, "nontarget_reads": 0}
    assert episode["edits"]["attempted"] == 2 and episode["edits"]["applied"] == 1
    assert episode["edits"]["final_source_changed"] is True
    assert episode["recovery"]["failed_actions"] >= 2 and episode["recovery"]["recovered"] >= 1
    assert episode["public_tests"]["final_passed"] is True
    assert episode["public_tests"]["finished_while_failing"] is False
    assert episode["controller_assists"]["normalized_tool_calls"] == 1
    assert episode["controller_assists"]["automatic_public_tests"] == 1
    assert episode["compute"]["generated_tokens"] > 0
    assert episode["trusted_grade"] is True and episode["final_success"] is True
    agg = result["aggregate"]
    assert agg["episodes"] == 2 and agg["completion_rate"] == 1.0 and agg["final_success_rate"] == 1.0
    assert agg["synthetic"] is True


def test_controller_ablation_changes_outcome_with_same_policy(tmp_path):
    grader = FakeGrader()
    full = run_once(tmp_path / "full", grader=grader)
    minimal = run_once(tmp_path / "minimal", controller="minimal", grader=grader)
    m = minimal["episodes"][0]
    assert m["stop_reason"] == "no_tool_call_retries_exhausted"
    assert m["controller_assists"] == {"normalized_tool_calls": 0, "reminders": 0, "recovered_bare_edits": 0,
                                       "automatic_public_tests": 0, "finish_refusals": 0}
    assert m["final_success"] is False and m["grade_reason"] == "no_source_change_not_graded"
    comparison = lh.paired_comparison(full["episodes"], minimal["episodes"])
    assert comparison["kind"] == "controller_ablation"
    assert comparison["pairs"] == 2 and comparison["left_rate"] == 1.0 and comparison["right_rate"] == 0.0


def test_policy_comparison_requires_same_controller(tmp_path):
    a = run_once(tmp_path / "a", grader=FakeGrader())
    b = run_once(tmp_path / "b", controller="minimal", grader=FakeGrader(), arm=stub_arm("other"))
    with pytest.raises(ValueError):
        lh.paired_comparison(a["episodes"], b["episodes"])


def test_budget_stop_is_recorded_terminal(tmp_path):
    controller = lh.ControllerConfig(name="tiny", max_total_new_tokens=8)
    task = lh.fixture_task()
    h = lh.EpisodeHarness(tmp_path / "r", stub_arm(), controller, factories={task.task_id: lh.write_fixture})
    h.enroll([lh.EpisodeSpec(task, 0)])
    h.run()
    episode = lh.run_metrics(tmp_path / "r")["episodes"][0]
    assert episode["stop_reason"] == "token_budget_exhausted"
    assert episode["compute"]["generated_tokens"] <= 8


# ---------------------------------------------------------------- fail-closed grading


@pytest.mark.parametrize("grader,reason", [
    (None, "ungraded"),
    (FakeGrader(result=trusted_result(True, isolation={"mode": "legacy_native"})), "isolation_not_required"),
    (FakeGrader(result=trusted_result(True, isolation={"mode": "required", "attested": False,
                                                       "verdict_channel": "hmac_frames_v1"})), None),
    (FakeGrader(result=trusted_result(True, schema="learning_pipeline_repo_env_grade_v1")), None),
    (FakeGrader(result=trusted_result(True, executed=False)), None),
    (FakeGrader(result={**trusted_result(True), "passed": "yes"}), None),
    (FakeGrader(error=RuntimeError("sandbox down")), "grader_error:RuntimeError"),
])
def test_untrusted_or_missing_grades_are_never_success(tmp_path, grader, reason):
    result = run_once(tmp_path / "run", grader=grader)
    for episode in result["episodes"]:
        assert episode["final_success"] is False
        assert episode["trusted_grade"] is False
        if reason is not None:
            assert episode["grade_reason"].startswith(reason.split(":")[0])
    assert result["aggregate"]["final_success_rate"] == 0.0


def test_grader_unavailable_for_task_is_recorded_untrusted(tmp_path):
    task = lh.fixture_task()
    h = lh.EpisodeHarness(tmp_path / "r", stub_arm(), lh.CONTROLLER_PRESETS["full"],
                          factories={task.task_id: lh.write_fixture}, grader_for=lambda _t: None)
    h.enroll([lh.EpisodeSpec(task, 0)])
    h.run()
    episode = lh.run_metrics(tmp_path / "r")["episodes"][0]
    assert episode["grade_recorded"] is True and episode["trusted_grade"] is False
    assert episode["grade_reason"] == "grader_unavailable" and episode["final_success"] is False


def test_trusted_verdict_requires_every_attestation_field():
    assert lh.trusted_verdict(trusted_result(True)) == {"trusted": True, "passed": True, "reason": "ok"}
    for key in ("mode", "attested", "verdict_channel"):
        raw = trusted_result(True)
        del raw["isolation"][key]
        assert lh.trusted_verdict(raw)["trusted"] is False
    assert lh.trusted_verdict(None)["trusted"] is False
    assert lh.trusted_verdict({"passed": True})["trusted"] is False


def test_grader_receives_only_changed_editable_files(tmp_path):
    grader = FakeGrader()
    run_once(tmp_path / "run", grader=grader)
    assert grader.calls and all(set(c) == {"calc/ops.py"} for c in grader.calls)


# ---------------------------------------------------------------- determinism


def test_stub_policy_trajectories_identical_across_runs(tmp_path):
    run_once(tmp_path / "one", grader=FakeGrader())
    run_once(tmp_path / "two", grader=FakeGrader())
    assert canonical(tmp_path / "one") == canonical(tmp_path / "two")
    first = lh.run_metrics(tmp_path / "one")["episodes"]
    second = lh.run_metrics(tmp_path / "two")["episodes"]
    assert [e["trajectory_head"] for e in first] != [] and [e["episode_id"] for e in first] == [
        e["episode_id"] for e in second]


# ---------------------------------------------------------------- resume after kill

CHILD = textwrap.dedent("""
    import os, signal, sys
    from pathlib import Path
    sys.path[:0] = [{root!r}, {src!r}]
    from tests.test_long_horizon_eval import harness, FakeGrader

    def crash(stage, index):
        if stage == {stage!r} and index == {index}:
            os.kill(os.getpid(), signal.SIGKILL)

    harness(Path({run!r}), grader=FakeGrader(), lease_seconds=0.5, crash_hook=crash,
            worker_id="worker-a").run()
""")


@pytest.mark.parametrize("stage,index", [("after_tools", 3), ("after_commit", 5), ("after_tools", 0)])
def test_resume_after_kill_has_no_duplicated_or_lost_steps(tmp_path, stage, index):
    run = tmp_path / "killed"
    code = CHILD.format(root=str(ROOT), src=str(ROOT / "src"), stage=stage, index=index, run=str(run))
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == -signal.SIGKILL, proc.stderr
    store = TrajectoryStore(run / "trajectories.sqlite", audit_only=True)
    first = sorted(store.counts().items())
    assert dict(first).get("running", 0) >= 1
    persisted = sum(1 for _ in store.iter_steps(lh.episode_id("stub", lh.CONTROLLER_PRESETS["full"], lh.fixture_task().task_id, 0)))
    assert persisted == (index + 1 if stage == "after_commit" else index)

    harness(run, grader=FakeGrader(), lease_seconds=0.5, worker_id="worker-b").run()
    run_once(tmp_path / "clean", grader=FakeGrader())
    assert canonical(run) == canonical(tmp_path / "clean")
    result = lh.run_metrics(run)
    assert result["audit"]["pass"] is True and result["audit"]["effective_duplicates"] == 0
    killed = [e for e in result["episodes"] if e["seed"] == 0][0]
    assert killed["resumed"] is True and killed["execute_attempts"] == 2
    assert killed["final_success"] is True
    assert [s["index"] for s in TrajectoryStore(run / "trajectories.sqlite", audit_only=True).iter_steps(
        killed["episode_id"])] == list(range(16))


def test_resume_refuses_different_controller(tmp_path):
    run = tmp_path / "r"
    task = lh.fixture_task()
    h = harness(run, grader=FakeGrader())
    ident = h.enroll([lh.EpisodeSpec(task, 0)])[0]
    lease = h.store.claim("execute", "w", lease_seconds=5)
    assert lease is not None and lease.trajectory_id == ident
    other = lh.EpisodeHarness(run, stub_arm(), lh.CONTROLLER_PRESETS["minimal"],
                              factories={task.task_id: lh.write_fixture})
    other.tasks[ident] = task
    with pytest.raises(lh.ResumeMismatchError):
        other._execute(lease)


def test_replay_detects_workspace_drift(tmp_path):
    run = tmp_path / "r"
    run_once(run, grader=FakeGrader())
    task = lh.fixture_task()

    def drifted(target: Path) -> None:
        lh.write_fixture(target)
        (target / "calc" / "ops.py").write_text("def add(a, b):\n    return 0\n", encoding="utf-8")

    h = lh.EpisodeHarness(run, stub_arm(), lh.CONTROLLER_PRESETS["full"], factories={task.task_id: drifted},
                          public_tests={task.task_id: lh.fixture_public_tests})
    ident = h.enroll([lh.EpisodeSpec(task, 0)])[0]
    with pytest.raises(lh.ResumeMismatchError):
        h.rebuild(ident)


# ---------------------------------------------------------------- tamper detection


def test_reenrollment_with_different_metadata_is_refused(tmp_path):
    run = tmp_path / "r"
    harness(run)
    task = lh.fixture_task()
    h = lh.EpisodeHarness(run, stub_arm(), lh.CONTROLLER_PRESETS["full"], factories={task.task_id: lh.write_fixture})
    with pytest.raises(ValueError):
        h.enroll([lh.EpisodeSpec(task, 0)])  # public-test availability differs from the enrolled episode


def test_tampered_step_is_detected_and_metrics_refused(tmp_path):
    run = tmp_path / "r"
    run_once(run, grader=FakeGrader())
    db = sqlite3.connect(run / "trajectories.sqlite")
    seq, event = db.execute("SELECT sequence, event FROM events WHERE event LIKE '%read_file%' LIMIT 1").fetchone()
    db.execute("UPDATE events SET event=? WHERE sequence=?", (event.replace("calc/ops.py", "calc/util.py", 1), seq))
    db.commit()
    db.close()
    with pytest.raises(ValueError):
        lh.run_metrics(run)


# ---------------------------------------------------------------- task boundaries


def test_heldout_and_test_paths_are_refused(tmp_path):
    with pytest.raises(ValueError):
        lh.EpisodeTask(task_id="t", repository="r", split="heldout", issue="i", readable_files=("a.py",),
                       editable_files=("a.py",), target_files=("a.py",))
    with pytest.raises(ValueError):
        lh.EpisodeTask(task_id="t", repository="r", split="train", issue="i",
                       readable_files=("a.py", "tests/test_a.py"), editable_files=("a.py",), target_files=("a.py",))
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"instance_id": "x", "split": "heldout"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        lh.load_train_manifest(manifest)


def test_repo_workspace_factory_applies_bug_on_pinned_commit(tmp_path):
    clone = tmp_path / "clone"
    (clone / "pkg").mkdir(parents=True)
    (clone / "tests").mkdir()
    (clone / "pkg" / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (clone / "tests" / "test_m.py").write_text("def test_f():\n    pass\n", encoding="utf-8")
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q"], cwd=clone, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True, env=env)
    subprocess.run([*git, "commit", "-qm", "base"], cwd=clone, check=True, env=env)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, check=True, env=env, capture_output=True,
                            text=True).stdout.strip()
    patch = ("diff --git a/pkg/m.py b/pkg/m.py\n--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1,2 +1,2 @@\n"
             " def f():\n-    return 1\n+    return 2\n")
    assert lh.readable_python_files(clone, commit) == ("pkg/m.py",)
    target = tmp_path / "ws"
    target.mkdir()
    lh.repo_workspace_factory(clone, commit, patch)(target)
    assert (target / "pkg" / "m.py").read_text(encoding="utf-8").endswith("return 2\n")
    assert (target / ".git").is_dir()
    record = {"instance_id": "x__1", "upstream_repo": "o/x", "split": "train", "_problem": "f is wrong",
              "target_file": "pkg/m.py"}
    task = lh.learning_pipeline_task(record, lh.readable_python_files(clone, commit))
    assert task.editable_files == ("pkg/m.py",) and not any(lh.is_test_path(p) for p in task.readable_files)


def test_smoke_cli(tmp_path, capsys):
    assert lh.main(["smoke", "--output", str(tmp_path), "--controller", "full", "--seeds", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["audit"]["pass"] is True and out["aggregate"]["completion_rate"] == 1.0
    assert out["aggregate"]["final_success_rate"] == 0.0  # no trusted grader in the smoke run
