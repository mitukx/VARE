"""CPU tests for the fixed-controller long-horizon evaluation harness."""

from __future__ import annotations

import json
import hashlib
import dataclasses
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

from scripts import long_horizon_eval as lh
from vare.learning_pipeline.policy import MLXPolicy
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
        source = files.get("calc/ops.py") or next(iter(files.values()), "")
        candidate_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if self.result is not None:
            result = dict(self.result)
            if result.get("candidate_sha256") == "a" * 64:
                result["candidate_sha256"] = candidate_sha256
            return result
        passed = "return a + b" in files.get("calc/ops.py", "")
        return trusted_result(passed, candidate_sha256=candidate_sha256)


def trusted_result(passed: bool, **overrides):
    result = {"schema": "vare_function_oracle_v1", "executed": True, "passed": passed,
              "reason": "ok" if passed else "fail_to_pass_failed",
              "candidate_sha256": "a" * 64, "case_set_sha256": "b" * 64,
              "isolation": {"mode": "docker_function_call", "candidate_isolated": True,
                            "expected_values_parent_owned": True},
              "call_report": {"task_kind": "correctness", "passed": passed,
                              "cases": [{"case_id": "case-1", "passed": passed}]}}
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
    assert episode["raw_grade"]["call_report"]["cases"][0]["case_id"] == "case-1"
    agg = result["aggregate"]
    assert agg["episodes"] == 2 and agg["completion_rate"] == 1.0 and agg["final_success_rate"] == 1.0
    assert agg["synthetic"] is True


def test_mlx_policy_keeps_exact_prompt_and_completion_token_ids(monkeypatch):
    events = [
        types.SimpleNamespace(text="A", token=10, generation_tokens=1, finish_reason=None),
        types.SimpleNamespace(text="B", token=11, generation_tokens=2, finish_reason="stop"),
    ]
    sampler_arguments = {}

    def stream_generate(_model, _tokenizer, prompt_ids, *, max_tokens, sampler):
        assert prompt_ids == [101, 102]
        assert max_tokens == 8
        assert sampler == "sampler"
        yield from events

    def make_sampler(**kwargs):
        sampler_arguments.update(kwargs)
        return "sampler"

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.__path__ = []
    mlx_lm.stream_generate = stream_generate
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    policy = MLXPolicy.__new__(MLXPolicy)
    policy._mx = types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda _seed: None))
    policy.model = object()
    policy.tokenizer = types.SimpleNamespace(
        apply_chat_template=lambda _messages, **_kwargs: [101, 102],
    )
    policy.max_prompt_tokens = 32
    generation = policy.generate(
        [{"role": "user", "content": "task"}], max_tokens=8,
        temperature=0.7, top_p=1.0, seed=4,
    )

    assert generation.text == "AB"
    assert generation.prompt_ids == [101, 102]
    assert generation.completion_ids == [10, 11]
    assert generation.prompt_tokens == 2 and generation.completion_tokens == 2
    assert generation.sampling_temperature == 0.7 and generation.sampling_top_p == 1.0
    assert sampler_arguments == {"temp": 0.7, "top_p": 1.0, "min_p": 0.0, "top_k": 0}


def test_verified_grpo_group_extracts_only_shared_initial_action(tmp_path):
    task = dataclasses.replace(
        lh.fixture_task(), synthetic=False, source_revision="d" * 40,
        workspace_patch_sha256="e" * 64,
    )
    script = list(lh.FIXTURE_SCRIPT)
    bad_script = list(script)
    bad_script[5] = lh._call(
        "replace_text", path="calc/ops.py", old_text="return a - b",
        new_text="return a * b", expected_source_sha256=lh.FIXTURE_OPS_SHA256,
    )

    class TracePolicy(lh.ScriptedPolicy):
        synthetic = False

        def generate(self, messages, *, max_tokens, temperature, top_p, seed, task=None):
            sample_seed = seed // 4096
            self.calls += 1
            index = sum(message["role"] == "assistant" for message in messages)
            selected = bad_script if sample_seed == 1 else script
            text = selected[index] if index < len(selected) else "I have nothing further."
            if len(text.split()) > max_tokens:
                text = " ".join(text.split()[:max_tokens])
            return lh.Generation(
                text, 2, 2, 0.0, "stop",
                prompt_ids=[101, 102], completion_ids=[200, 300 + sample_seed],
                sampling_temperature=temperature, sampling_top_p=top_p,
            )

    policy = TracePolicy(script, policy_id="trace-policy")
    arm = lh.PolicyArm("base", policy, PolicyCheckpoint("base", 0, policy.checkpoint_sha256()))
    controller = dataclasses.replace(lh.CONTROLLER_PRESETS["full"], name="grpo-train", temperature=0.7)
    grader = FakeGrader()
    root = tmp_path / "grpo-group"
    harness = lh.EpisodeHarness(
        root, arm, controller, factories={task.task_id: lh.write_fixture},
        public_tests={task.task_id: lh.fixture_public_tests},
        grader_for=lambda _task: grader,
    )
    ids = harness.enroll_grpo_group(
        [lh.EpisodeSpec(task, seed) for seed in (0, 1)], group_id="train:fixture:group-0",
    )
    harness.run()

    metrics = lh.run_metrics(root)
    assert [episode["seed"] for episode in metrics["episodes"]] == [0, 1]
    assert [episode["final_success"] for episode in metrics["episodes"]] == [True, False]
    store = TrajectoryStore(root / "trajectories.sqlite", audit_only=True)
    samples = lh.grpo_samples_from_verified_group(store, "train:fixture:group-0")
    assert len(samples) == 2
    assert [sample.reward for sample in samples] == [1.0, 0.0]
    assert [sample.sample_index for sample in samples] == [0, 1]
    assert samples[0].prompt_ids == samples[1].prompt_ids == [101, 102]
    assert samples[0].completion_ids != samples[1].completion_ids
    assert samples[0].behavior_checkpoint_id == samples[1].behavior_checkpoint_id
    assert all(sample.behavior_logprobs is None for sample in samples)
    assert set(ids) == {row["id"] for row in store.group("train:fixture:group-0")}


def test_grpo_group_rejects_truncated_sampler_before_enrollment(tmp_path):
    task = dataclasses.replace(lh.fixture_task(), synthetic=False)
    arm = stub_arm()
    controller = dataclasses.replace(
        lh.CONTROLLER_PRESETS["full"], name="truncated", temperature=0.7, top_p=0.9,
    )
    harness = lh.EpisodeHarness(
        tmp_path / "truncated", arm, controller, factories={task.task_id: lh.write_fixture},
    )
    with pytest.raises(ValueError, match="top_p == 1.0"):
        harness.enroll_grpo_group(
            [lh.EpisodeSpec(task, seed) for seed in (0, 1)], group_id="train:fixture:truncated",
        )
    assert harness.store.group("train:fixture:truncated") == []


def test_budget_stop_requires_public_and_trusted_grade_for_success(tmp_path):
    task = lh.fixture_task()
    source_sha256 = hashlib.sha256(lh.FIXTURE_SOURCE["calc/ops.py"].encode()).hexdigest()
    script = (
        lh._call("list_files"),
        lh._call("read_file", path="calc/ops.py"),
        lh._call("replace_text", path="calc/ops.py", old_text="return a - b",
                 new_text="return a * b", expected_source_sha256=source_sha256),
    )
    arm = stub_arm(script=script)
    controller = lh.ControllerConfig(name="public-gate-budget", max_turns=3)
    grader = FakeGrader(trusted_result(True))
    run = tmp_path / "public-gate-budget"
    h = lh.EpisodeHarness(
        run, arm, controller,
        factories={task.task_id: lh.write_fixture},
        public_tests={task.task_id: lh.fixture_public_tests},
        grader_for=lambda _task: grader,
    )
    h.enroll([lh.EpisodeSpec(task, seed=0)])
    h.run()

    episode = lh.run_metrics(run)["episodes"][0]
    assert episode["stop_reason"] == "turn_budget_exhausted"
    assert episode["trusted_grade"] is True  # the hidden grader passes the intentionally bad edit
    assert episode["public_tests"]["final_passed"] is False
    assert episode["final_success"] is False


def _final_success_fixture(**overrides):
    verification = {
        "trusted": True,
        "passed": True,
        "metadata": {
            "trajectory_sha256": "a" * 64,
            "verifier_config_sha256": "b" * 64,
            "task_sha256": "c" * 64,
            "source_revision": "d" * 40,
            "workspace_patch_sha256": "e" * 64,
            "raw_grade": {"candidate_sha256": "f" * 64},
            "graded_files": ["calc/ops.py"],
        },
    }
    values = {
        "store_state": "verified",
        "expected_steps": 8,
        "committed_steps": 8,
        "verification": verification,
        "expected_trajectory_sha256": "a" * 64,
        "expected_verifier_config_sha256": "b" * 64,
        "initial_source_hashes": {"calc/ops.py": "0" * 64},
        "final_source_hashes": {"calc/ops.py": "f" * 64},
        "public_tests_configured": True,
        "public_tests_passed": True,
        "expected_task_sha256": "c" * 64,
        "expected_source_revision": "d" * 40,
        "expected_workspace_patch_sha256": "e" * 64,
        "require_task_binding": True,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"public_tests_passed": False}, False),  # hidden pass, public fail
        ({"public_tests_passed": None}, False),  # public execution missing or unfinished
        ({"public_tests_configured": False, "public_tests_passed": None}, True),
        ({"verification": {**_final_success_fixture()["verification"], "passed": False}}, False),
        ({"verification": {**_final_success_fixture()["verification"], "trusted": False}}, False),
        ({"verification": None}, False),  # timeout or missing grade
        ({"store_state": "running"}, False),  # partial/uncommitted episode
        ({"committed_steps": 7}, False),
        ({"expected_verifier_config_sha256": "9" * 64}, False),
        ({"expected_source_revision": "9" * 40}, False),
        ({"initial_source_hashes": {"calc/ops.py": "f" * 64}}, False),
        ({"initial_source_hashes": {}}, False),
        ({"final_source_hashes": {"calc/ops.py": "9" * 64}}, False),
        ({"verification": {**_final_success_fixture()["verification"], "metadata": {
            **_final_success_fixture()["verification"]["metadata"], "graded_files": [],
        }}}, False),
    ],
)
def test_final_success_fails_closed_on_public_grade_provenance_and_completion(
    overrides, expected,
):
    assert lh.trusted_final_success(**_final_success_fixture(**overrides)) is expected


def test_recovered_complete_episode_can_succeed_after_provenance_replay():
    assert lh.trusted_final_success(**_final_success_fixture(store_state="verified")) is True


def test_protocol_digest_is_bound_to_episode_and_metrics(tmp_path):
    protocol_sha = "c" * 64
    result = run_once(tmp_path / "run", grader=FakeGrader(), protocol_sha256=protocol_sha)
    assert result["episodes"][0]["protocol_sha256"] == protocol_sha


def test_development_enrollment_requires_a_frozen_protocol(tmp_path):
    base = lh.fixture_task()
    task = lh.EpisodeTask(
        task_id="fixture-development", repository=base.repository, split="development",
        issue=base.issue, readable_files=base.readable_files,
        editable_files=base.editable_files, target_files=base.target_files, synthetic=True,
    )
    without_lock = lh.EpisodeHarness(
        tmp_path / "unlocked", stub_arm(), lh.CONTROLLER_PRESETS["full"],
        factories={task.task_id: lh.write_fixture},
    )
    with pytest.raises(ValueError, match="Development episodes require a validated frozen protocol"):
        without_lock.enroll([lh.EpisodeSpec(task, 1)])

    locked = lh.EpisodeHarness(
        tmp_path / "locked", stub_arm(), lh.CONTROLLER_PRESETS["full"],
        factories={task.task_id: lh.write_fixture}, protocol_sha256="d" * 64,
    )
    assert locked.enroll([lh.EpisodeSpec(task, 1)])
    metadata = json.loads(locked.store.trajectory("stub:full:fixture-development:s1")["task"])["task"]["metadata"]
    assert metadata["task_split"] == "development"


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
    (FakeGrader(result=trusted_result(True, isolation={"mode": "legacy_native"})), "isolation_not_function_call"),
    (FakeGrader(result=trusted_result(True, isolation={"mode": "docker_function_call",
                                                       "candidate_isolated": False,
                                                       "expected_values_parent_owned": True})),
     "candidate_isolation_unverified"),
    (FakeGrader(result=trusted_result(True, schema="learning_pipeline_repo_env_grade_v2")), "untrusted_grade_schema"),
    (FakeGrader(result=trusted_result(True, executed=False)), None),
    (FakeGrader(result={**trusted_result(True), "passed": "yes"}), None),
    (FakeGrader(result={**trusted_result(True), "call_report": {"task_kind": "correctness", "passed": False,
                                                                  "cases": [{"case_id": "x", "passed": True}]}}),
     "call_report_inconsistent"),
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
    for key in ("mode", "candidate_isolated", "expected_values_parent_owned"):
        raw = trusted_result(True)
        del raw["isolation"][key]
        assert lh.trusted_verdict(raw)["trusted"] is False
    assert lh.trusted_verdict(trusted_result(True, schema="learning_pipeline_repo_env_grade_v2"))["trusted"] is False
    assert lh.trusted_verdict(None)["trusted"] is False
    assert lh.trusted_verdict({"passed": True})["trusted"] is False
    assert lh.trusted_verdict({**trusted_result(True), "call_report": {"cases": [None]}})["reason"] == "case_results_invalid"


def test_function_oracle_grader_uses_parent_owned_calls_and_scoped_source(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import vare.isolated_candidate as isolated_candidate
    from vare.isolated_candidate import FunctionCall

    task = lh.fixture_task()
    call = FunctionCall("sum", "calc.ops", "add", [2, 3], 5)
    grader = lh.FunctionOracleGrader(task_id=task.task_id, target_file="calc/ops.py", calls=[call],
                                    image="python@sha256:" + "a" * 64,
                                    support_files={"calc/__init__.py": "", "calc/util.py": "VALUE = 5\n"},
                                    additional_source_hashes={"scripts/test_grader.py": "b" * 64})
    observed = {}

    def grade_calls(subject, calls, **kwargs):
        observed["source"] = (Path(subject) / "calc/ops.py").read_text(encoding="utf-8")
        observed["support"] = (Path(subject) / "calc/util.py").read_text(encoding="utf-8")
        observed["calls"] = list(calls)
        observed["kwargs"] = kwargs
        return SimpleNamespace(to_dict=lambda: {"task_kind": "correctness", "passed": True,
                                                "cases": [{"case_id": "sum", "passed": True}]},
                               passed=True)

    monkeypatch.setattr(isolated_candidate, "grade_calls", grade_calls)
    raw = grader.grade(task=task, files={"calc/ops.py": "def add(a, b): return a + b\n"})
    assert observed["calls"] == [call]
    assert observed["source"] == "def add(a, b): return a + b\n"
    assert observed["support"] == "VALUE = 5\n"
    assert observed["kwargs"]["task_kind"] == "correctness"
    assert set(raw["source_identity"]) == {
        "grader_sha256", "candidate_harness_sha256", "isolation_runner_sha256",
        "support_files_sha256", "wrapper_sha256", "additional_sources_sha256",
    }
    assert all(len(value) == 64 for value in raw["source_identity"].values()
               if isinstance(value, str))
    assert len(raw["source_identity"]["support_files_sha256"]["calc/util.py"]) == 64
    assert raw["source_identity"]["additional_sources_sha256"] == {"scripts/test_grader.py": "b" * 64}
    assert raw["case_set_sha256"] == grader.config_sha256()
    assert lh.trusted_verdict(raw) == {"trusted": True, "passed": True, "reason": "all_cases_passed"}
    with pytest.raises(ValueError, match="relative Python source path"):
        lh.FunctionOracleGrader(task_id="t", target_file="../escape.py", calls=[call], image="python@sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="safe relative Python"):
        lh.FunctionOracleGrader(task_id="t", target_file="calc/ops.py", calls=[call],
                                image="python@sha256:" + "a" * 64,
                                support_files={"../escape.py": "x = 1\n"})


def test_function_oracle_configs_have_distinct_verifier_identities(tmp_path):
    from vare.isolated_candidate import FunctionCall

    image = "python@sha256:" + "a" * 64
    first = lh.FunctionOracleGrader(
        task_id="task-one", target_file="calc/ops.py",
        calls=[FunctionCall("case-one", "calc.ops", "add", [1, 2], 3)], image=image)
    second = lh.FunctionOracleGrader(
        task_id="task-two", target_file="calc/ops.py",
        calls=[FunctionCall("case-two", "calc.ops", "add", [1, 2], 3)], image=image)
    assert first.config_sha256() != second.config_sha256()
    assert first.name != second.name

    store = TrajectoryStore.create(tmp_path / "verifiers.sqlite", run_id="verifier-identities")
    store.register_verifier(first.name, first.version, first.config_sha256())
    store.register_verifier(second.name, second.version, second.config_sha256())


def _fixture_protocol(task, grader, controller, *, source_root, factory="tests.fake:grader_factory"):
    return {
        "preregistered_before_model_inference": True,
        "protocol_id": "fixture-protocol-v1",
        "model": {"id": "fixture-model", "revision": "a" * 40},
        "controller": {"name": controller.name, "config": controller.to_dict(),
                       "sha256": controller.sha256()},
        "budgets": {"lease_seconds": 0.5, "max_attempts": 5, "max_prompt_tokens": 8192,
                    "task_limit": 1, "seed": 0,
                    "max_calls_per_turn": controller.max_calls_per_turn,
                    "max_new_tokens_per_turn": controller.max_new_tokens_per_turn,
                    "max_total_new_tokens": controller.max_total_new_tokens,
                    "max_turns": controller.max_turns,
                    "temperature": controller.temperature, "top_p": controller.top_p,
                    "wall_seconds": controller.wall_seconds},
        "resources": {"min_available_memory_gib": 8.0, "min_free_disk_gib": 4.0},
        "execution": {"base_uninterrupted": {"arm": "base", "seed": 0, "adapter": None},
                      "effective_episode_seed": 0, "seeds_argument": 1, "task_count": 1},
        "task": {"task_id": task.task_id, "repository": task.repository, "split": task.split,
                 "source_revision": task.source_revision, "episode_task_sha256": task.sha256(),
                 "issue_sha256": hashlib.sha256(task.issue.encode()).hexdigest(),
                 "readable_file_count": len(task.readable_files), "target_file": task.target_files[0],
                 "train_manifest_sha256": "b" * 64},
        "grader": {"factory": factory, "class": type(grader).__name__,
                   "config_sha256": grader.config_sha256()},
        "source_hashes": {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in (
                "scripts/long_horizon_eval.py", "scripts/funcy_tree_oracle.py",
                "scripts/run_local_agent.py", "src/vare/trajectory_store.py",
                "src/vare/isolated_candidate.py", "src/vare/container_isolation.py",
            )
        },
    }


def test_frozen_protocol_binds_model_controller_task_grader_and_source():
    task = lh.fixture_task()
    grader = FakeGrader()
    controller = lh.CONTROLLER_PRESETS["full"]
    protocol = _fixture_protocol(task, grader, controller, source_root=ROOT)
    kwargs = dict(
        model_id="fixture-model", revision="a" * 40, controller=controller,
        tasks=[task], graders={task.task_id: grader}, arm="base", policy_version=0,
        adapter=None, seeds=[0], crash_after_commit_index=None, lease_seconds=0.5,
        max_attempts=5, manifest_sha256="b" * 64, max_prompt_tokens=8192, task_limit=1,
        grader_factory="tests.fake:grader_factory",
        source_root=ROOT, min_available_memory_gib=8.0, min_free_disk_gib=4.0,
    )
    lh.validate_frozen_protocol(protocol, **kwargs)

    with pytest.raises(ValueError, match="controller"):
        lh.validate_frozen_protocol(protocol, **dict(kwargs, controller=lh.CONTROLLER_PRESETS["minimal"]))
    with pytest.raises(ValueError, match="model.revision"):
        lh.validate_frozen_protocol(protocol, **dict(kwargs, revision="c" * 40))
    with pytest.raises(ValueError, match="scenario"):
        lh.validate_frozen_protocol(protocol, **dict(kwargs, seeds=[1]))
    wrong_source = dict(protocol, source_hashes={
        **protocol["source_hashes"], "scripts/long_horizon_eval.py": "0" * 64})
    with pytest.raises(ValueError, match="source_hashes"):
        lh.validate_frozen_protocol(wrong_source, **kwargs)


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


def test_resume_waits_for_expired_verification_lease(tmp_path):
    run = tmp_path / "verify-recovery"
    initial = harness(run, lease_seconds=0.35, worker_id="worker-before-crash")
    initial.run()  # Finish execution but leave verification queued.
    abandoned = initial.store.claim("verify", "worker-crashed", lease_seconds=0.35)
    assert abandoned is not None

    resumed = harness(run, grader=FakeGrader(), lease_seconds=0.35, worker_id="worker-after-crash")
    started = time.monotonic()
    resumed.run(wait_for_expired=True, poll_seconds=0.01)
    elapsed = time.monotonic() - started

    assert elapsed >= 0.2
    assert resumed.store.counts().get("verified") == 2
    result = lh.run_metrics(run)
    assert result["audit"]["pass"] is True
    assert all(episode["raw_grade"] is not None for episode in result["episodes"])


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
    record["_patch"] = patch
    task = lh.learning_pipeline_task(record, lh.readable_python_files(clone, commit), source_revision=commit)
    assert task.editable_files == ("pkg/m.py",) and not any(lh.is_test_path(p) for p in task.readable_files)
    assert task.source_revision == commit
    assert task.workspace_patch_sha256 == hashlib.sha256(patch.encode()).hexdigest()


def test_learning_pipeline_task_rejects_invalid_source_provenance():
    with pytest.raises(ValueError, match="full commit SHA"):
        lh.EpisodeTask(task_id="t", repository="r", split="train", issue="i", readable_files=("a.py",),
                       editable_files=("a.py",), target_files=("a.py",), source_revision="short")
    with pytest.raises(ValueError, match="SHA-256 digest"):
        lh.EpisodeTask(task_id="t", repository="r", split="train", issue="i", readable_files=("a.py",),
                       editable_files=("a.py",), target_files=("a.py",), workspace_patch_sha256="short")


def test_funcy_oracle_factory_is_scoped_and_pins_support_sources(tmp_path):
    from types import SimpleNamespace

    from scripts.funcy_tree_oracle import TASK_ID, grader_factory

    clone = tmp_path / "funcy"
    (clone / "funcy").mkdir(parents=True)
    (clone / "funcy" / "types.py").write_text("is_seqcont = lambda x: isinstance(x, list)\n", encoding="utf-8")
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "init", "-q"], cwd=clone, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True, env=env)
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "base"], cwd=clone, check=True, env=env)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, check=True, env=env,
                            capture_output=True, text=True).stdout.strip()
    fake_env = SimpleNamespace(path=clone, commit=commit, upstream="Suor/funcy")
    record = {"instance_id": TASK_ID, "upstream_repo": "Suor/funcy", "split": "train",
              "target_file": "funcy/tree.py"}
    grader = grader_factory(record, fake_env)
    assert grader.config_sha256()
    assert grader.support_files["funcy/types.py"] == "is_seqcont = lambda x: isinstance(x, list)\n"
    assert len(grader.calls) == 4
    assert all(call.module == "vare_oracle_target" for call in grader.calls)
    assert grader.wrapper_source == "from funcy.tree import ltree_leaves\n"
    assert grader.source_identity["wrapper_sha256"]
    assert grader.source_identity["additional_sources_sha256"]["scripts/funcy_tree_oracle.py"]
    with pytest.raises(ValueError, match="scoped to its one frozen Train task"):
        grader_factory({**record, "instance_id": "other"}, fake_env)


def test_smoke_cli(tmp_path, capsys):
    assert lh.main(["smoke", "--output", str(tmp_path), "--controller", "full", "--seeds", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["audit"]["pass"] is True and out["aggregate"]["completion_rate"] == 1.0
    assert out["aggregate"]["final_success_rate"] == 0.0  # no trusted grader in the smoke run
