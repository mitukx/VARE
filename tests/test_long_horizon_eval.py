"""CPU tests for the fixed-controller long-horizon evaluation harness."""

from __future__ import annotations

import json
import hashlib
import dataclasses
import math
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


class _FakeScalar(float):
    def item(self):
        return float(self)

    def __sub__(self, other):
        return _FakeScalar(float(self) - float(other))


class _FakeLogProbs(list):
    """Small scalar-math stand-in; no MLX kernel or model is exercised."""

    ndim = 1

    @property
    def shape(self):
        return (len(self),)

    def __mul__(self, scale):
        return _FakeLogProbs(float(value) * scale for value in self)

    def astype(self, _dtype):
        return self

    def __getitem__(self, index):
        return _FakeScalar(super().__getitem__(index))


def _fake_mx():
    def logsumexp(values):
        maximum = max(values)
        return _FakeScalar(maximum + math.log(math.fsum(math.exp(x - maximum) for x in values)))

    return types.SimpleNamespace(
        random=types.SimpleNamespace(seed=lambda _seed: None),
        float32="float32", logsumexp=logsumexp,
    )


def test_mlx_policy_keeps_exact_prompt_completion_and_sampling_logprobs(monkeypatch):
    probabilities = [(index + 1) / 78 for index in range(12)]
    logprobs = _FakeLogProbs(math.log(value) for value in probabilities)
    events = [
        types.SimpleNamespace(text="A", token=10, generation_tokens=1, finish_reason=None,
                              logprobs=logprobs),
        types.SimpleNamespace(text="B", token=11, generation_tokens=2, finish_reason="stop",
                              logprobs=logprobs),
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

    policy = lh.TraceableMLXPolicy.__new__(lh.TraceableMLXPolicy)
    policy._mx = _fake_mx()
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
    normalizer = math.fsum(value ** (1.0 / 0.7) for value in probabilities)
    expected = [math.log(probabilities[token] ** (1.0 / 0.7) / normalizer) for token in (10, 11)]
    assert generation.sampling_behavior_logprobs == pytest.approx(expected)
    assert sampler_arguments == {"temp": 0.7, "top_p": 1.0, "min_p": 0.0, "top_k": 0}


def test_sampling_logprob_uses_temperature_normalized_distribution():
    logprobs = _FakeLogProbs([math.log(0.2), math.log(0.8)])
    actual = lh._sampling_token_logprob(logprobs, 1, 0.5, _fake_mx())
    assert actual == pytest.approx(math.log(0.64 / 0.68))
    assert actual != pytest.approx(math.log(0.8))


def _fake_verified_grpo_group(tmp_path, *, mutate_turn=None, mutate_commit=None, mutate_generation=None,
                              mutate_verification=None,
                              synthetic_task=False, synthetic_policy=False,
                              all_successful=False, run=True, score_action=None, checkpoint=None,
                              script_override=None, controller_overrides=None):
    """Fresh fake policy/grader data exercising the real-input contract only.

    No historical task, model or result is relabeled by this fixture. The
    false synthetic flags are test inputs for the converter's eligibility
    checks, not statements that a real model or isolated grader ran.
    """
    task = dataclasses.replace(
        lh.fixture_task(), synthetic=synthetic_task, source_revision="d" * 40,
        workspace_patch_sha256="e" * 64,
    )
    script = list(lh.FIXTURE_SCRIPT if script_override is None else script_override)
    bad_script = list(script)
    bad_script[5] = lh._call(
        "replace_text", path="calc/ops.py", old_text="return a - b",
        new_text="return a * b", expected_source_sha256=lh.FIXTURE_OPS_SHA256,
    )

    class TracePolicy(lh.ScriptedPolicy):
        synthetic = synthetic_policy

        def __init__(self, script, policy_id):
            super().__init__(script, policy_id=policy_id)
            self.generated = {}

        def generate(self, messages, *, max_tokens, temperature, top_p, seed, task=None):
            sample_seed = seed // 4096
            self.calls += 1
            index = sum(message["role"] == "assistant" for message in messages)
            selected = bad_script if sample_seed == 1 and not all_successful else script
            text = selected[index] if index < len(selected) else "I have nothing further."
            if len(text.split()) > max_tokens:
                text = " ".join(text.split()[:max_tokens])
            # Later contexts branch by member and shrink on odd turns. The
            # repeated token 102 checks that target selection is positional.
            prompt_ids = ([101, 102, 103] if index == 0 else
                          [102, 9100 + sample_seed] if index % 2 else
                          [9200 + sample_seed, index, 102, 9300 + sample_seed])
            completion_ids = ([102, 400 + sample_seed] if index == 0 else
                              [102] + [500 + index] * ((index + sample_seed) % 3))
            logprobs = [-0.25 - 0.1 * (index + offset + sample_seed)
                        for offset in range(len(completion_ids))]
            if score_action is not None:
                logprobs = score_action(prompt_ids, completion_ids, temperature)
            self.generated[(sample_seed, index)] = {
                "prompt_ids": list(prompt_ids), "completion_ids": list(completion_ids),
                "behavior_logprobs": list(logprobs),
            }
            generation = lh.TraceGeneration(
                text, len(prompt_ids), len(completion_ids), 0.0, "stop",
                prompt_ids=prompt_ids, completion_ids=completion_ids,
                sampling_temperature=temperature, sampling_top_p=top_p,
                sampling_behavior_logprobs=logprobs,
            )
            return mutate_generation(generation) if index == 1 and mutate_generation else generation

    policy = TracePolicy(script, policy_id="trace-policy")
    arm = lh.PolicyArm("base", policy, checkpoint or PolicyCheckpoint("base", 0, policy.checkpoint_sha256()))
    controller = dataclasses.replace(lh.CONTROLLER_PRESETS["full"], name="grpo-train", temperature=0.7)
    if controller_overrides:
        controller = dataclasses.replace(controller, **controller_overrides)
    grader = FakeGrader()
    root = tmp_path / "grpo-group"
    harness = lh.EpisodeHarness(
        root, arm, controller, factories={task.task_id: lh.write_fixture},
        public_tests={task.task_id: lh.fixture_public_tests},
        grader_for=lambda _task: grader,
    )
    if mutate_turn is not None:
        original_turn = harness._turn

        def turn_with_invalid_producer(*args):
            state, action = original_turn(*args)
            if args[-1] == 1:
                mutate_turn(action)
            return state, action

        harness._turn = turn_with_invalid_producer
    if mutate_commit is not None:
        original_commit = harness._commit

        def commit_with_invalid_producer(lease, step):
            original_commit(lease, mutate_commit(harness.store, step))

        harness._commit = commit_with_invalid_producer
    if mutate_verification is not None:
        original_verification = harness.store.record_verification

        def record_invalid_verification(lease, value, **kwargs):
            original_verification(lease, mutate_verification(harness.store, value), **kwargs)

        harness.store.record_verification = record_invalid_verification
    ids = harness.enroll_grpo_group(
        [lh.EpisodeSpec(task, seed) for seed in (0, 1)], group_id="train:fixture:group-0",
    )
    if run:
        harness.run()
    return harness, ids, policy


def test_verified_grpo_group_preserves_all_actions_and_exact_contexts(tmp_path):
    from vare.grpo_mlx import GRPOConfig, GroupError, group_advantages, group_samples, informative_samples

    harness, ids, policy = _fake_verified_grpo_group(tmp_path)
    root = harness.root
    metrics = lh.run_metrics(root)
    assert [episode["seed"] for episode in metrics["episodes"]] == [0, 1]
    assert [episode["final_success"] for episode in metrics["episodes"]] == [True, False]
    store = TrajectoryStore(root / "trajectories.sqlite", audit_only=True)
    original_store_digest = hashlib.sha256(store.path.read_bytes()).hexdigest()
    samples = lh.grpo_samples_from_verified_group(store, "train:fixture:group-0")
    assert hashlib.sha256(store.path.read_bytes()).hexdigest() == original_store_digest
    assert len(samples) == 2
    assert [sample.reward for sample in samples] == [1.0, 0.0]
    assert [sample.sample_index for sample in samples] == [0, 1]
    assert samples[0].prompt_ids == samples[1].prompt_ids == [101, 102, 103]
    assert samples[0].completion_ids != samples[1].completion_ids
    assert samples[0].behavior_checkpoint_id == samples[1].behavior_checkpoint_id
    assert all(sample.trajectory_segments for sample in samples)
    for member, sample in enumerate(samples):
        expected = [policy.generated[key] for key in sorted(policy.generated) if key[0] == member]
        segments = sample.trajectory_segments
        assert len(segments) == len(expected)
        assert len(segments) > 1
        assert len(segments[1].prompt_ids) < len(segments[0].prompt_ids)
        for segment, recorded in zip(segments, expected):
            assert segment.prompt_ids == recorded["prompt_ids"]
            assert segment.completion_ids == recorded["completion_ids"]
            assert segment.behavior_logprobs == recorded["behavior_logprobs"]
            assert len(segment.behavior_logprobs) == len(segment.completion_ids)
        assert sample.prompt_ids == segments[0].prompt_ids
        assert sample.completion_ids == segments[0].completion_ids
        assert sample.behavior_logprobs == segments[0].behavior_logprobs
        assert sample.meta["behavior_logprob_source"] == "sampling_distribution"
        assert sample.meta["credit_assignment"] == "full_trajectory_terminal_reward"
        assert sample.meta["trajectory_step_indices"] == list(range(len(expected)))
        assert sample.meta["n_action_segments"] == len(expected)
        assert sample.meta["n_action_tokens"] == sum(len(row["completion_ids"]) for row in expected)
        assert all(102 in segment.prompt_ids and 102 in segment.completion_ids for segment in segments)
    # Different tool histories and stopping times must not force equal later
    # prompts, action lengths or trajectory lengths across group members.
    assert samples[0].trajectory_segments[1].prompt_ids != samples[1].trajectory_segments[1].prompt_ids
    assert len(samples[0].trajectory_segments[1].completion_ids) != len(samples[1].trajectory_segments[1].completion_ids)
    assert len(samples[0].trajectory_segments) != len(samples[1].trajectory_segments)
    assert len(group_samples(samples, group_size=2)) == 1
    with pytest.raises(GroupError):
        group_samples(samples[:1])  # Recorded group completeness survives conversion.
    assert informative_samples(samples, GRPOConfig(group_size=2)) == samples
    assert group_advantages([sample.reward for sample in samples]) == ([1.0, -1.0], True)
    assert set(ids) == {row["id"] for row in store.group("train:fixture:group-0")}


def test_forced_action_fixture_reaches_mlx_update_and_checkpoint_continuation(tmp_path, request):
    """Synthetic forced actions, not model sampling or coding-capability evidence.

    The model's four input classes are deliberate. First-turn action contexts
    use rows 2/3; the distinct fake tool observations end in rows 0/1, so a
    first-action-only bridge cannot reproduce their gradient or update.
    """
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    optim = pytest.importorskip("mlx.optimizers")
    np = pytest.importorskip("numpy")
    from mlx.utils import tree_flatten

    from vare import grpo_mlx as G

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    request.addfinalizer(lambda: mx.set_default_device(previous_device))

    class ContextClassPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Embedding(4, 1024)
            self.table.weight = mx.zeros((4, 1024), dtype=mx.float32)

        def __call__(self, tokens):
            return self.table(tokens % 4)

    model = ContextClassPolicy()
    optimizer = optim.Adam(learning_rate=0.01)
    config = G.GRPOConfig(group_size=2, temperature=0.7, max_grad_norm=None)
    mx.eval(model.parameters())

    def score_forced_action(prompt, completion, temperature):
        # Score the actual chosen synthetic tokens before creating their
        # durable record. Never fill missing scores from an updated model.
        sequence = prompt + completion
        inputs = mx.array([sequence[:-1]], dtype=mx.int32)
        logits = model(inputs).astype(mx.float32) / temperature
        action_logits = logits[0, len(prompt) - 1:]
        normalizers = mx.logsumexp(action_logits, axis=-1)
        targets = mx.array(completion, dtype=mx.int32)
        selected = mx.take_along_axis(action_logits, targets[:, None], axis=-1)[:, 0]
        scores = selected - normalizers
        mx.eval(scores)
        return [float(value) for value in scores.tolist()]

    def collect(step):
        checkpoint = PolicyCheckpoint("forced-action-toy", step, G.trainable_digest(model))
        h, _, _ = _fake_verified_grpo_group(
            tmp_path / f"step-{step}", score_action=score_forced_action, checkpoint=checkpoint,
        )
        return lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")

    samples = collect(0)
    original_records = json.dumps([dataclasses.asdict(sample) for sample in samples], sort_keys=True)
    checkpoint_id = samples[0].behavior_checkpoint_id
    G.compute_behavior_logprobs(model, samples, config)
    assert json.dumps([dataclasses.asdict(sample) for sample in samples], sort_keys=True) == original_records
    advantages, informative = G.group_advantages([sample.reward for sample in samples])
    assert informative and advantages == [1.0, -1.0]
    action_count = sum(len(sample.trajectory_segments) for sample in samples)
    token_count = sum(len(segment.completion_ids)
                      for sample in samples for segment in sample.trajectory_segments)
    weights = np.asarray(model.table.weight, dtype=np.float64)
    expected_gradient = np.zeros_like(weights)
    first_contexts, later_first_contexts = set(), set()
    for sample, advantage in zip(samples, advantages):
        for segment_index, segment in enumerate(sample.trajectory_segments):
            sequence = segment.prompt_ids + segment.completion_ids
            _, mask = G.build_batch([(segment.prompt_ids, segment.completion_ids)], pad_id=0)
            assert mask[0].tolist() == ([0.0] * (len(segment.prompt_ids) - 1)
                                        + [1.0] * len(segment.completion_ids))
            if segment_index:
                later_first_contexts.add(segment.prompt_ids[-1] % 4)
            for position in range(len(segment.prompt_ids) - 1, len(sequence) - 1):
                context, target = sequence[position] % 4, sequence[position + 1]
                if segment_index == 0:
                    first_contexts.add(context)
                logits = weights[context] / config.temperature
                probabilities = np.exp(logits - np.max(logits))
                probabilities /= probabilities.sum()
                categorical_gradient = probabilities.copy()
                categorical_gradient[target] -= 1.0
                expected_gradient[context] += advantage * categorical_gradient / (token_count * config.temperature)
    assert first_contexts == {2, 3}
    assert later_first_contexts == {0, 1}
    assert np.any(np.abs(expected_gradient[[0, 1]]) > 0)
    gradients, diagnostics = G.policy_gradient(
        model, list(zip(samples, advantages)), config, n_tokens=token_count,
        policy_checkpoint_id=checkpoint_id, policy_step=0,
    )
    np.testing.assert_allclose(np.asarray(gradients["table"]["weight"]), expected_gradient,
                               rtol=1e-5, atol=1e-6)
    assert diagnostics["max_abs_log_ratio"] <= G.ON_POLICY_LOG_RATIO_TOLERANCE
    before = G.trainable_digest(model)
    later_before = np.asarray(model.table.weight)[[0, 1]].copy()
    first_update = G.train_step(model, optimizer, samples, config,
                                policy_checkpoint_id=checkpoint_id, policy_step=0)
    assert first_update["update_applied"] and first_update["optimizer_step"] == 1
    assert first_update["n_trained_action_segments"] == action_count
    assert first_update["n_tokens"] == token_count
    assert first_update["credit_assignment"] == "full_trajectory_terminal_reward"
    assert G.trainable_digest(model) != before
    assert np.any(np.asarray(model.table.weight)[[0, 1]] != later_before)
    assert json.dumps([dataclasses.asdict(sample) for sample in samples], sort_keys=True) == original_records

    destination = tmp_path / "checkpoint"
    manifest = G.save_checkpoint(destination, model, optimizer, base_model_id="fixture/context-class@v1",
                                 step=1, base_seed=71,
                                 extra={"fixture": "forced_actions", "completed_groups": 1})
    restored_model = ContextClassPolicy()
    restored_optimizer = optim.Adam(learning_rate=0.01)
    restored_manifest = G.load_checkpoint(destination, restored_model, restored_optimizer,
                                          base_model_id="fixture/context-class@v1")
    assert restored_manifest == manifest
    assert G.trainable_digest(restored_model) == G.trainable_digest(model)
    for name, value in tree_flatten(optimizer.state):
        restored = dict(tree_flatten(restored_optimizer.state))[name]
        np.testing.assert_array_equal(np.asarray(restored), np.asarray(value))

    next_samples = collect(1)
    next_records = json.dumps([dataclasses.asdict(sample) for sample in next_samples], sort_keys=True)
    next_checkpoint = next_samples[0].behavior_checkpoint_id
    continued = G.train_step(model, optimizer, next_samples, config,
                             policy_checkpoint_id=next_checkpoint, policy_step=1)
    recovered = G.train_step(restored_model, restored_optimizer, next_samples, config,
                             policy_checkpoint_id=next_checkpoint, policy_step=1)
    assert continued == recovered
    assert continued["optimizer_step"] == 2
    assert G.trainable_digest(restored_model) == G.trainable_digest(model)
    for name, value in tree_flatten(optimizer.state):
        restored = dict(tree_flatten(restored_optimizer.state))[name]
        np.testing.assert_array_equal(np.asarray(restored), np.asarray(value))
    assert json.dumps([dataclasses.asdict(sample) for sample in next_samples], sort_keys=True) == next_records


@pytest.mark.parametrize("mutation", [
    "missing_trace", "legacy_first_turn_schema", "missing_logprobs", "logprob_length",
    "positive_logprob", "bool_logprob", "logprob_digest", "logprob_source",
    "completion_digest", "response_digest", "invalid_token", "empty_prompt",
    "prompt_count", "action_count", "truncated_sampler", "changed_temperature",
])
def test_grpo_bridge_rejects_malformed_later_action_with_intact_store(tmp_path, mutation):
    def malform(action):
        # Change the producer output before append_step computes the store
        # hash chain. This tests the converter, not only raw SQLite tampering.
        trace = action["behavior_trace"]
        if mutation == "missing_trace":
            del action["behavior_trace"]
        elif mutation == "legacy_first_turn_schema":
            trace["schema"] = "vare_long_horizon_behavior_trace_v1"
        elif mutation == "missing_logprobs":
            del trace["sampling_behavior_logprobs"]
        elif mutation in {"logprob_length", "positive_logprob", "bool_logprob"}:
            if mutation == "logprob_length":
                trace["sampling_behavior_logprobs"].pop()
            else:
                trace["sampling_behavior_logprobs"][0] = True if mutation == "bool_logprob" else 0.25
            trace["sampling_behavior_logprobs_sha256"] = lh.sha256_json(trace["sampling_behavior_logprobs"])
        elif mutation == "logprob_source":
            trace["logprob_source"] = "recomputed_current_policy"
        elif mutation in {"logprob_digest", "completion_digest", "response_digest"}:
            key = {"logprob_digest": "sampling_behavior_logprobs_sha256",
                   "completion_digest": "completion_ids_sha256",
                   "response_digest": "response_sha256"}[mutation]
            trace[key] = "0" * 64
        elif mutation == "invalid_token":
            trace["completion_ids"][0] = True
            trace["completion_ids_sha256"] = lh.sha256_json(trace["completion_ids"])
        elif mutation == "empty_prompt":
            trace["prompt_ids"] = []
            trace["prompt_ids_sha256"] = lh.sha256_json([])
        elif mutation in {"prompt_count", "action_count"}:
            key = "prompt_tokens" if mutation == "prompt_count" else "completion_tokens"
            action["generation"][key] += 1
        elif mutation == "truncated_sampler":
            trace["top_p"] = 0.9
        elif mutation == "changed_temperature":
            trace["temperature"] = 0.8

    h, _, _ = _fake_verified_grpo_group(tmp_path, mutate_turn=malform)
    assert lh.audit_trajectory_store(h.store.path)["pass"] is True
    expected_reason = {
        "missing_trace": "exact action behavior trace", "legacy_first_turn_schema": "exact action behavior trace",
        "missing_logprobs": "valid sampling-time", "logprob_length": "valid sampling-time",
        "positive_logprob": "valid sampling-time", "bool_logprob": "valid sampling-time",
        "logprob_digest": "log-probability digest", "logprob_source": "valid sampling-time",
        "completion_digest": "content digest", "response_digest": "content digest",
        "invalid_token": "token IDs", "empty_prompt": "token IDs",
        "prompt_count": "token counts", "action_count": "token counts",
        "truncated_sampler": "behavior sampler", "changed_temperature": "behavior sampler",
    }[mutation]
    with pytest.raises(ValueError, match=expected_reason):
        lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")


def test_grpo_bridge_rejects_later_policy_change_even_when_registered(tmp_path):
    def change_policy(store, step):
        if step.index == 1:
            other = PolicyCheckpoint("other-policy", 1, "f" * 64)
            store.register_policy(other)
            return dataclasses.replace(step, policy=other)
        return step

    h, _, _ = _fake_verified_grpo_group(tmp_path, mutate_commit=change_policy)
    assert lh.audit_trajectory_store(h.store.path)["pass"] is True
    with pytest.raises(ValueError, match="checkpoint|policy"):
        lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")


def test_grpo_bridge_rejects_generated_action_reclassified_as_empty_padding(tmp_path):
    def hide_action(_store, step):
        if step.index == 1:
            # Erase every explicit action payload, including the observation
            # receipt. The state transition still reveals generated tokens.
            return dataclasses.replace(
                step, action={"kind": "observation", "padding": True},
                observation={}, generation=None,
            )
        return step

    h, _, _ = _fake_verified_grpo_group(tmp_path, mutate_commit=hide_action)
    assert lh.audit_trajectory_store(h.store.path)["pass"] is True
    with pytest.raises(ValueError, match="padding requires an unchanged already-stopped state"):
        lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")


def _fake_budget_stopped_grpo_group(tmp_path, *, mutate_commit=None, run=True):
    """Create a genuine token-budget terminal followed by padding, on fakes."""
    script = [lh.FIXTURE_SCRIPT[5], lh._call("run_public_tests"), *lh.FIXTURE_SCRIPT[2:]]

    def consume_remaining_budget(generation):
        # Both members have two first-action tokens. The next synthetic
        # action consumes the remaining 510 tokens of the declared budget.
        generation.completion_ids = [102] * 510
        generation.completion_tokens = 510
        generation.sampling_behavior_logprobs = [-0.5] * 510
        return generation

    return _fake_verified_grpo_group(
        tmp_path, all_successful=True, script_override=script,
        controller_overrides={"max_total_new_tokens": 512},
        mutate_generation=consume_remaining_budget, mutate_commit=mutate_commit, run=run,
    )


def test_grpo_bridge_accepts_unchanged_terminal_and_padding_without_policy_targets(tmp_path):
    h, ids, _ = _fake_budget_stopped_grpo_group(tmp_path)
    samples = lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")
    assert [sample.reward for sample in samples] == [1.0, 1.0]
    for trajectory_id, sample in zip(ids, samples):
        steps = list(h.store.iter_steps(trajectory_id))
        terminal = steps[2]
        assert terminal["action"] == {
            "kind": "observation", "terminal": True, "stop_reason": "token_budget_exhausted",
        }
        assert terminal["state"] == {**steps[1]["state"], "stop_reason": "token_budget_exhausted"}
        assert terminal["state"]["tokens_total"] == 512
        for padding in steps[3:]:
            assert padding["action"] == {"kind": "observation", "padding": True}
            assert padding["state"] == terminal["state"]
        assert sample.meta["trajectory_step_indices"] == [0, 1]
        assert sample.meta["n_action_segments"] == 2
        assert sample.meta["n_action_tokens"] == 512


@pytest.mark.parametrize("kind", ["terminal", "padding"])
@pytest.mark.parametrize("mutation", ["tokens_total", "public_passed", "initial_file_sha256"])
def test_grpo_bridge_rejects_observation_state_changes(tmp_path, kind, mutation):
    def change_state(_store, step):
        index = 2 if kind == "terminal" else 3
        if step.index != index:
            return step
        state = dict(step.state)
        if mutation == "tokens_total":
            state[mutation] += 1
        elif mutation == "public_passed":
            state[mutation] = False
        else:
            state[mutation] = {"calc/ops.py": "9" * 64}
        return dataclasses.replace(step, state=state)

    h, _, _ = _fake_budget_stopped_grpo_group(tmp_path, mutate_commit=change_state)
    assert lh.audit_trajectory_store(h.store.path)["pass"] is True
    with pytest.raises(ValueError, match="unchanged|source|initial|state"):
        lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")


def test_grpo_bridge_rejects_generated_action_after_terminal(tmp_path):
    latest_action = {}

    def resume_action(_store, step):
        if step.index == 1:
            latest_action["template"] = json.loads(json.dumps(step.action))
        if step.index == 3:
            action = json.loads(json.dumps(latest_action["template"]))
            action["turn"] = step.index
            action["operation_id"] = f"{step.event_id}:resumed"
            state = {**step.state, "stop_reason": None,
                     "tokens_total": step.state["tokens_total"] + action["generation"]["completion_tokens"]}
            return dataclasses.replace(step, action=action, state=state, observation={})
        return step

    h, _, _ = _fake_budget_stopped_grpo_group(tmp_path, mutate_commit=resume_action)
    assert lh.audit_trajectory_store(h.store.path)["pass"] is True
    with pytest.raises(ValueError, match="generated action cannot resume"):
        lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")


def test_grpo_bridge_rejects_repeated_terminal_after_stop(tmp_path):
    def repeat_terminal(_store, step):
        if step.index == 3:
            return dataclasses.replace(step, action={
                "kind": "observation", "terminal": True, "stop_reason": "token_budget_exhausted",
            })
        return step

    h, _, _ = _fake_budget_stopped_grpo_group(tmp_path, mutate_commit=repeat_terminal)
    assert lh.audit_trajectory_store(h.store.path)["pass"] is True
    with pytest.raises(ValueError, match="terminal observation may only stop an unchanged"):
        lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")


@pytest.mark.parametrize("index", [2, 3], ids=["terminal", "padding"])
def test_observation_cannot_change_workspace_files_during_replay(tmp_path, index):
    def change_file_digest(_store, step):
        if step.index == index:
            return dataclasses.replace(step, state={
                **step.state, "file_sha256": {"calc/ops.py": "9" * 64},
            })
        return step

    h, _, _ = _fake_budget_stopped_grpo_group(tmp_path, mutate_commit=change_file_digest, run=False)
    while (lease := h._claim("execute")) is not None:
        h._execute(lease)
    assert lh.audit_trajectory_store(h.store.path)["pass"] is True
    lease = h._claim("verify")
    assert lease is not None
    with pytest.raises(lh.ResumeMismatchError, match="workspace hash drift"):
        h._verify(lease)
    with pytest.raises(ValueError, match="complete verified"):
        lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")


@pytest.mark.parametrize("bad_logprob", [float("nan"), float("inf"), float("-inf"), True, 0.01])
def test_collector_rejects_invalid_sampling_logprob_before_persisting(tmp_path, bad_logprob):
    def invalid_generation(generation):
        generation.sampling_behavior_logprobs[0] = bad_logprob
        return generation

    with pytest.raises(ValueError, match="invalid sampling-time"):
        _fake_verified_grpo_group(tmp_path, mutate_generation=invalid_generation)


def test_grpo_bridge_requires_every_group_member_to_be_verified(tmp_path):
    h, _, _ = _fake_verified_grpo_group(tmp_path, run=False)
    lease = h._claim("execute")
    assert lease is not None
    h._execute(lease)
    verification = h._claim("verify")
    assert verification is not None
    h._verify(verification)
    rows = h.store.group("train:fixture:group-0")
    assert len(rows) == 2
    assert sum(row["state"] == "verified" for row in rows) == 1
    assert lh.audit_trajectory_store(h.store.path)["pass"] is True
    with pytest.raises(ValueError, match="complete verified"):
        lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")


@pytest.mark.parametrize("mutation", [
    "task_sha256", "source_revision", "workspace_patch_sha256", "grade_sha256",
    "graded_files", "candidate_sha256",
])
def test_grpo_bridge_rejects_grade_with_wrong_source_provenance(tmp_path, mutation):
    def change_verification(_store, value):
        if mutation == "graded_files":
            value.metadata[mutation] = ["calc/ungraded.py"]
        elif mutation == "candidate_sha256":
            value.metadata["raw_grade"]["candidate_sha256"] = "9" * 64
            value.metadata["grade_sha256"] = lh.sha256_json(value.metadata["raw_grade"])
        else:
            value.metadata[mutation] = "9" * (40 if mutation == "source_revision" else 64)
        return value

    h, _, _ = _fake_verified_grpo_group(tmp_path, mutate_verification=change_verification)
    assert lh.audit_trajectory_store(h.store.path)["pass"] is True
    with pytest.raises(ValueError, match="grade lacks intact task and source provenance"):
        lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")


def test_grpo_bridge_rejects_group_with_different_registered_verifiers(tmp_path):
    count = 0

    def change_verification(store, value):
        nonlocal count
        count += 1
        if count == 2:
            value.verifier_version = 2
            value.metadata["verifier_config_sha256"] = "c" * 64
            store.register_verifier(value.verifier_name, 2, "c" * 64)
        return value

    h, _, _ = _fake_verified_grpo_group(tmp_path, mutate_verification=change_verification)
    assert lh.audit_trajectory_store(h.store.path)["pass"] is True
    with pytest.raises(ValueError, match="group mixes task, policy, verifier"):
        lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")


@pytest.mark.parametrize("synthetic", ["synthetic_policy", "synthetic_task"])
def test_grpo_bridge_does_not_admit_synthetic_labels_as_real_training_data(tmp_path, synthetic):
    h, _, _ = _fake_verified_grpo_group(tmp_path, **{synthetic: True})
    with pytest.raises(ValueError, match="real-policy, real Train"):
        lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")


def test_grpo_bridge_preserves_constant_reward_without_inventing_signal(tmp_path):
    from vare.grpo_mlx import GRPOConfig, group_advantages, informative_samples

    h, _, _ = _fake_verified_grpo_group(tmp_path, all_successful=True)
    samples = lh.grpo_samples_from_verified_group(h.store, "train:fixture:group-0")
    assert [sample.reward for sample in samples] == [1.0, 1.0]
    assert all(len(sample.trajectory_segments) > 1 for sample in samples)
    assert group_advantages([sample.reward for sample in samples]) == ([0.0, 0.0], False)
    assert informative_samples(samples, GRPOConfig(group_size=2)) == []


def test_grpo_group_rejects_development_before_enrollment(tmp_path):
    task = dataclasses.replace(lh.fixture_task(), split="development")
    controller = dataclasses.replace(lh.CONTROLLER_PRESETS["full"], temperature=0.7)
    h = lh.EpisodeHarness(tmp_path / "development", stub_arm(), controller,
                          factories={task.task_id: lh.write_fixture})
    with pytest.raises(ValueError, match="only Train"):
        h.enroll_grpo_group([lh.EpisodeSpec(task, seed) for seed in (0, 1)],
                            group_id="development:fixture:group-0")
    assert h.store.group("development:fixture:group-0") == []


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
