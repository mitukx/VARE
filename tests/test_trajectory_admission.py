from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from vare.durable import FenceError
from vare.integrations.rvl_grpo import RVLGRPOHooks
from vare.trajectory_admission import TrajectoryRewardRoot, admit_terminal_grpo_group
from vare.trajectory_store import AgentStep, PolicyCheckpoint, TrajectoryStore
from vare.types import Task, Verification


P0 = PolicyCheckpoint("p0", 0, "a" * 64)
P1 = PolicyCheckpoint("p1", 1, "c" * 64)


def group(tmp_path, fault=None):
    store = TrajectoryStore.create(tmp_path / "state.db", run_id="admission-test", max_inflight=4)
    root = TrajectoryRewardRoot(b"x" * 32, "admission-test", "d" * 64)
    store.register_policy(P0)
    store.register_policy(P1)
    store.register_verifier("v", 0, "b" * 64)
    store.register_verifier("v", 1, "e" * 64)
    rule = "unsupported" if fault == "attribution" else "terminal_action_on_policy_grpo_v1"
    task = Task("task", "tools", metadata={"learning_rule": rule})
    for member in range(1 if fault == "partial" else 2):
        store.submit(f"t{member}", task, expected_steps=2, group_id="g", member=member, group_size=2, now=100)
    for member in range(1 if fault == "partial" else 2):
        lease = store.claim("execute", "worker", now=100)
        for i in range(2):
            policy = P1 if fault == "mixed" and i == 1 else P0
            prompt = f"different-{member}" if fault == "prompt" else "same terminal prompt"
            generation = {"prompt_id": "task", "prompt": prompt, "response": "yes", "logprob": -.1, "token_count": 1, "latency_s": 0.,
                          "metadata": {"prompt_token_ids": [0], "response_token_ids": [1], "response_token_logprobs": [-.1]}}
            if fault == "no_logprobs":
                generation["metadata"]["response_token_logprobs"] = None
            if fault == "no_tokens":
                generation["metadata"]["response_token_ids"] = []
                generation["metadata"]["response_token_logprobs"] = []
                generation["token_count"], generation["logprob"] = 0, 0.
            step = AgentStep(i, f"e{i}", policy, {"kind": "inference"}, {}, {"cursor": i}, generation)
            store.append_step(lease, step, now=100)
        store.finish(lease, now=100)
    if fault != "unverified":
        for member in range(1 if fault == "partial" else 2):
            lease = store.claim("verify", "verifier", now=100)
            head = store.trajectory(lease.trajectory_id)["head"]
            value = Verification(.5, True, 1., 0, "v", trusted=True,
                                 metadata={"verifier_config_sha256": "b" * 64, "trajectory_sha256": head})
            if fault != "unsigned":
                value = root.attest(lease.trajectory_id, head, value)
            store.record_verification(lease, value, now=100)
    return store, root


def admit(store, root, **kw):
    return admit_terminal_grpo_group(store, "g", active_policy=P0, verifier_name="v",
                                    verifier_version=0, verifier_config_sha256="b" * 64,
                                    trust_root=root, **kw)


def test_signed_whole_group_and_atomic_consumer_ack(tmp_path):
    store, root = group(tmp_path)
    result = admit(store, root)
    assert result.admitted and len(result.experiences) == 2
    assert result.experiences[0].attempt.task.prompt == "same terminal prompt"
    assert result.experiences[0].attempt.task.id == "task"
    # Exercise the existing adapter's serialization contract without a model
    # or a fake optimizer. Native Generation constructor keys must survive.
    adapter = SimpleNamespace(_generation_factory=SimpleNamespace, _verified_factory=SimpleNamespace)
    native = RVLGRPOHooks._to_verified_generation(adapter, result.experiences[0])
    assert native.generation.prompt_id == "task"
    assert native.generation.metadata["response_token_ids"] == [1]
    assert native.generation.metadata["response_token_logprobs"] == [-.1]
    store.close_group("g", result.heads, decision={"rule": "terminal_action_on_policy_grpo_v1"})
    assert store.counts() == {"consumed": 2}
    with pytest.raises(FenceError):
        store.close_group("g", result.heads)


@pytest.mark.parametrize("fault,reason", [
    ("partial", "incomplete_group"), ("unverified", "unverified_group"),
    ("unsigned", "unauthenticated_reward"), ("mixed", "multi_policy_trajectory"),
    ("attribution", "unsupported_reward_attribution"), ("prompt", "heterogeneous_group_prompt"),
    ("no_tokens", "missing_terminal_behavior"), ("no_logprobs", "missing_terminal_behavior"),
])
def test_unsupported_training_signals_fail_closed(tmp_path, fault, reason):
    store, root = group(tmp_path, fault)
    result = admit(store, root)
    assert not result.admitted and result.experiences == ()
    assert reason in result.reasons


def test_staleness_recomputed_after_wait_and_verifier_refresh(tmp_path):
    store, root = group(tmp_path)
    assert admit(store, root).admitted
    stale = admit_terminal_grpo_group(store, "g", active_policy=P1, verifier_name="v", verifier_version=1,
                                     verifier_config_sha256="e" * 64, trust_root=root)
    assert stale.reasons == ("off_policy_checkpoint", "verifier_stale_or_mismatched")
    assert stale.policy_lags == (1, 1) and stale.verifier_lags == (1, 1)


@pytest.mark.parametrize("fault", ["reward", "replayed_receipt", "foreign_run", "trace"])
def test_authentication_binds_reward_member_run_and_trace(tmp_path, fault):
    store, root = group(tmp_path)
    if fault == "foreign_run":
        root = replace(root, run_id="foreign-run")
    else:
        with store.connection(True) as db:
            rows = list(db.execute("SELECT * FROM trajectories ORDER BY id"))
            value = json.loads(rows[0]["verification"])
            if fault == "reward":
                value["score"] = .9
            elif fault == "replayed_receipt":
                value["metadata"]["trajectory_reward_receipt"] = json.loads(rows[1]["verification"])["metadata"]["trajectory_reward_receipt"]
            else:
                value["metadata"]["trajectory_sha256"] = "f" * 64
            db.execute("UPDATE trajectories SET verification=? WHERE id=?", (json.dumps(value), rows[0]["id"]))
    result = admit(store, root)
    assert not result.admitted and "unauthenticated_reward" in result.reasons


def test_incomplete_group_cannot_be_acknowledged(tmp_path):
    store, root = group(tmp_path, "partial")
    result = admit(store, root)
    with pytest.raises(FenceError):
        store.close_group("g", result.heads)
