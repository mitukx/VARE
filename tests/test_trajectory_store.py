from dataclasses import replace
import json
import multiprocessing
import os
import signal
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from vare.durable import FenceError, RuntimeMismatchError
from vare.evidence import sha256_json
from vare.trajectory_store import (
    AgentStep, BackpressureError, PolicyCheckpoint, TrajectoryLease,
    TrajectoryStore, audit_trajectory_store,
)
from vare.types import Task, Verification


POLICY = PolicyCheckpoint("p0", 0, "a" * 64)


def step(index=0, policy=POLICY):
    return AgentStep(index, f"event-{index}", policy,
                     {"kind": "tool", "operation_id": f"op-{index}", "semantics": "at_least_once"},
                     {"result": index + 1}, {"counter": index + 1},
                     {"prompt": "count", "response": str(index + 1),
                      "response_token_ids": [index + 1], "response_logprobs": [-.1]})


@pytest.fixture
def store(tmp_path):
    value = TrajectoryStore.create(tmp_path / "state.db", run_id="test-run", max_inflight=2)
    value.register_policy(POLICY)
    value.register_verifier("oracle", 0, "b" * 64)
    return value


def submit(store, ident="t0", **kw):
    return store.submit(ident, Task("task", "count"), expected_steps=2,
                        group_id=ident, member=0, group_size=1, now=100, **kw)


def verify(store, lease, **metadata):
    row = store.trajectory(lease.trajectory_id)
    return store.record_verification(lease, Verification(1., True, 1., 0, "oracle", trusted=True,
        metadata={"trajectory_sha256": row["head"], "verifier_config_sha256": "b" * 64, **metadata}), now=103)


def killable_worker(path, pipe):
    store = TrajectoryStore(path)
    lease = store.claim("execute", "dying", lease_seconds=.1, now=100)
    store.append_step(lease, step(), now=100.01)
    pipe.send(lease)
    pipe.recv()


def test_step_resume_sigkill_and_effective_once(store):
    submit(store)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=killable_worker, args=(store.path, child))
    process.start()
    try:
        assert parent.poll(10), "worker did not reach durable boundary"
        old = parent.recv()
        os.kill(process.pid, signal.SIGKILL)
        process.join(10)
        assert process.exitcode == -signal.SIGKILL
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        parent.close()
        child.close()
    reopened = TrajectoryStore(store.path)
    lease = reopened.claim("execute", "replacement", now=101)
    assert lease.generation == 2
    assert reopened.resume_state("t0")["state"] == {"counter": 1}
    with pytest.raises(FenceError):
        reopened.append_step(old, step(1), now=101)
    assert reopened.append_step(lease, step(), now=101) is False
    assert reopened.append_step(lease, step(1), now=101) is True
    reopened.finish(lease, now=101)
    verifier = reopened.claim("verify", "v0", now=102)
    verify(reopened, verifier)
    row = reopened.trajectory("t0")
    reopened.close_verified("t0", head=row["head"])
    audit = audit_trajectory_store(reopened.path)
    assert audit["steps"] == 2 and audit["completions"] == 1
    assert audit["effective_duplicates"] == 0
    with reopened.connection() as db:
        attempts = [tuple(r) for r in db.execute("SELECT generation,state FROM attempts WHERE stage='execute' ORDER BY generation")]
    assert attempts == [(1, "lease_expired"), (2, "done")]


def test_backpressure_includes_unconsumed_verified_trajectories(store):
    assert submit(store) is True
    assert submit(store) is False
    submit(store, "t1")
    with pytest.raises(BackpressureError):
        submit(store, "t2")
    lease = store.claim("execute", "worker", now=100)
    for i in range(2):
        store.append_step(lease, step(i), now=100)
    store.finish(lease, now=100)
    verifier = store.claim("verify", "verifier", now=102)
    verify(store, verifier)
    with pytest.raises(BackpressureError):
        submit(store, "t2")
    store.close_verified("t0", head=store.trajectory("t0")["head"])
    assert submit(store, "t2")


def test_partial_or_out_of_order_cannot_complete(store):
    submit(store)
    lease = store.claim("execute", "worker", now=100)
    with pytest.raises(ValueError, match="out-of-order"):
        store.append_step(lease, step(1), now=100)
    with pytest.raises(ValueError, match="incomplete"):
        store.finish(lease, now=100)
    assert store.claim("verify", "verifier", now=100) is None
    assert audit_trajectory_store(store.path)["steps"] == 0


def test_atomic_partial_persistence_failure(store):
    submit(store)
    lease = store.claim("execute", "worker", now=100)
    with store.connection(True) as db:
        db.execute("CREATE TRIGGER fail_step BEFORE INSERT ON steps BEGIN SELECT RAISE(ABORT,'injected step failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.append_step(lease, step(), now=100)
    assert store.resume_state("t0")["next_step"] == 0
    assert audit_trajectory_store(store.path)["steps"] == 0
    with store.connection(True) as db:
        db.execute("DROP TRIGGER fail_step")
    store.retry(lease, now=100)
    replacement = store.claim("execute", "replacement", now=100)
    assert store.append_step(replacement, step(), now=100)


def test_heartbeat_expiry_and_cancellation(store):
    submit(store)
    lease = store.claim("execute", "worker", lease_seconds=1, now=100)
    store.heartbeat(lease, lease_seconds=2, now=100.5)
    assert store.claim("execute", "other", now=101.5) is None
    with pytest.raises(FenceError):
        store.heartbeat(lease, now=102.5)
    newer = store.claim("execute", "other", now=102.5)
    assert newer.generation == 2
    assert store.cancel("t0") is True
    assert store.cancel("t0") is False
    with pytest.raises(FenceError):
        store.append_step(newer, step(), now=102.6)
    assert store.trajectory("t0")["state"] == "cancelled"


def test_retry_budget_and_lifetime_preserve_history(store):
    submit(store, max_attempts=2)
    first = store.claim("execute", "one", now=100)
    store.retry(first, now=100)
    second = store.claim("execute", "two", now=100)
    store.retry(second, now=100)
    assert store.claim("execute", "three", now=100) is None
    assert store.trajectory("t0")["reason"] == "retry_exhausted"
    submit(store, "t1", lifetime_seconds=1)
    assert store.claim("execute", "one", now=102) is None
    assert store.trajectory("t1")["reason"] == "lifetime_exceeded"
    assert audit_trajectory_store(store.path)["pass"]


@pytest.mark.parametrize("fault", ["policy_id", "policy_version", "checkpoint", "group", "verifier", "trace", "event", "tokens", "bytes"])
def test_provenance_and_bounds_fail_closed(store, fault):
    submit(store)
    lease = store.claim("execute", "worker", now=100)
    item = step()
    if fault == "policy_id":
        item = replace(item, policy=replace(POLICY, policy_id="unknown"))
    elif fault == "policy_version":
        item = replace(item, policy=replace(POLICY, version=1))
    elif fault == "checkpoint":
        item = replace(item, policy=replace(POLICY, checkpoint_sha256="c" * 64))
    elif fault == "tokens":
        item.generation["response_logprobs"].append(-.2)
    elif fault == "bytes":
        item.state["oversize"] = "x" * 65536
    elif fault == "event":
        store.append_step(lease, item, now=100)
        item.observation["changed"] = True
    elif fault == "group":
        with pytest.raises(ValueError, match="group"):
            store.submit("other", Task("other-task", "count"), expected_steps=2,
                         group_id="t0", member=0, group_size=2, now=100)
        return
    elif fault in {"verifier", "trace"}:
        for i in range(2):
            store.append_step(lease, step(i), now=100)
        store.finish(lease, now=100)
        verifier = store.claim("verify", "v", now=102)
        key = "verifier_config_sha256" if fault == "verifier" else "trajectory_sha256"
        with pytest.raises(ValueError, match="verification"):
            verify(store, verifier, **{key: "c" * 64})
        return
    with pytest.raises(ValueError):
        store.append_step(lease, item, now=100)


def test_mixed_policy_steps_are_recorded_without_relabeling(store):
    submit(store)
    policy1 = PolicyCheckpoint("p1", 1, "c" * 64)
    store.register_policy(policy1)
    with pytest.raises(ValueError, match="rebound"):
        store.register_policy(replace(policy1, checkpoint_sha256="d" * 64))
    lease = store.claim("execute", "worker", now=100)
    store.append_step(lease, step(), now=100)
    store.append_step(lease, step(1, policy1), now=100)
    assert [s["policy"]["version"] for s in store.iter_steps("t0")] == [0, 1]
    store.finish(lease, now=100)
    assert audit_trajectory_store(store.path)["pass"]


def test_tampering_and_runtime_change_are_rejected(store, tmp_path):
    submit(store)
    lease = store.claim("execute", "worker", now=100)
    store.append_step(lease, step(), now=100)
    backup = tmp_path / "snapshot.db"
    store.backup(backup)
    assert audit_trajectory_store(backup)["steps"] == 1
    with sqlite3.connect(backup) as db:
        db.execute("UPDATE steps SET hash=?", ("0" * 64,))
    with pytest.raises(ValueError, match="mismatch"):
        audit_trajectory_store(backup)
    with store.connection(True) as db:
        config = store.config()
        config["source_sha256"] = "0" * 64
        db.execute("UPDATE meta SET value=? WHERE key='config'", (json.dumps(config),))
    with pytest.raises(RuntimeMismatchError):
        store.claim("execute", "worker", now=100)


@pytest.mark.parametrize("value", [True, -1, 11, 1.1])
def test_lease_generation_is_strict(value):
    with pytest.raises(ValueError):
        TrajectoryLease("t", "execute", value, "worker")


def test_verifier_rebinding_rejected(store):
    with pytest.raises(ValueError, match="rebound"):
        store.register_verifier("oracle", 0, sha256_json("different"))


def test_claim_race_has_distinct_authoritative_owners(store):
    submit(store)
    submit(store, "t1")
    with ThreadPoolExecutor(max_workers=2) as pool:
        leases = list(pool.map(lambda owner: store.claim("execute", owner, now=100), ["one", "two"]))
    assert len({lease.trajectory_id for lease in leases}) == 2
    assert {lease.generation for lease in leases} == {1}


def test_page_budget_fails_atomically(tmp_path):
    store = TrajectoryStore.create(tmp_path / "bounded.db", run_id="storage-budget", storage_bytes=1024**2)
    store.register_policy(POLICY)
    store.submit("t", Task("task", "x"), expected_steps=64, group_id="g", member=0, group_size=1, now=100)
    lease = store.claim("execute", "worker", now=100)
    committed = 0
    for index in range(64):
        item = step(index)
        item.state["payload"] = "x" * 40000
        try:
            store.append_step(lease, item, now=100)
        except sqlite3.OperationalError as exc:
            assert "full" in str(exc)
            break
        committed += 1
    assert 0 < committed < 64
    assert store.resume_state("t")["next_step"] == committed
    assert audit_trajectory_store(store.path)["steps"] == committed
    assert store.path.stat().st_size <= 1024**2


def test_current_state_needs_corresponding_history(store):
    submit(store)
    lease = store.claim("execute", "worker", now=100)
    for i in range(2):
        store.append_step(lease, step(i), now=100)
    # A DB-state-only terminal edit must not masquerade as completion.
    with store.connection(True) as db:
        db.execute("UPDATE trajectories SET state='complete'")
    with pytest.raises(ValueError, match="missing completion"):
        audit_trajectory_store(store.path)
