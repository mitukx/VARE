import asyncio
import threading
import time

import pytest

from vare.trajectory_admission import TrajectoryRewardRoot, admit_terminal_grpo_group
from vare.trajectory_store import AgentStep, BackpressureError, PolicyCheckpoint, TrajectoryStore
from vare.trajectory_workers import execute_trajectories, owned_io, verify_trajectories
from vare.types import Task, Verification


def test_independent_stages_bounded_backlog_and_slow_step_heartbeats(tmp_path):
    async def scenario():
        store = TrajectoryStore.create(tmp_path / "state.db", run_id="async-test", max_inflight=4)
        policy = PolicyCheckpoint("p0", 0, "a" * 64)
        store.register_policy(policy)
        store.register_verifier("v", 0, "b" * 64)
        root = TrajectoryRewardRoot(b"x" * 32, "async-test", "c" * 64)
        stop = asyncio.Event()
        pressure = 0
        heartbeats = 0
        renew = store.heartbeat

        def recorded_heartbeat(*args, **kwargs):
            nonlocal heartbeats
            result = renew(*args, **kwargs)
            heartbeats += 1
            return result

        store.heartbeat = recorded_heartbeat

        async def step_fn(resumed, trajectory_id):
            index = resumed["next_step"]
            await asyncio.sleep(.9 if trajectory_id == "g0-0" and index == 0 else .001)
            return AgentStep(index, f"e{index}", policy, {"kind": "inference"}, {}, {"index": index},
                             {"prompt_id": "task", "prompt": "same", "response": "x", "logprob": -.1, "token_count": 1, "latency_s": 0.,
                              "metadata": {"model": "fixture-model", "sampling_temperature": 1., "logprob_distribution": "neutral_temperature_scaled_policy", "prompt_token_ids": [0], "response_token_ids": [1], "response_token_logprobs": [-.1]}})

        async def verify_fn(store, ident):
            row = await owned_io(store.trajectory, ident)
            value = Verification(.5, True, 1., 0, "v", trusted=True,
                                 metadata={"verifier_config_sha256": "b" * 64, "trajectory_sha256": row["head"]})
            return root.attest(ident, row["head"], value)

        async def producer():
            nonlocal pressure
            for group in range(8):
                for member in range(2):
                    while True:
                        try:
                            await owned_io(store.submit, f"g{group}-{member}", Task("task", "x", metadata={"learning_rule": "terminal_action_on_policy_grpo_v1"}), expected_steps=2, group_id=f"g{group}", member=member, group_size=2)
                            break
                        except BackpressureError:
                            pressure += 1
                            await asyncio.sleep(.005)

        async def consumer():
            done = 0
            while done < 8:
                for ident in await owned_io(store.verified_groups):
                    result = await owned_io(admit_terminal_grpo_group, store, ident, active_policy=policy,
                                            verifier_name="v", verifier_version=0, verifier_config_sha256="b" * 64, trust_root=root)
                    if result.admitted:
                        await owned_io(store.close_group, ident, result.heads)
                        done += 1
                await asyncio.sleep(.001)

        workers = [asyncio.create_task(execute_trajectories(store, f"w{i}", step_fn, stop, lease_seconds=.6)) for i in range(2)]
        workers.append(asyncio.create_task(verify_trajectories(store, "v", verify_fn, stop, lease_seconds=.6)))
        try:
            async with asyncio.timeout(15):
                await asyncio.gather(producer(), consumer())
        finally:
            stop.set()
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        assert store.counts() == {"consumed": 16}
        assert pressure > 0
        assert heartbeats > 0
        with store.connection() as db:
            assert db.execute("SELECT count(*) FROM attempts WHERE state NOT IN ('done','lease_expired')").fetchone()[0] == 0
            first_verified = db.execute("SELECT min(sequence) FROM events WHERE json_extract(event,'$.kind')='trajectory_verified'").fetchone()[0]
            last_complete = db.execute("SELECT max(sequence) FROM events WHERE json_extract(event,'$.kind')='trajectory_complete'").fetchone()[0]
        assert first_verified < last_complete

    asyncio.run(scenario())


def test_cancelled_io_is_terminal_before_returning():
    async def scenario():
        began, finished = threading.Event(), threading.Event()

        def slow():
            began.set()
            time.sleep(.04)
            finished.set()

        task = asyncio.create_task(owned_io(slow))
        while not began.is_set():
            await asyncio.sleep(.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    asyncio.run(scenario())
