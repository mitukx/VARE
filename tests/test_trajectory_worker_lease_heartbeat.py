import asyncio

from vare.trajectory_store import AgentStep, PolicyCheckpoint, TrajectoryStore
from vare.types import Task


def test_paused_step_iterator_does_not_hold_store_lock_from_heartbeat(tmp_path):
    store = TrajectoryStore.create(tmp_path / "state.db", run_id="heartbeat-read-test")
    policy = PolicyCheckpoint("p0", 0, "a" * 64)
    store.register_policy(policy)
    store.submit("t0", Task("task", "steps"), expected_steps=2, group_id="g0", member=0, group_size=1)
    lease = store.claim("execute", "worker", lease_seconds=2)
    for index in range(2):
        store.append_step(
            lease,
            AgentStep(
                index,
                f"e{index}",
                policy,
                {"kind": "tool", "operation_id": f"op{index}", "semantics": "at_least_once"},
                {"index": index},
                {"index": index},
            ),
        )

    iterator = store.iter_steps("t0")
    assert next(iterator)["index"] == 0

    async def renew_while_paused():
        heartbeat = asyncio.create_task(asyncio.to_thread(store.heartbeat, lease, lease_seconds=2))
        completed = True
        try:
            await asyncio.wait_for(asyncio.shield(heartbeat), timeout=0.5)
        except asyncio.TimeoutError:
            completed = False
        finally:
            iterator.close()
            await heartbeat
        assert completed, "paused step iteration blocked the worker lease heartbeat"

    asyncio.run(renew_while_paused())
