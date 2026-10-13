"""Cancellation must not release a model while a generation worker uses it."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from vare.integrations.rvl_grpo import RVLGRPOHooks
from vare.types import Task


class Trainer:
    def snapshot_training_state(self):
        return {}

    def restore_training_state(self, state):
        pass


class ThreadBackend:
    def __init__(self, fail):
        self.fail = fail
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.calls = 0

    def work(self, prompt_id, prompt):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            try:
                if not self.release.wait(5):
                    raise TimeoutError("test worker was not released")
                if self.fail:
                    raise RuntimeError("worker failure during cancellation")
            finally:
                self.finished.set()
        return [SimpleNamespace(prompt_id=prompt_id, prompt=prompt, response="ok",
                                logprob=-0.1, token_count=1, latency_s=0.001, metadata={})]

    async def generate(self, prompt_id, prompt, **kwargs):
        return await asyncio.to_thread(self.work, prompt_id, prompt)


@pytest.mark.parametrize("operation", ["evaluate", "rollout"])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_repeated_cancel_drains_worker_before_next_model_access(operation, worker_fails):
    async def run():
        backend = ThreadBackend(worker_fails)
        hooks = RVLGRPOHooks(
            backend=backend, trainer=Trainer(), eval_tasks=[Task("eval", "prompt")],
            score_fn=lambda *_: 1.0, generation_factory=SimpleNamespace,
            verified_generation_factory=SimpleNamespace,
        )
        errors = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: errors.append(context))
        task = asyncio.create_task(hooks.evaluate("policy-0") if operation == "evaluate"
                                   else hooks.rollout(Task("first", "prompt"), "policy-0", 0, 0))
        follower = None
        try:
            assert await asyncio.to_thread(backend.entered.wait, 5)
            task.cancel()
            # Observe that a follower has actually attempted model access.
            attempted = asyncio.Event()

            async def next_rollout():
                attempted.set()
                return await hooks.rollout(Task("next", "prompt"), "policy-0", 0, 1)

            follower = asyncio.create_task(next_rollout())
            await attempted.wait()
            task.cancel()
            await asyncio.sleep(0)  # Deliver the second cancellation.
            assert not task.done()
            assert hooks._model_lock.locked()
            assert not backend.finished.is_set()
            assert backend.calls == 1
            assert not follower.done()
            backend.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await follower
            assert backend.finished.is_set()
            assert backend.calls == 2
            assert not hooks._model_lock.locked()
            assert hooks._loaded_id == "policy-0"
            assert not errors  # Worker exceptions must be retrieved.
        finally:
            backend.release.set()
            await asyncio.gather(task, *([follower] if follower else []), return_exceptions=True)

    asyncio.run(run())
