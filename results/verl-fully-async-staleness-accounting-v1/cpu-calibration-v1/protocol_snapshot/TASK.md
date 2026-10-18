# Async rollout staleness accounting

Fix staleness accounting in `FullyAsyncRollouter` so a policy reset reflects samples that were dispatched and have not been removed from the learner queue. The result must not depend on when completed asyncio tasks are reaped. A sample waiting for rollout capacity must not have its staleness budget reset before it is actually dispatched.

Limit changes to `verl/experimental/fully_async_policy/fully_async_rollouter.py`. Preserve the trainer API and make no changes to task files, grader files, or unrelated source.

The grader runs CPU fixtures against the production `reset_staleness` method and checks that the worker reserves a sample only after its capacity wait. It does not start Ray, load a model, or measure training throughput.

This is a public historical issue whose fix and reproduction are already visible in a contributor fork. Treat the task only as evaluator calibration and a source-level regression exercise. It cannot measure agent discovery, model learning, capability improvement, or production-scale RL behavior.
