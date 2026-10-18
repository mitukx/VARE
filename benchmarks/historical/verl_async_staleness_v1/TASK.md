# Async rollout staleness accounting

Fix staleness accounting in `FullyAsyncRollouter` so a policy reset reflects samples that were dispatched and have not been removed from the learner queue. The lifetime dispatch counter must be initialized in `__init__` and preserved across resets. The result must not depend on when completed asyncio tasks are reaped or on two non-atomic queue reads. A sample waiting for rollout capacity must not have its staleness budget reset before it is actually dispatched, and both dispatch counters must be updated on every successful task dispatch.

Limit changes to `verl/experimental/fully_async_policy/fully_async_rollouter.py`. Preserve the trainer API and make no changes to task files, grader files, or unrelated source.

The baseline is pinned to verl main at `8ced5d2f4b148c2991217554e0c6f03028fa1470`. The graded source file is byte-for-byte identical to the issue-report revision `3efe38c759c14622fd1b2c9e3679f2d02f86bdac`; the issue remains open only as `not planned` and the defect remains present at the pinned main revision.

The grader runs CPU fixtures against the production `reset_staleness` method and any same-class helpers it calls, checks lifetime-counter initialization and preservation, then checks that the worker reserves a sample only after its capacity wait and reaches both counter updates after dispatch. It tests publication between separate queue reads and a consumed control record with no matching dispatch. It does not start Ray, load a model, or measure training throughput.

This is a public historical issue whose fix and reproduction are already visible in a contributor fork. Treat the task only as evaluator calibration and a source-level regression exercise. It cannot measure agent discovery, model learning, capability improvement, or production-scale RL behavior.
