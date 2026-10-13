# Corrected evaluation-lifecycle review

Engine SHA256: `5ca49b9e1ec94e85c0de1f3f68a003806e30d4f7b926bcf083b9bfb5b391b9fa`.

The current rollback guard checks both `not task.done()` and `not task.cancelling()` before injecting cancellation. This addresses the independently reproduced already-cancelling-evaluator failure. Original failure files remain unchanged in this directory.

## Executed confirmation

Ran `already_cancelling_evaluator_corrected.py`, retaining output in the matching JSON. Four cases cross self/external cancellation predating sibling failure with a secondary cleanup exception or ordinary cancellation propagation. Every case additionally issues three caller cancellations while cleanup is pending.

Before releasing cleanup, all four cases show:

- the round remains pending and holds its round lock;
- the candidate snapshot is retained;
- the candidate evaluator remains in cleanup;
- its cancellation request count stays at one;
- cleanup has not been interrupted.

After releasing cleanup, all four cases show:

- the original incumbent error object is preserved by identity;
- candidate cleanup completed before discard;
- both evaluators are terminal;
- candidate snapshot is removed;
- round lock is released;
- no other asyncio tasks remain pending in the probe event loop.

No repository files were changed. The production 13-case suite was read, not rerun by this reviewer.

## Static scope confirmation

The outer exception handler shields a separate recovery task, so repeated caller cancellation cannot directly cancel sibling cleanup or candidate discard. It drains evaluation tasks before discard. Same-policy evaluations omit candidate discard but still drain both owned tasks. Secondary evaluation failures are retrieved without replacing the initially observed exception. Discard errors are reported as unknown policy state with the original cause.

The supported contract requires hooks to make their own cleanup cancellation-safe and ensure terminal coroutine means resource work is complete. A hook that suppresses prior cancellation and resumes ordinary work should clear the cancellation request state, or the engine can only wait for it. Noncooperative hooks can make cancellation latency unbounded. Detached worker ownership, rollout-phase tasks, and success-path promotion/discard cancellation are outside this bounded evaluation cleanup change.

No new actionable issue was found within this stated contract after the guard correction.
