# Engine rollout task-binding regression v1

## Question

Can `CapabilityLoop` verify and train on an `Attempt` for a different task than the one dispatched by the loop?

## Frozen property and scope

The protocol `protocols/engine_rollout_task_binding_v1.lock.json` was committed before validating the fix. A rollout result must be an `Attempt`, and its `Task` must match the dispatched task's `id`, `prompt`, and `family` before verification, replay admission, or training. Exact policy-version equality is deliberately excluded because configured lag semantics allow older policies. Arbitrary `Task.metadata` is not deeply compared.

This is an engine trust-boundary regression, not a study of verifier truthfulness, model quality, policy improvement, or capability.

## Baseline counterexample

At baseline commit `ff434df132b026aed60cab833c2ccb92b51474f1`, the frozen test failed 3/4 cases: task-ID, prompt, and family mismatches did not raise. The fourth case is the valid-task positive control. A direct probe dispatched `task-a`; its hook returned `task-b` with the same family and current policy/verifier versions. The verifier saw `task-b`, training received `task-b`, and one experience was admitted. The baseline source hash and observation are retained in `results/engine-rollout-task-binding-v1/run-1/baseline-observation.json`.

## Change and evidence

`CapabilityLoop._one_rollout` now rejects a non-`Attempt` result and rejects a task whose ID, prompt, or family differs from the dispatched task, before calling the verifier. The frozen regression passed 4/4 after the change. The focused task-binding, concurrency, transaction, grouped-replay, lag, and replay-identity suite passed 18 tests.

The full local suite completed with four failures. All concern the pre-existing `smoke-stable-logsumexp` oracle/evaluator path. The exact same four failures reproduce from baseline commit `ff434df` in an isolated detached worktree; the baseline and current focused smoke suites each report `4 failed, 10 passed`. Logs are retained under `results/engine-rollout-task-binding-v1/run-1/`. These failures are not caused by the task-binding patch, but remain unresolved repository failures. The local suite is not a CI pass.

## Decision and limits

**Decision: keep the narrow fix; continue to independent review/reproduction.** This establishes one local engine contract under deterministic hooks. It does not show that all adapters construct the intended prompt, that distributed or RVL paths bind task provenance correctly, that task metadata is identical, or that any policy improves. Do not count this as research evidence for a new RL method or real-model capability.

The next check should inspect adapter boundaries for equivalent task-identity trust gaps, beginning with any generation adapter that reconstructs an `Attempt` from prompt IDs. Do not broaden this patch into policy-lineage checks without a separate protocol that respects allowed lag.

## Reproduction

From the repository root with the project test environment installed:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_rollout_task_binding.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_rollout_task_binding.py tests/test_engine_concurrency.py tests/test_engine_concurrent_round_transactions.py tests/test_grouped_replay.py tests/test_lag.py tests/test_replay_group_prompt_identity.py
PYTHONPATH=src .venv/bin/python -m pytest -q
```

To inspect the baseline failure, check out `ff434df132b026aed60cab833c2ccb92b51474f1` and run the frozen test file from this commit with that checkout's source on `PYTHONPATH`. The 3 expected identity-mismatch cases fail on baseline and pass with the fix.
