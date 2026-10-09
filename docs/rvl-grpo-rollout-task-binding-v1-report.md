# RVL GRPO rollout task binding v1

## Question

Can `RVLGRPOHooks.rollout` return an `Attempt` for the dispatched task while storing token-exact generation data for another prompt?

## Why this matters

The VARE engine verifies `Attempt.task` and `Attempt.output`. The RVL GRPO trainer later reconstructs its training sample from `Attempt.metadata["rvl_generation"]`. If those task identities differ, the verifier can score one prompt/response while the optimizer receives token IDs and prompt text from another. Matching the outer `Attempt.task` alone does not protect this adapter boundary.

## Frozen protocol and baseline

Protocol `protocols/rvl_grpo_rollout_task_binding_v1.lock.json` was committed before the adapter change. The test varies the returned generation's `prompt_id` and `prompt` independently, with an exact-match positive control. It intentionally does not test family metadata, policy/version lineage, token IDs, or model quality.

At baseline `d8cc153dd242d3b989efcc42eea79408fbbb75c7`, the test produced `2 failed, 1 passed`: both mismatched-generation cases returned an `Attempt` without error, and the positive control worked. In this baseline behavior, `Attempt.task` remained the dispatched `task-a` while the retained `rvl_generation` fields could identify `task-b` or a different prompt. The baseline log is retained in `results/rvl-grpo-rollout-task-binding-v1/run-1/baseline-pytest.log`.

## Fix and results

`RVLGRPOHooks.rollout` now requires the backend's returned generation `prompt_id` and `prompt` to equal the dispatched task's ID and prompt before embedding the raw generation in an `Attempt`. The frozen test passed 3/3 after the change. The combined rollout-binding, RVL hook, RVL group-boundary, and engine task-binding tests passed 15 tests.

This is an E0 adapter-contract result. It does not run an optimizer, inspect a real RVL backend's behavior, establish verifier correctness, prove all metadata provenance, or show task-success or capability improvement. It does not enforce policy ID/version equality, which would conflict with allowed lag semantics.

## Decision

**Keep the fix.** It closes a concrete mismatch between the prompt an oracle evaluates and the token-exact generation an RL trainer consumes. Do not treat the repeated task-binding pattern as a new method or claim of policy improvement. The next priority remains independent reproduction or a real-model task-success update only after the existing no-cost feasibility gates pass.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_rvl_grpo_rollout_task_binding.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_rvl_grpo_rollout_task_binding.py tests/test_rvl_grpo_hooks.py tests/test_rvl_grpo_group_boundary.py tests/test_rollout_task_binding.py
```
