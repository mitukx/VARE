# RVL GRPO rollout task binding v1

## Question

Can `RVLGRPOHooks.rollout` return an `Attempt` for the dispatched task while storing token-exact generation data for another prompt?

## Why this matters

The VARE engine verifies `Attempt.task` and `Attempt.output`. The RVL GRPO trainer later reconstructs its training sample from `Attempt.metadata["rvl_generation"]`. If those task identities differ, the metadata permits the verifier to score one prompt/response while the optimizer is given generation data labeled as another prompt. Matching the outer `Attempt.task` alone does not protect this adapter boundary.

## Frozen protocol and baseline

Protocol `protocols/rvl_grpo_rollout_task_binding_v1.lock.json` was committed before the adapter change. The test varies the returned generation's `prompt_id` and `prompt` independently, with an exact-match positive control. It intentionally does not test family metadata, policy/version lineage, token IDs, or model quality.

At baseline `d8cc153dd242d3b989efcc42eea79408fbbb75c7`, the test produced `2 failed, 1 passed`: both mismatched-generation cases returned an `Attempt` without error, and the positive control worked. A direct baseline replay confirmed that `Attempt.task` remained the dispatched `task-a` while the retained `rvl_generation` fields could identify `task-b` or a different prompt; the observation is retained in `results/rvl-grpo-rollout-task-binding-v1/run-1/baseline-observation.json`. The fake generation contained no token IDs and the trainer step was not run, so the test demonstrates the inconsistent identity fields and the reachable adapter path, not an observed optimizer consuming wrong token IDs.

## Fix and results

`RVLGRPOHooks.rollout` now requires the backend's returned generation `prompt_id` and `prompt` to equal the dispatched task's ID and prompt before embedding the raw generation in an `Attempt`. The frozen test passed 3/3 after the change. The combined rollout-binding, RVL hook, RVL group-boundary, and engine task-binding tests passed 15 tests.

This is an E0 adapter-contract result using a deterministic fake backend that violates the requested identity. It does not establish that a real RVL backend has returned mismatched identity fields, run a trainer step with mismatched token IDs, establish verifier correctness, prove all metadata provenance, or show task-success or capability improvement. The test also does not cover token IDs that disagree despite matching prompt ID and prompt text. It does not enforce policy ID/version equality, which would conflict with allowed lag semantics.

A read-only implementation review found the check consistent with the generation API shape; this does not substitute for a real-backend integration run or outside human review.

## Decision

**Keep the fix.** It closes a concrete mismatch between the prompt an oracle evaluates and the token-exact generation an RL trainer consumes. Do not treat the repeated task-binding pattern as a new method or claim of policy improvement. The next priority remains independent reproduction or a real-model task-success update only after the existing no-cost feasibility gates pass.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_rvl_grpo_rollout_task_binding.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_rvl_grpo_rollout_task_binding.py tests/test_rvl_grpo_hooks.py tests/test_rvl_grpo_group_boundary.py tests/test_rollout_task_binding.py
```
