# RVL GRPO training-generation task identity v1

## Finding

`RVLGRPOHooks._to_verified_generation()` converted retained `rvl_generation`
metadata into an RVL `Generation` without checking that its `prompt_id` and
`prompt` matched the `Experience` task. The later identity check in
`_group_relative_advantages()` applied only when VARE rollout-group metadata
was present. An ungrouped, structurally valid experience could therefore pass
a generation for a different task into `trainer.train_step()`.

The frozen source-bound regression uses the task `expected-task` with the
prompt `expected prompt` and two malformed records: one changes only `prompt_id`, the other
changes only `prompt`. On main `96b3cb86046ccde579caa1c65ccde9f8a2c76e65`,
both cases reached the deterministic fake trainer (`train_step_calls=1`) and
the test failed because the adapter did not reject them. Candidate training
then rolled the fake optimizer counter back, so the first exploratory fixture
could not use that counter to show the call. The frozen protocol's append-only
run-1 erratum records this observation; the corrected regression observes a
non-snapshotted call counter. A second run used a clean base worktree with only
the regression test copied from the feature branch and reproduced both
baseline failures again.

## Change

The adapter now checks the retained RVL `prompt_id` and exact prompt text
against `Experience.attempt.task` before constructing the `Generation`. A
mismatch raises `ValueError` before the trainer is called. This makes task
binding unconditional across grouped and ungrouped training batches.

## Reproduction and validation

Run from the repository root. The plugin-autoload setting avoids an unrelated
global pytest plugin conflict in the author's Python 3.12 environment.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short \
  tests/test_rvl_grpo_hooks.py::test_rvl_trainer_boundary_rejects_un_grouped_generation_for_other_task
```

The frozen regression failed on the baseline in both cases: the mismatched
generation reached `train_step`. After the fix, the full RVL adapter-focused
suite passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short \
  tests/test_rvl_grpo_eval_identity.py \
  tests/test_rvl_grpo_group_boundary.py \
  tests/test_rvl_grpo_group_boundary_calibration.py \
  tests/test_rvl_grpo_hooks.py \
  tests/test_rvl_grpo_inference_cancellation.py \
  tests/test_rvl_grpo_rollout_task_binding.py \
  tests/test_rvl_grpo_shared_module_mode.py \
  tests/test_rvl_grpo_torch_state.py \
  tests/test_rvl_grpo_verifier_trainer_prompt_alignment.py \
  tests/test_rvl_integration.py \
  tests/test_rvl_evaluation_promotion_e2e.py
```

Result: **79 passed**. `ruff check src/vare/integrations/rvl_grpo.py
tests/test_rvl_grpo_hooks.py` and `python -m py_compile` on those files passed.
Raw baseline and fixed-suite logs are under
[`results/rvl-grpo-training-generation-task-binding-v1/`](../results/rvl-grpo-training-generation-task-binding-v1/);
their hashes are listed in that directory's `SHA256SUMS`.

## Scope and decision

This is a VARE adapter-boundary integrity defect reproduced with a
deterministic fake trainer. It does not establish that any historical run
contained mismatched records, and the fake trainer is not a real policy
optimizer. No model inference, training, coding repair, held-out evaluation,
reward-quality measurement, or capability improvement was performed. The
task inventory remains at zero fresh Development candidates; no new model
attempt is authorized by this finding.

Decision: fix the adapter boundary, retain the regression, and integrate only
after exact-head CI and review. The next model-learning gate remains a
qualified fresh Development cohort and a separately frozen protocol.
