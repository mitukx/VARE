# RVL evaluation task identity v1

## Finding

`RVLGRPOHooks.evaluate` stores task scores in a mapping keyed by `Task.id`. If two evaluation rows share an ID, the later score overwrites the earlier score. The adapter previously completed both generations but returned `n` and `primary` from the collapsed mapping, silently changing the evaluation population and its result. It also failed to check that a backend generation returned for the requested task actually carried that task's ID and prompt, despite applying this check on the rollout path.

The pinned pre-fix adapter reproduced the duplicate-ID failure with two distinct prompts sharing one ID. It called the backend twice, then reported `n=1`, primary `1.0`, and a single retained score. A row-wise oracle over the actual per-prompt outputs gives `n=2` and primary `0.5`. Duplicate IDs can therefore hide a regression and overstate held-out performance. In a second fixture, a request for `(task-a, prompt-a)` received a generation tagged `(task-b, prompt-b)`; the old adapter scored it as task-a and returned a report. This violates the task-to-generation provenance contract and can apply the wrong task label. Both fixtures are synthetic; they establish adapter failure modes, not that an existing run used them.

## Correction

The adapter now rejects duplicate evaluation IDs at construction, before model generation or scoring. During evaluation it also checks the returned generation's ID and prompt before calling the grader. A mismatched row raises an explicit error and cannot receive a score. Unique IDs and correctly bound generations remain fully supported; a two-row control reports `n=2`, primary `0.5`, and both task scores as expected.

## Reproduction

```bash
.venv/bin/python reproducers/rvl_grpo_eval_identity_v1.py --output results/rvl-grpo-eval-identity-v1/reproduction.json
.venv/bin/python -m pytest -q tests/test_rvl_grpo_eval_identity.py tests/test_rvl_grpo_hooks.py tests/test_rvl_grpo_rollout_task_binding.py
```

The reproducer executes the actual pre-fix adapter from commit `31615b6` and the current adapter against the same CPU backends. The duplicate-ID comparison uses an independent row-wise arithmetic oracle; the response-binding case checks the exact requested/returned identity tuple and proves the grader is never called on mismatch. The regressions were added first and failed before the checks were implemented. See [raw reproduction](../results/rvl-grpo-eval-identity-v1/reproduction.json) and [tests](../tests/test_rvl_grpo_eval_identity.py).

## Limits

- No pretrained model or optimizer was used.
- The finding does not show that duplicate task IDs occur in VARE's retained production runs.
- This corrects task identity and score retention at the RVL adapter boundary; it does not establish better model behavior, capability gain, or external adoption.
