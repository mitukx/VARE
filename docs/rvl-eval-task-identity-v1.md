# RVL evaluation task identity v1

## Finding

`RVLGRPOHooks.evaluate` stores task scores in a mapping keyed by `Task.id`. If two evaluation rows share an ID, the later score overwrites the earlier score. The adapter previously completed both generations but returned `n` and `primary` from the collapsed mapping, silently changing the evaluation population and its result.

The pinned pre-fix adapter reproduced this with two distinct prompts sharing one ID. It called the backend twice, then reported `n=1`, primary `1.0`, and a single retained score. A row-wise oracle over the actual per-prompt outputs gives `n=2` and primary `0.5`. Duplicate IDs can therefore hide a regression and overstate held-out performance. The fixture is synthetic; this establishes an adapter defect, not that an existing run used duplicate IDs.

## Correction

The adapter now rejects duplicate evaluation IDs at construction, before model generation or scoring. This preserves unique keys as an explicit precondition for paired evaluation evidence. Unique IDs remain fully supported; a two-row control reports `n=2`, primary `0.5`, and both task scores as expected.

## Reproduction

```bash
.venv/bin/python reproducers/rvl_grpo_eval_identity_v1.py --output results/rvl-grpo-eval-identity-v1/reproduction.json
.venv/bin/python -m pytest -q tests/test_rvl_grpo_eval_identity.py tests/test_rvl_grpo_hooks.py tests/test_rvl_grpo_rollout_task_binding.py
```

The reproducer executes the actual pre-fix adapter from commit `31615b6` and the current adapter against the same CPU backend, then compares with a row-wise arithmetic oracle. The regression was added first and failed because no `ValueError` was raised. The current adapter rejects the fixture with zero backend calls. See [raw reproduction](../results/rvl-grpo-eval-identity-v1/reproduction.json) and [tests](../tests/test_rvl_grpo_eval_identity.py).

## Limits

- No pretrained model or optimizer was used.
- The finding does not show that duplicate task IDs occur in VARE's retained production runs.
- This corrects task identity and score retention at the RVL adapter boundary; it does not establish better model behavior, capability gain, or external adoption.
