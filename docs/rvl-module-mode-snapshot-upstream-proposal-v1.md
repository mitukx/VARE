# RVL proposal: preserve module modes in trainer snapshots

## Finding

At RVL commit [`c7e646b043cb56e5ea3c2623bb8a61e065451f72`](https://github.com/mitukx/Recursive-Verification-Lag/commit/c7e646b043cb56e5ea3c2623bb8a61e065451f72), `HFCausalLMGRPOTrainer.snapshot_training_state()` stores model tensors, optimizer, and RNG, but omits each PyTorch module's `training` flag. `train_step()` calls `model.train()`. Restoring a rejected candidate therefore restores its weights and optimizer while leaving the model in training mode. The same snapshot/restore methods are used by RVL's `run_qwen_rlvr_experiment.py` transactional-promotion path, so this is useful beyond the VARE adapter.

The issue is consequential for models with dropout or other mode-sensitive modules: subsequent rollout/evaluation can use different behavior after a rejected candidate. The frozen fixture uses zero dropout; it proves state mismatch, not a measured output-quality difference or production incidence.

## Minimal patch

The [combined patch](../contributions/rvl-module-mode-snapshot-v1.patch) adds a name-to-boolean module-mode map to snapshots and restores it in `named_modules()` preorder after loading weights/optimizer/RNG. Prevalidation rejects changed module topology or malformed mode values before mutating the model. `state.get("module_modes")` preserves compatibility with legacy snapshots. It also adds a regression that starts from mixed parent/child modes, performs a real tiny GRPO update, restores the snapshot, and compares every module flag.

Separate [test-only](../contributions/rvl-module-mode-test-only-v1.patch) and [fix-only](../contributions/rvl-module-mode-fix-only-v1.patch) patches are retained. Hashes and frozen acceptance criteria are in [protocol v1](../protocols/rvl_module_mode_snapshot_upstream_v1.lock.json).

## Validation

The pinned base was cloned locally at the exact upstream commit. On the base, the new mixed-mode regression failed as predicted; the legacy-snapshot control passed. After applying the combined patch, `git diff --check` passed and the complete optional PyTorch/Transformers test module passed (13 tests, 0.346 s):

| Run | Outcome |
| --- | --- |
| Test-only patch on base | 1 expected failure (`test_hf_trainer_transaction_restores_mixed_module_modes`), 1 compatibility control passed |
| Combined patch on base | Patch applies cleanly; `tests.test_mini_lab_torch`: 13/13 passed |
| Dependencies / hardware | Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, CPU; no model download or paid compute |
| Fresh-clone replay (2026-10-10) | New driver cloned the pinned RVL commit into a disposable clean tree: baseline regression failed and legacy control passed; combined patch passed all 13 tests |
| Runtime negative control | Python 3.9.6 ran 9 tests but errored in 4 unrelated tests because `asyncio.TaskGroup` is unavailable; this is an environment incompatibility, not a patch failure. The full suite requires Python 3.11+. Raw output is retained beside the clean-clone summary. |

Raw baseline/fixed output and the clean-clone summary are retained in [`results/rvl-grpo-module-mode-upstream-proposal/`](../results/rvl-grpo-module-mode-upstream-proposal/). The separate VARE clean-tree integration replay remains in [`results/rvl-grpo-midstep-fault-v2/review-replay-20261010/`](../results/rvl-grpo-midstep-fault-v2/review-replay-20261010/).

For one-command clean-clone reproduction, install the optional test dependencies (`torch` and `transformers`) in advance and use Python 3.11 or newer (`asyncio.TaskGroup` is used elsewhere in the test module). The driver checks patch hashes against the frozen protocol, clones the exact upstream revision in a disposable directory, retains command logs and hashes, and never resets or modifies the caller's checkout:

```bash
python scripts/reproduce_rvl_module_mode_snapshot_v1.py \
  --python /path/to/python3.12 \
  --output-dir /tmp/rvl-module-mode-replay
```

The regression uses a randomly initialized 3,696-parameter GPT-2 fixture and requires no model or dataset download. The driver does not install dependencies; its output is same-host reproduction, not outside human review.

## External-validation status and limits

This is a source-level fix with an actual trainer update/restore regression, exact-base reproduction, and an upstream-applicable patch. The VARE-to-RVL injected optimizer-fault integration also changes from 12/14 baseline checks to 14/14 after VARE's adapter fix. These are same-author, same-host results. The patch has **not** been submitted; RVL maintainer review and an independent outside reproduction are outstanding. No real-model quality, capability, or production-frequency claim is made.
