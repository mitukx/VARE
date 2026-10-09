# RVL module-mode rollback patch: final maintainer review

## Defect and root cause

At current RVL `main` HEAD `c7e646b043cb56e5ea3c2623bb8a61e065451f72` (confirmed 2026-10-10), [`snapshot_training_state()` / `restore_training_state()`](https://github.com/mitukx/Recursive-Verification-Lag/blob/c7e646b043cb56e5ea3c2623bb8a61e065451f72/src/rvl_systems/hf_trainer.py#L120-L147) preserve model tensors, optimizer state, and RNG but omit each module's `training` flag. [`train_step()` calls `model.train()`](https://github.com/mitukx/Recursive-Verification-Lag/blob/c7e646b043cb56e5ea3c2623bb8a61e065451f72/src/rvl_systems/hf_trainer.py#L149-L164). A caller that snapshots eval or mixed modes, then rejects an update, therefore receives the original weights and optimizer with changed module modes.

Scope: this is a snapshot API completeness bug. RVL's standard Qwen path begins and remains in train mode and disables dropout by default; this review does not demonstrate a behavior change in that default path or production incidence.

## Fix and regression

The source change adds a per-module name-to-boolean map, validates topology and value types before any state mutation, then restores modes in `named_modules()` preorder after model, optimizer, and RNG. Preorder matters because `Module.train()` recursively updates descendants. Missing `module_modes` remains valid for legacy snapshots.

The v3 regression uses the actual trainer and optimizer. It first creates populated AdamW state, sets the root to eval and a nested MLP subtree to train, and independently copies model tensors, optimizer state, RNG, and named-module flags. The optimizer wrapper performs the real update, confirms state mutation, consumes RNG, then raises. After rollback, every captured tensor/state value, RNG byte, and module flag must match. A legacy-shaped snapshot separately checks model, populated optimizer, and RNG restoration.

## Verification

Frozen protocol: [v3 lock](../protocols/rvl_module_mode_snapshot_upstream_v3.lock.json). The upstream remote HEAD and the pinned base both resolved to `c7e646b043cb56e5ea3c2623bb8a61e065451f72`. A fresh clone ran the test-only regression against base, then reset and applied the combined patch:

| Check | Result |
| --- | --- |
| Unpatched base | The new regression fails only at final per-module mode equality; model, populated optimizer, and RNG restoration checks pass. Legacy snapshot control passes. |
| Patched implementation | All 13 tests in `tests.test_mini_lab_torch` pass (0.474 s). |
| Patch hygiene | Applies to pinned base; `git diff --check` passes. |
| Runtime | Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, CPU; no model or dataset download. |
| Clean-clone output | [baseline and patched log](../results/rvl-module-mode-upstream-v3/clean-clone-replay-20261010.log). |

The source/test patch is [here](../contributions/rvl-module-mode-snapshot-v3.patch); replay it with:

```bash
PYTHON=python3.12 bash scripts/reproduce_rvl_module_mode_snapshot_v3.sh
```

Limitations: randomly initialized 3,696-parameter GPT-2 with zero dropout; CPU only; CUDA RNG branch is unchanged but untested; no pretrained-model or output-quality measurement; no independent outside reproduction or maintainer review. The report deliberately does not imply production impact or capability gain.

## Proposed PR description

**Title:** Restore per-module train/eval modes from GRPO trainer snapshots

**Body:**

`HFCausalLMGRPOTrainer.snapshot_training_state()` saves model tensors, optimizer state, and RNG, but omits per-module `nn.Module.training` flags. Because `train_step()` calls `model.train()`, restoring a rejected update from an eval or mixed-mode snapshot leaves the caller's model in training mode.

This patch records flags by `named_modules()` key and restores them after model, optimizer, and RNG state. It validates module topology and boolean values before mutation, restores in preorder so nested modules retain mixed modes, and continues to accept older snapshots without `module_modes`.

The regression performs a real optimizer mutation and then injects an exception. It checks exact restoration of model tensors, populated optimizer state, CPU RNG, and every named module flag against independent pre-snapshot copies. A legacy-format control checks the existing model/optimizer/RNG fields. On the pinned base, the new mixed-mode regression fails while the legacy control passes; after the patch, all 13 optional trainer tests pass on CPU.

This is a snapshot API correctness fix. Tests use a random tiny GPT-2 with dropout disabled. They do not establish production incidence, impact on RVL's default Qwen path, pretrained-model behavior, task-success gain, or independent external validation.

## Recommendation

**GO for submission as a narrow snapshot API correctness fix.** The correctness evidence is strong for exact rollback under the tested non-default mixed-mode state. The practical scope is limited; default-path impact is unproven. This patch has not been submitted, and independent external review remains outstanding. Submission requires the user's approval.
