# RVL rollback mode restoration: shared-module follow-up

## Finding

The merged rollback fix in RVL `27ebf7fe239d97504eeb960bf1407de220584dac` restores modes by iterating `named_modules()` and calling `module.train(saved_mode)`. PyTorch deduplicates shared module objects in `named_modules()`, while `Module.train()` recursively changes all descendants. If a shared child is restored through its first parent and a later aliased parent has another saved mode, the later parent's recursive call overwrites the child's restored mode.

## Minimal reproduction

The regression uses the actual tiny GPT-2 fixture and `HFCausalLMGRPOTrainer` methods. It seeds AdamW state, registers the existing embedding Dropout beneath a second parent, then sets a valid mixed state: model in eval mode, alias parent in train mode, and the shared Dropout itself in eval mode. It snapshots that state, executes a real trainer optimizer step, injects an exception immediately after the parameter mutation, and restores. An independent copy of model tensors, optimizer state, and CPU RNG confirms the rollback, while the regression checks every module flag and exact next-forward logits.

At the merged upstream revision, the regression fails only on per-module modes and the subsequent Dropout forward. Model tensors, optimizer state, and CPU RNG are restored. The shared Dropout is set to `training=True` instead of `False`: it is enumerated beneath `transformer` once, then restoring the later alias parent with `.train(True)` recursively changes the same object.

## Minimal correction and evidence

The follow-up changes the final restoration loop to assign each already-validated `module.training` boolean directly. This avoids recursive writes after the snapshot's per-object values have been applied. It leaves state-dict, optimizer, RNG, topology validation, and legacy-snapshot behavior unchanged.

Pinned base: RVL `27ebf7fe239d97504eeb960bf1407de220584dac`. Runtime: Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, CPU.

| Check | Merged base | Follow-up patch |
| --- | --- | --- |
| Shared-child mode regression | Fails exact mode equality | Passes |
| Full `tests.test_mini_lab_torch` module | Not run on base for this follow-up | **14/14 passed** |
| Clean-clone reproduction | Test-only patch fails as expected | Follow-up patch passes |

Reproduce from VARE with:

```bash
PYTHON=python3.12 bash scripts/reproduce_rvl_shared_module_mode_followup_v1.sh results/rvl-module-mode-shared-alias-v1/replay-YYYYMMDD
```

The script checks out the pinned merged revision in separate baseline and patched clones, records raw logs, requires the regression to fail on base, and runs all 14 trainer tests with the patch. The retained clean-clone run is [`clean-clone-replay-20261010c`](../results/rvl-module-mode-shared-alias-v1/clean-clone-replay-20261010c/). The final patch is [`contributions/rvl-module-mode-shared-alias-followup.patch`](../contributions/rvl-module-mode-shared-alias-followup.patch); its SHA-256 is recorded in the [replay manifest](../results/rvl-module-mode-shared-alias-v1/manifest.json). Raw outputs are in [`results/rvl-module-mode-shared-alias-v1/`](../results/rvl-module-mode-shared-alias-v1/).

## Proposed follow-up PR description

**Title:** Preserve rollback modes for shared child modules

`restore_training_state()` currently calls `Module.train(saved_mode)` for every entry from `named_modules()`. `named_modules()` deduplicates shared module objects, but `Module.train()` recursively changes descendants. Restoring a later alias parent can therefore overwrite a shared child's saved mode.

This follow-up restores the already-validated `training` flags directly on each module object. A CPU regression uses RVL's real tiny GPT-2 trainer, performs an AdamW update, injects an exception after mutation, and checks model tensors, optimizer state, CPU RNG, per-module flags, and exact subsequent Dropout logits. It fails on merged RVL `27ebf7f` on the mode and forward checks; with the patch, all 14 tests in `tests.test_mini_lab_torch` pass. The clean-clone reproducer and raw logs are linked in the VARE report.

This covers a deliberately constructed shared-module graph. It does not establish that the default RVL Qwen configuration uses shared module objects, that a production run was affected, or that model quality changes. CUDA and distributed RNG restoration and custom `train()` side effects are outside this test.

## Scope and decision

This establishes a narrow rollback correctness failure for valid shared-module graphs and a local correction against the merged RVL revision. It does not show that RVL's default Qwen path shares module objects, that a production run encountered the issue, or that model quality changes. The CPU fixture does not test CUDA RNG, distributed rollback, or custom `Module.train()` side effects; the patch restores the recorded PyTorch `training` flags.

**Decision: retain as a follow-up patch for independent maintainer review.** No issue, PR, or external message has been sent for this follow-up. The merged PR #90 remains an upstream contribution, but its mode-restoration guarantee needs this edge case before it can be described as exact for shared-module graphs.
