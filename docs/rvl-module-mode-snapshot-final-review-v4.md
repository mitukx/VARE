# RVL GRPO snapshot mode rollback: submission review v4

## Finding

RVL `main` remains at `c7e646b043cb56e5ea3c2623bb8a61e065451f72`. In that revision, `HFCausalLMGRPOTrainer.train_step()` calls `model.train()`, while `snapshot_training_state()` / `restore_training_state()` preserve parameters, optimizer state, and RNG but not the `training` flag on each `nn.Module`. A rejected update from eval or mixed mode therefore restores the stored tensors but can leave later forwards in a different mode.

This is a snapshot API completeness defect. The default RVL Qwen path starts in train mode and disables dropout; these tests do not show that the standard path has changed behavior or that a production run encountered the defect.

## Minimal fix

The proposed source change stores a name-to-boolean map for `named_modules()`. Restore validates the module names and boolean values before mutating state, restores model/optimizer/RNG as before, then calls `train(flag)` in named-module preorder. Preorder is required because `Module.train()` recursively changes descendants; later child entries restore nested mixed modes. A snapshot that lacks the new field remains accepted as a legacy snapshot.

The change touches only `src/rvl_systems/hf_trainer.py` and `tests/test_mini_lab_torch.py`. It does not alter the update objective, rollout behavior, or model weights beyond the trainer's ordinary optimizer step.

## Failure-path regression

The deterministic CPU regression exercises the actual trainer and a real AdamW update:

1. Seed AdamW state with an initial trainer update.
2. Set the GPT-2 fixture to mixed module modes and leave a real embedding `Dropout(p=0.5)` active; configure the trainer not to disable it.
3. Save independent copies of every model tensor/buffer, optimizer state, CPU RNG, module flag, and a reference forward output under the saved RNG.
4. Run `train_step()`. Its real optimizer step mutates model and optimizer state; the injected wrapper consumes RNG and raises immediately afterward.
5. Restore the snapshot and compare every saved item, then run the same forward with the restored RNG and compare exact logits.

On the unpatched pinned baseline, the regression fails only on `per-module modes` and `subsequent dropout forward`; parameter/buffer, optimizer, and CPU RNG checks pass. With the proposed fix, those checks pass, the exact forward logits match, and the legacy snapshot control passes.

## Reproduction evidence

Pinned upstream revision: `c7e646b043cb56e5ea3c2623bb8a61e065451f72` (also confirmed as RVL `main` on 2026-10-10). Runtime: Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, CPU. No pretrained model, dataset, GPU, or paid service is used.

| Clean-clone check | Result |
| --- | --- |
| Test-only patch on upstream baseline | Expected failure: one test, exactly the two mode/forward checks fail |
| Full `tests.test_mini_lab_torch` with patch | **13/13 passed** |
| Patch application and `git diff --check` | pass |
| Legacy snapshot missing `module_modes` | pass |

Raw baseline and patched logs, the test-only diff, and the runtime summary are retained under [`results/rvl-module-mode-upstream-v4/clean-clone-replay-20261010/`](../results/rvl-module-mode-upstream-v4/clean-clone-replay-20261010/). The exact proposed patch is [`contributions/rvl-module-mode-snapshot-v4.patch`](../contributions/rvl-module-mode-snapshot-v4.patch). Reproduce both clean-clone paths with:

```bash
PYTHON=python3.12 bash scripts/reproduce_rvl_module_mode_snapshot_v4.sh results/rvl-module-mode-upstream-v4/replay-YYYYMMDD
```

The script requires an empty output directory and records both raw test logs. Patch SHA-256: `3af3e82812e03a3a742f25f7f5d6ac5a7c3fe1dbe89052302818c8b95e70935d`.

## Proposed upstream PR

**Title:** Restore module modes from GRPO trainer snapshots

**Body:**

`HFCausalLMGRPOTrainer.snapshot_training_state()` saves model tensors, optimizer state, and RNG, but omits each module's `nn.Module.training` flag. `train_step()` calls `model.train()`, so restoring a rejected update from an eval or mixed-mode snapshot can leave later forwards in a different mode.

This patch snapshots modes by `named_modules()` key and reapplies them after the existing model, optimizer, and RNG restore. It validates topology and boolean values before mutation, restores in preorder so nested modules can retain mixed modes, and continues to accept legacy snapshots without `module_modes`.

The CPU regression runs the real trainer and AdamW update, injects an exception after optimizer mutation, and checks model state, optimizer state, CPU RNG, every module mode, and exact subsequent Dropout forward logits. On upstream `c7e646b`, the test-only patch fails on mode equality and forward equality; with the fix, all 13 tests in `tests.test_mini_lab_torch` pass. The baseline, patched logs, and one-command clean-clone reproducer are retained in VARE.

The fixture is a randomly initialized tiny GPT-2 and tests a deliberately mixed-mode caller state. This establishes rollback correctness for that state, not production incidence, behavior change in RVL's default Qwen path, pretrained-model quality, or task-success improvement. CUDA RNG and other accelerator/distributed rollback paths remain untested.

## Independent review and recommendation

**GO for upstream review after approval.** The patch has a concrete failure case, an observable behavior difference, an independent state oracle, legacy compatibility, and a clean-clone baseline/fix comparison. A fresh reviewer should verify the module traversal semantics and run the reproduction command. Independent human review requires an RVL maintainer or another contributor to inspect the eventual PR and leave substantive review feedback; author-run tests, AI review, CI, and self-merge do not count.

## Status update — 2026-10-10

After this review was written, the v4 patch and Dropout regression were submitted in RVL PR [#90](https://github.com/mitukx/Recursive-Verification-Lag/pull/90). The PR was merged by the repository owner as [`27ebf7f`](https://github.com/mitukx/Recursive-Verification-Lag/commit/27ebf7fe239d97504eeb960bf1407de220584dac), and all seven required CI checks passed. The recorded review was automated Codex review; no independent human review was recorded. This establishes upstream integration, not independent review, production incidence, default-path behavior change, or model-quality improvement. The pinned baseline/fix and clean-clone evidence above remain unchanged.
