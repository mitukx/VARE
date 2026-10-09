# RVL trainer rollback: per-module mode restoration (v2 intermediate record)

This v2 protocol and its clean-clone output are preserved as an earlier review iteration. The final patch and maintainer review are in [v3](rvl-module-mode-snapshot-final-review-v3.md); do not submit the v2 patch.

## Finding and scope

RVL `HFCausalLMGRPOTrainer.snapshot_training_state()` stores model tensors, optimizer state, and RNG, but omits each `nn.Module.training` flag. `train_step()` calls `model.train()`. If a caller snapshots a model in eval or mixed mode, performs an update, and rejects it, `restore_training_state()` restores the existing fields but leaves the model in training mode.

This is a snapshot API completeness defect, not an observed failure in RVL's standard Qwen run. The current `HFLocalBackend` initializes the model in train mode, generation restores that mode, and dropout is disabled by default. The fixture below validates a non-default but valid caller state. It does not establish a default-run behavior change, production incidence, pretrained-model impact, or capability gain.

## Minimal fix

The patch adds one name-to-boolean map to the snapshot. Restore validates that the saved names match the current module topology and that every value is a real `bool` before changing model state. It restores flags in `named_modules()` preorder after model, optimizer, and RNG state; parent `Module.train()` calls recursively set children, so later child entries restore nested mixed modes. Snapshots without the new key continue through the previous restore path.

The combined patch changes only `src/rvl_systems/hf_trainer.py` and `tests/test_mini_lab_torch.py`. V1 is retained as frozen history; the final maintainer-reviewed test patch is v2.

## Validation

Protocol: [frozen v2 lock](../protocols/rvl_module_mode_snapshot_upstream_v2.lock.json). Baseline and patched tests were run from a clean clone of the exact upstream current `main` HEAD, `c7e646b043cb56e5ea3c2623bb8a61e065451f72` (rechecked 2026-10-10). Runtime: Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, CPU; no model download or paid compute.

The mixed-mode test first performs a real update to seed nonempty AdamW state. It then sets the root to eval and a nested MLP subtree to train, snapshots model/optimizer/RNG/module modes, and wraps the real optimizer step so it mutates parameters and optimizer state, consumes RNG, then raises. An independent pre-state map from `named_modules()` and snapshots of the existing state provide the restore oracle.

| Run | Result |
| --- | --- |
| Test-only v2 patch on unmodified base | Expected failure only on exact module-mode equality; legacy compatibility control passed. Assertions before the final mode comparison confirmed the optimizer and model had mutated and that model/optimizer/RNG were restored. |
| Combined v2 patch on base | Full `tests.test_mini_lab_torch`: **13/13 passed** (0.370 s). Includes nested mixed modes, exception after optimizer mutation, exact model/optimizer/CPU-RNG restoration, and legacy snapshots missing `module_modes`. |
| Patch hygiene | Applies cleanly to pinned base; `git diff --check` passed. |
| Reproducibility | One-command clean-clone script and full baseline/fixed output are retained at [`clean-clone-replay-20261010.log`](../results/rvl-module-mode-upstream-v2/clean-clone-replay-20261010.log). |

The test uses a randomly initialized 3,696-parameter GPT-2 fixture with zero dropout. It proves exact state restoration on CPU; CUDA RNG restoration follows unchanged code and was not exercised. It does not measure output quality or establish that a live RVL run has encountered this state mismatch. The source patch has not received independent external review or maintainer review.

## Proposed upstream PR text

**Title:** Restore per-module train/eval modes from GRPO trainer snapshots

**Body:**

`HFCausalLMGRPOTrainer.snapshot_training_state()` currently saves model tensors, optimizer state, and RNG but omits `nn.Module.training` flags. Since `train_step()` calls `model.train()`, restoring a rejected update from an eval or mixed-mode snapshot restores the existing state but leaves every module in training mode.

This patch records modes by `named_modules()` key and restores them after model, optimizer, and RNG state. It validates the saved module topology and boolean values before mutating the model. Restoration uses named-module preorder so nested modules can retain mixed modes. Older snapshots without `module_modes` remain supported.

The regression runs a real GRPO optimizer update, injects an exception immediately after the optimizer has mutated state, then checks exact model tensors, populated optimizer state, CPU RNG, and every module mode. A legacy-snapshot control checks that existing model/optimizer/RNG restoration still works when the new field is absent. On the pinned base, the new mixed-mode regression fails and the legacy control passes; with the patch, all 13 tests in `tests.test_mini_lab_torch` pass on CPU.

This is a snapshot API correctness fix. The fixture is a randomly initialized tiny GPT-2 with dropout disabled; no production incidence, default Qwen path behavior change, pretrained-model effect, or task-success improvement is claimed. CUDA RNG and independent external reproduction were not tested.

Reproduce with Python 3.11+ and installed `torch`/`transformers`:

```bash
PYTHON=python3.12 bash scripts/reproduce_rvl_module_mode_snapshot_v2.sh
```

**Recommendation: GO for submission as a narrow rollback state-completeness fix.** Its external impact is bounded to callers that snapshot non-training or mixed module modes. The tests establish local correctness only; maintainer acceptance and independent outside review remain unknown. Do not submit until the user approves.
