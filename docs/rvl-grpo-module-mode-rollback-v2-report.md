# RVL GRPO rollback of PyTorch module modes v2

## Finding

The previous mid-step fault test checked model tensors, optimizer state, CPU RNG, policy identity, candidate cleanup, and the next rollout. It did not inspect `torch.nn.Module.training` flags. In the pinned RVL trainer, `train_step()` calls `model.train()` before doing the optimizer update, while `snapshot_training_state()` and `restore_training_state()` preserve tensors, optimizer state, and RNG only. When the real optimizer completed and the test injected an exception before `train_step()` returned, VARE restored the weights but left the model in training mode.

The mode change matters to rollback because dropout and other train/eval-sensitive modules can change subsequent inference and evaluation behavior. The minimal 3,696-parameter fixture uses zero dropout, so this experiment demonstrates state leakage, not a measured quality effect.

## Frozen reproduction

Protocol v2 was frozen in commit [`eeae165`](https://github.com/mitukx/VARE/commit/eeae165) against RVL commit [`c7e646b`](https://github.com/mitukx/Recursive-Verification-Lag/commit/c7e646b043cb56e5ea3c2623bb8a61e065451f72). The run used Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, one CPU thread, no pretrained model, and no dataset.

The validator ran the actual pinned `HFCausalLMGRPOTrainer.train_step` and real `optimizer.step`; the injection raised only after both model parameters and optimizer state changed. Its independent checks compare every named module's `training` flag, in addition to the v1 rollback checks.

| State | Before fix | After fix |
| --- | ---: | ---: |
| Model tensors restored | pass | pass |
| Optimizer restored | pass | pass |
| CPU RNG restored | pass | pass |
| Policy/candidate/counter state restored | pass | pass |
| All module modes restored | **fail** | pass |
| All frozen checks | **12/14** | **14/14** |
| Wall time | 3.026 s | 3.006 s |

The baseline output is retained at [`baseline-da5d457.json`](../results/rvl-grpo-midstep-fault-v2/baseline-da5d457.json); the initial patched output is at [`patched-da5d457-plus-adapter.json`](../results/rvl-grpo-midstep-fault-v2/patched-da5d457-plus-adapter.json), and a committed-code rerun is at [`patched-167e9d1.json`](../results/rvl-grpo-midstep-fault-v2/patched-167e9d1.json). A fresh local clone at the patched commit also passed; its output is [`clean-clone-65bdf.json`](../results/rvl-grpo-midstep-fault-v2/clean-clone-65bdf.json). The baseline passed the previous 12 checks, but failed the two new mode checks. The clone check is still same-host reproduction, not independent outside review or evidence of production incidence.

## Change

`RVLGRPOHooks` now snapshots the trainer state together with each named PyTorch module's mode. Restore first reloads the trainer state, then reapplies modes in `named_modules()` preorder so nested modules can retain distinct flags. Trainers without a PyTorch-style `named_modules()` interface retain the existing snapshot contract.

## Reproduction

Use the runtime and package setup in [the v1 report](rvl-grpo-midstep-fault-report.md#clean-checkout-reproduction). The first checkout reproduces the two mode failures; the second passes all 14 checks:

```bash
# Expected failure: the frozen validator detects the original rollback gap.
git checkout eeae165
python scripts/validate_rvl_grpo_midstep_fault_v2.py \
  --rvl-source /tmp/Recursive-Verification-Lag/src/rvl_systems

# Patched implementation: every frozen check passes.
git checkout 167e9d1ca3c2041a2cf5b5a6a924e364de18027d
python scripts/validate_rvl_grpo_midstep_fault_v2.py \
  --rvl-source /tmp/Recursive-Verification-Lag/src/rvl_systems
```

The validator verifies upstream file hashes and runtime versions. Run it on the patched VARE revision to obtain 14/14 checks. The retained before/after outputs are author-produced on one host; there has not yet been an outside reproduction or RVL maintainer review.

Focused VARE regressions passed: `tests/test_rvl_grpo_hooks.py`, `tests/test_rvl_evaluation_promotion_e2e.py`, and `tests/test_rvl_grpo_group_boundary.py` (9 passed). The full suite completed with four failures in the previously recorded `smoke-stable-logsumexp` environment tests; these failures predate this change and are tracked in [`current-gaps.md`](current-gaps.md).

## Scope and next external check

This verifies one exception boundary and one pinned RVL revision. It does not cover optimizer-kernel interruption, process loss, CUDA/distributed state, arbitrary mutable module buffers, mixed-mode modules in this fixture, pretrained-model quality, or task success. The useful external path is a reviewer rerunning the frozen command and checking the small adapter change. No external issue, pull request, or maintainer contact has been made.
