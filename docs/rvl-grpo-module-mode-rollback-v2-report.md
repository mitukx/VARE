# RVL GRPO rollback of PyTorch module modes v2

## Finding

The previous mid-step fault test checked model tensors, optimizer state, CPU RNG, policy identity, candidate cleanup, and the next rollout. It did not inspect `torch.nn.Module.training` flags. In the pinned RVL trainer, `train_step()` calls `model.train()` before doing the optimizer update, while `snapshot_training_state()` and `restore_training_state()` preserve tensors, optimizer state, and RNG only. When the real optimizer completed and the test injected an exception before `train_step()` returned, VARE restored the weights but left the model in training mode.

The test begins with the model in eval mode, so it demonstrates the adapter failing to preserve a deliberately selected mode through an actual optimizer-fault rollback. The minimal 3,696-parameter fixture uses zero dropout, so this experiment demonstrates state leakage, not a measured output effect. In RVL's standard `HFLocalBackend` path, model loading explicitly enters train mode, generation restores that mode, and dropout is disabled by default; this test does not show that the existing standard Qwen transactional path changes behavior.

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

Use the runtime and package setup in [the v1 report](rvl-grpo-midstep-fault-report.md#clean-checkout-reproduction), check out this report's VARE revision, and check out the pinned RVL source revision. The one-command driver extracts baseline and patched VARE commits into temporary clean trees, runs the real validator in both, and requires the only baseline failures to be the two module-mode checks:

```bash
python scripts/reproduce_rvl_grpo_midstep_fault_v2.py \
  --rvl-source /tmp/Recursive-Verification-Lag/src/rvl_systems \
  --output-dir results/rvl-grpo-midstep-fault-v2/review-replay
```

The driver refuses a different RVL commit and a non-empty output directory. It does not alter the current checkout. On 2026-10-10 it reproduced 12/14 baseline checks (only `module_modes_restored` and `root_mode_restored` fail) and 14/14 patched checks in 7.418 seconds of summed validator time. The raw baseline/fixed JSON and stderr files plus summary are retained in [`review-replay-20261010`](../results/rvl-grpo-midstep-fault-v2/review-replay-20261010/). This remains an author-run, same-host clean-tree reproduction; outside reproduction and RVL maintainer review have not occurred.

Focused VARE regressions passed: `tests/test_rvl_grpo_hooks.py`, `tests/test_rvl_evaluation_promotion_e2e.py`, and `tests/test_rvl_grpo_group_boundary.py` (9 passed). The full suite completed with four failures in the previously recorded `smoke-stable-logsumexp` environment tests; these failures predate this change and are tracked in [`current-gaps.md`](current-gaps.md).

## Scope and next external check

This verifies one exception boundary and one pinned RVL revision. It does not cover optimizer-kernel interruption, process loss, CUDA/distributed state, arbitrary mutable module buffers, nonzero-dropout output divergence, the standard path's mode behavior, pretrained-model quality, or task success. The adapter fix protects callers that deliberately use eval or mixed module modes. No external issue, pull request, or maintainer contact has been made.
