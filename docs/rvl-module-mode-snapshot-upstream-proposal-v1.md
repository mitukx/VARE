# RVL proposal: preserve module modes in trainer snapshots

## Finding

At RVL commit [`c7e646b043cb56e5ea3c2623bb8a61e065451f72`](https://github.com/mitukx/Recursive-Verification-Lag/commit/c7e646b043cb56e5ea3c2623bb8a61e065451f72), `HFCausalLMGRPOTrainer.snapshot_training_state()` stores model tensors, optimizer, and RNG, but omits each PyTorch module's `training` flag ([snapshot/restore](https://github.com/mitukx/Recursive-Verification-Lag/blob/c7e646b043cb56e5ea3c2623bb8a61e065451f72/src/rvl_systems/hf_trainer.py#L120-L147)). `train_step()` calls `model.train()` ([source](https://github.com/mitukx/Recursive-Verification-Lag/blob/c7e646b043cb56e5ea3c2623bb8a61e065451f72/src/rvl_systems/hf_trainer.py#L149-L164)). Restoring a rejected candidate therefore restores its weights and optimizer while leaving the model in training mode. The same snapshot/restore methods are used by RVL's transactional promotion path ([call sites](https://github.com/mitukx/Recursive-Verification-Lag/blob/c7e646b043cb56e5ea3c2623bb8a61e065451f72/src/run_qwen_rlvr_experiment.py#L218-L257)). On 2026-10-10, upstream's remote `HEAD` still resolved to this exact commit, so the omission is present on the current default branch. This is a proposed fix, not a reproduction of an existing upstream correction.

This is a real snapshot API omission, but source inspection narrows its impact in RVL's current default experiment path. `HFLocalBackend.ensure_loaded()` calls `model.train()`, generation temporarily enters eval mode and restores the previous mode, and `HFTTrainerConfig.disable_dropout` defaults to true. `run_qwen_rlvr_experiment.py` snapshots after that evaluation path, then `train_step()` calls `model.train()` again; therefore its usual all-training-mode snapshot is not observably changed by restore. The frozen mixed-mode regression deliberately exercises a valid but non-default caller state. It proves that `snapshot_training_state()` is not a complete state snapshot for such callers; it does not demonstrate a behavior change in the standard Qwen transactional run, nonzero-dropout impact, or production incidence.

## Why this is the selected external contribution

No reviewed candidate combines a new upstream fix with a demonstrated effect in the upstream project's standard run. The TRL #7249 reproductions have the clearest measured gradient effect, but reproduce an existing open upstream proposal. The evaluation-integrity fixes change VARE-owned code. The RVL patch is the only new, narrow upstream-applicable code change in this set, so it is retained as an API-completeness proposal; its effect on RVL's default Qwen transactional path is currently unproven and likely absent under that path's all-training-mode/default-no-dropout setup. Treat it as a bounded maintainer-review candidate, not as the strongest measured systems impact.

## Minimal patch

The [combined patch](../contributions/rvl-module-mode-snapshot-v1.patch) adds a name-to-boolean module-mode map to snapshots and restores it in `named_modules()` preorder after loading weights/optimizer/RNG. Prevalidation rejects changed module topology or malformed mode values before mutating the model. `state.get("module_modes")` preserves compatibility with legacy snapshots. It also adds a regression that starts from mixed parent/child modes, performs a real tiny GRPO update, restores the snapshot, and compares every module flag.

Separate [test-only](../contributions/rvl-module-mode-test-only-v1.patch) and [fix-only](../contributions/rvl-module-mode-fix-only-v1.patch) patches are retained. Hashes and frozen acceptance criteria are in [protocol v1](../protocols/rvl_module_mode_snapshot_upstream_v1.lock.json).

## Validation

The pinned base was cloned locally at the exact upstream commit. On the base, the new mixed-mode regression failed as predicted; the legacy-snapshot control passed. After applying the combined patch, `git diff --check` passed and the complete optional PyTorch/Transformers test module passed (13 tests, 0.346 s):

| Run | Outcome |
| --- | --- |
| Test-only patch on base | 1 expected failure (`test_hf_trainer_transaction_restores_mixed_module_modes`), 1 legacy-snapshot compatibility control passed; independent expected state is captured from `model.named_modules()` before training |
| Combined patch on base | Patch applies cleanly; `tests.test_mini_lab_torch`: 13/13 passed, including existing exact model tensor, optimizer, and CPU RNG restoration checks |
| Dependencies / hardware | Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, CPU; no model download or paid compute |
| Fresh-clone replay (2026-10-10) | A disposable clone of the pinned RVL commit reproduced the baseline regression failure and passing legacy control; combined patch passed all 13 tests |
| Minimal shell replay (2026-10-10) | Repeated from a fresh clone with the shorter reproducer: expected base failure/control pass, then 13/13 passed in 0.365 s; complete output retained in [`minimal-shell-replay-20261010.log`](../results/rvl-grpo-module-mode-upstream-proposal/minimal-shell-replay-20261010.log) |
| Runtime negative control | Python 3.9.6 ran 9 tests but errored in 4 unrelated tests because `asyncio.TaskGroup` is unavailable; this is an environment incompatibility, not a patch failure. The full suite requires Python 3.11+. Raw output is retained beside the clean-clone summary. |

Raw baseline/fixed output and the clean-clone summary are retained in [`results/rvl-grpo-module-mode-upstream-proposal/`](../results/rvl-grpo-module-mode-upstream-proposal/). The separate VARE clean-tree integration replay remains in [`results/rvl-grpo-midstep-fault-v2/review-replay-20261010/`](../results/rvl-grpo-midstep-fault-v2/review-replay-20261010/).

For one-command clean-clone reproduction, install `torch` and `transformers` in advance and use Python 3.11 or newer. The short shell script clones the exact revision into a temporary directory, confirms the expected baseline failure and legacy control, then applies the patch and runs the full optional test module. It does not modify the caller's checkout. To save a fresh run's output, append `2>&1 | tee /tmp/rvl-module-mode-replay.log` to the command:

```bash
PYTHON=python3.12 bash scripts/reproduce_rvl_module_mode_snapshot_v1.sh
```

The regression uses a randomly initialized 3,696-parameter GPT-2 fixture and requires no model or dataset download. The script does not install dependencies; the captured run is same-host reproduction, not outside human review. Patch hashes and acceptance criteria remain pinned in [protocol v1](../protocols/rvl_module_mode_snapshot_upstream_v1.lock.json); the historical clean-clone and Python 3.9 control logs remain in the results directory.

## External-validation status and limits

This is a source-level API-completeness fix with an actual trainer update/restore regression, exact-base reproduction, and an upstream-applicable patch. The VARE-to-RVL injected optimizer-fault integration also changes from 12/14 to 14/14 under its deliberately eval/mixed-mode fixture after VARE's adapter fix. These are same-author, same-host results. The patch has **not** been submitted; RVL maintainer review and an independent outside reproduction are outstanding. No default-path behavior change, real-model quality, capability, or production-frequency claim is made.
