# RVL rollback mode restoration: shared-module follow-up v2

## Finding

RVL main at `27ebf7fe239d97504eeb960bf1407de220584dac` restored saved module modes by calling `module.train(saved_mode)` while traversing `named_modules()`. PyTorch's `named_modules()` deduplicates shared module objects, while `Module.train()` recursively propagates to descendants. In a graph where a child is registered under two parents with different modes, restoring the later parent can overwrite the shared child's saved mode.

The regression creates that graph with the actual tiny GPT-2 RVL trainer, populates AdamW state, snapshots a mixed train/eval state, performs a real optimizer update, injects an exception immediately afterward, and invokes rollback. The baseline restores model tensors, optimizer state, and CPU RNG, but leaves the shared Dropout in the wrong mode; the next forward therefore differs.

## Correction and verification

The patch restores the validated boolean flags by assigning `module.training` directly. It changes no optimizer, model tensor, RNG, snapshot schema, or rollback ordering behavior.

On 2026-10-11, the VARE reproduction script created clean baseline and patched clones at the current RVL main commit. The baseline regression failed on exactly `per-module modes` and `subsequent dropout forward`; the patched trainer test module passed 14/14. The raw logs, runtime, and status are retained in [`clean-clone-replay-20261011`](../results/rvl-module-mode-shared-alias-v1/clean-clone-replay-20261011/).

An additional run on the patched checkout executed:

```bash
python3.12 -m unittest tests.test_mini_lab_torch tests.test_mini_lab tests.test_mini_lab_promotion -v
```

Result: **51 tests passed**. This includes the shared-module regression, existing mixed-mode and legacy-snapshot tests, optimizer/RNG rollback checks, and adjacent replay and promotion tests. The run log is retained with the clean-clone artifacts. The clean-clone reproduction command is:

```bash
PYTHON=python3.12 bash scripts/reproduce_rvl_shared_module_mode_followup_v1.sh \
  results/rvl-module-mode-shared-alias-v1/replay-YYYYMMDD
```

The narrow change and regression are commit [`7568c6e`](https://github.com/mitukx/Recursive-Verification-Lag/commit/7568c6e); PR [#92](https://github.com/mitukx/Recursive-Verification-Lag/pull/92) is open against RVL `main`. It has not been merged or independently human-reviewed.

## Independent challenge and limits

An independent AI reviewer inspected the pinned restoration code and the proposed semantics, independently reproduced the shared-child mode failure, and checked that direct flag restoration preserves model state, optimizer state, CPU RNG, module flags, and the following Dropout forward. The reviewer ran the pre-fix `tests.test_mini_lab_torch` suite: 13 tests passed. No distinct additional defect was identified; the review noted the custom-override and CUDA limitations below. This is an AI-agent review, not independent human review. The full 51-test patched suite was run by the primary agent.

The regression deliberately constructs shared module aliasing. It does not establish that RVL's default Qwen path contains such aliases, that a production run encountered the defect, or that fixing it changes model quality. CUDA RNG and distributed rollback were not exercised because this environment has no CUDA device. Direct flag assignment restores the boolean `training` state captured by the snapshot; it does not reverse arbitrary side effects in custom `Module.train()` overrides.

## Evidence classification

- **Implemented:** local RVL branch, commit `7568c6e`; one source line changes plus regression coverage.
- **Unit tested:** 51 related CPU tests pass on the patched checkout.
- **Reproduced on CPU:** clean-clone baseline fails the targeted regression; clean-clone patched trainer module passes 14/14.
- **Measured on a real model:** no post-training model-quality experiment was run. The test uses RVL's actual tiny GPT-2 trainer fixture and a real optimizer update, not a capability evaluation.
- **Verified on GPU/distributed hardware:** no.
- **Merged upstream:** no; PR #92 is open.
- **Independently human-reviewed:** no.
