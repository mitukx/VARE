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

The earlier AI-review claim had no retained reviewer-run record at the time of the PR's first version; that unrecorded run is not evidence for this report. A fresh AI-agent review was run on 2026-10-11 and its packet is now retained in [`independent-review-2026-10-11`](../results/rvl-module-mode-shared-alias-v1/independent-review-2026-10-11/). The packet includes the exact reviewer prompt, host/runtime, baseline/candidate commits and source blob IDs, command lines, stdout/stderr logs (including an invalid initial baseline import attempt and the corrected rerun), findings, and a SHA-256 manifest. `SHA256SUMS` covers all 19 other packet files; its own SHA-256 is `364966ec394f7e6efe45907c2037c9f69e790b0714ca66aa4b958c44cd9880ef`. The execution environment identifies the reviewer as a GPT-6-family Codex AI subagent but does not expose an exact serving model/version ID.

The reviewer ran the candidate regression on `7568c6e` (pass) and the same test against extracted baseline `27ebf7f` (failures exactly `per-module modes` and `subsequent dropout forward`). The other model tensor, optimizer, and CPU RNG assertions did not fail on baseline. A separate tiny PyTorch graph reproduced the recursive overwrite under both registration orders and showed a matching-mode case that masks the failure. A synthetic custom `nn.Module.train()` override confirmed that direct flag assignment restores captured boolean modes but does not undo arbitrary override side effects. A search found no `Module.train()` override in RVL `src/` or `tests/`; external model subclasses were not audited. The review found no additional defect in the inspected scope. This was a separate AI-agent review, not human review. The full 51-test patched suite was run by the primary agent.

The regression deliberately constructs shared module aliasing. It does not establish that RVL's default Qwen path contains such aliases, that a production run encountered the defect, or that fixing it changes model quality. CUDA RNG and distributed rollback were not exercised because this environment has no CUDA device. Direct flag assignment restores the boolean `training` state captured by the snapshot; it does not reverse arbitrary side effects in custom `Module.train()` overrides.

## Evidence classification

- **Implemented:** local RVL branch, commit `7568c6e`; one source line changes plus regression coverage.
- **Unit tested:** 51 related CPU tests pass on the patched checkout.
- **Reproduced on CPU:** clean-clone baseline fails the targeted regression; clean-clone patched trainer module passes 14/14.
- **Measured on a real model:** no post-training model-quality experiment was run. The test uses RVL's actual tiny GPT-2 trainer fixture and a real optimizer update, not a capability evaluation.
- **Verified on GPU/distributed hardware:** no.
- **Merged upstream:** no; PR #92 is open.
- **Independently human-reviewed:** no.
