# Contribution artifacts

This directory retains source patches used for upstream reproductions and review. Frozen protocol paths refer to historical patch files, so old artifacts remain in place. For the latest state, start with the first two entries.

| Artifact | Status | Use |
| --- | --- | --- |
| [`rvl-module-mode-shared-alias-followup.patch`](rvl-module-mode-shared-alias-followup.patch) | Current follow-up candidate; tested against merged RVL `27ebf7f`; not submitted | Fixes recursive mode overwrite for a shared child module. See [report](../docs/rvl-module-mode-shared-alias-followup-v1.md) and [reproducer](../scripts/reproduce_rvl_shared_module_mode_followup_v1.sh). |
| [`rvl-module-mode-snapshot-v4.patch`](rvl-module-mode-snapshot-v4.patch) | Merged upstream in RVL PR [#90](https://github.com/mitukx/Recursive-Verification-Lag/pull/90); has the shared-child limitation covered by the follow-up above | Original exact per-module mode snapshot change; retained for provenance and clean-clone reproduction. See [v4 report](../docs/rvl-module-mode-snapshot-final-review-v4.md). |
| `rvl-module-mode-snapshot-v1.patch` through `v3.patch` | Historical iterations | Superseded by v4; retained because frozen protocols and reports reference them. |
| `rvl-module-mode-fix-only-v*.patch`, `rvl-module-mode-test-only-v*.patch` | Historical baseline/fix controls | Apply only with the matching frozen protocol and reproducer. Do not use as current proposed patches. |

Other upstream-facing studies are documented in their individual reports. In particular, VARE's TRL #7249 artifacts reproduce an existing upstream candidate; they are not VARE-authored fixes.
