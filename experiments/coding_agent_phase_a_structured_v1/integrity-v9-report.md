# Phase A integrity v9 status

The independent review of v8 found that the shared Phase A comparison helper rejects `None` even when both values are `None`. The locked tokenizer registry has ten optional runtime routes whose correct value is `None`, so v8 would reject the exact known fix. Protocol v8 and its implementation remain unchanged.

V9 runs the existing parent comparison function with a private globals dictionary containing a recursive JSON comparator. It treats matching nulls as equal, compares mappings and lists recursively, and delegates scalar and numeric comparisons to the existing comparator without changing the shared comparator module. Focused regressions check the complete pinned runtime registry and reject a null-to-non-null mutation. The v9 protocol and implementation hashes are separately locked.

**Status: NOT_RUN.** No baseline, known-fix, or adversarial candidate grading was started. The current host preflight reports 61% free memory and 1,369.56 MiB free swap against the frozen minimum of 2 GiB. The pinned grading image is present in the `colima` Docker context; the active `default` context has no running daemon. No model was loaded, no inference or optimizer update occurred, and no candidate was graded.

The next valid action is a fresh preflight after swap has naturally fallen below the protocol's limit and the `colima` context is selected explicitly. Do not change v9 limits based on this result. Calibration remains Development-only for the already-consumed Transformers task.
