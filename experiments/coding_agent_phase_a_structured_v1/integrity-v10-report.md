# Phase A integrity v10 status

The independent review of v9 found that its private null-safe comparator did not accept the `bool_as_number` keyword always passed by the actual parent `grade_calls` function. That made the known-fix comparison fail with a call-interface `TypeError` before checking candidate output. V9 files and its protocol lock remain unchanged.

V10 preserves the parent comparator signature and forwards `bool_as_number` recursively, including to the existing scalar comparator. A no-Docker regression clones the real trusted `grade_calls` function, replaces only its candidate invocation primitive, and verifies nested null comparison plus both boolean-number modes. The v10 runner, protocol, grader adapter, shared comparison and isolation sources, and regression are hash-locked.

**Status: NOT_RUN.** No candidate baseline, known-fix, or adversarial grading was started. Calibration remains Development-only for the already-consumed Transformers task. The frozen host-memory and free-swap thresholds remain in force; do not start grading until a fresh preflight satisfies them and the pinned image is accessible through the selected Docker context. No model inference or optimizer update occurred.
