# Crash-recovery harness correction v2

The first corrected crash attempt killed and restarted the Python learner, but the strict matched-candidate check failed. The worker had trained in source member order with RVL's default prompt grouping. The frozen VARE synchronous arm instead adds the authenticated group to `PrioritizedReplay`, calls `sample_current`, converts each experience through `RVLGRPOHooks`, and passes group-relative advantages to the trainer. Priority sorting changes the order of gradient accumulation, so the different harness path produced a numerically close but non-identical checkpoint.

The first failed crash attempt and its `SHA256.json` remain under `results/verifier_consistent_async_grpo_v6/crash-82106-corrected`; `FAILURE_POSTMORTEM.json` records the discrepancy. It is excluded from success counts.

Correction v2 applies the same whole-group replay ordering, hook conversion, and VARE group-relative advantage calculation as the frozen matched arm. The protocol and exact-state recovery criterion are unchanged. The corrected five-cutpoint run matched the synchronous candidate exactly at every cutpoint and passed the independent auditor.

`crash-runner-correction-v2.json` binds the corrected runner and regression test to the original v6 protocol and records that it supersedes the evaluator-cardinality-only correction. Reproduction commands and measured limits are in [`crash-recovery-results-v6.md`](crash-recovery-results-v6.md).
