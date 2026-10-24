# Experiment summary

**Status:** COMPLETED for the frozen source-level replay; BLOCKED for repository-task eligibility.

The independent probe reproduced the repeated-registration conflict at the pinned baseline revision. The pinned fix passed all three properties: same-fusion idempotence, retention of unrelated converters, and rejection of unrelated conflicts. All three predeclared incorrect mutations were detected.

| Case | Result | Observation |
|---|---|---|
| Baseline | BASELINE_FAIL_REPRODUCED | Second registration raises the conversion-mapping ValueError. |
| Fixed | PASS | All three properties pass. |
| No deduplication mutant | MUTANT_KILLED | Repeated registration raises the same conflict. |
| Clear unrelated mappings mutant | MUTANT_KILLED | The unrelated converter disappears. |
| Allow conflict mutant | MUTANT_KILLED | A pre-existing conflicting converter is silently accepted. |

All five cases matched the frozen protocol. Total in-container probe time was 0.122194 s and total host invocation time was 1.082732 s. Maximum per-process RSS was 18,612,224 bytes; maximum Python allocation peak was 632,198 bytes. Containers had one CPU, a 512 MiB memory limit, 64 PID limit, and no network. Host swap remained at 3,734.44 MiB used and 1,385.56 MiB free throughout. No timeout or OOM occurred.

A preflight bind mount from /private/tmp failed before any case ran; its raw record is retained. Repeating only the hash-read preflight from the shared Documents/Codex path succeeded. The five frozen outcome cases then ran once each.

The v1 baseline pytest failure was a setup failure: unittest.setUp attempted to patch _REGISTERED_FUSION_SOURCE_PATTERNS, an attribute absent at baseline, before the target function was called. This v2 replay directly invokes the pinned upstream registration function with test doubles for model discovery and registry persistence, so it confirms the source behavior without relying on that fixture.

This is narrow source-level evidence, not task certification. The task remains in the original frozen heldout allocation. No candidate patch, hidden evaluator, full upstream test suite, model inference, agent trajectory, reward, or optimizer update was run. Repository-patch isolation and the independent grader boundary remain blockers.

Reproduction inputs, commands, protocol, and hashes are in protocols/transformers_49412_source_replay_v2.json, freeze.json, the probe script, and raw/.
