# Experiment summary

**Replay status:** COMPLETED. **Task eligibility:** BLOCKED.

The v3 baseline test required the first registration to succeed and the second to raise the exact expected conflict. Both conditions passed. The fixed source passed same-fusion idempotence, unrelated-mapping retention, and unrelated-conflict rejection. All three predeclared source mutations were detected. A separate control injected an unrelated ValueError on the first registration; the baseline reproducer propagated it and classified it as FIRST_CALL_ERROR_REJECTED.

| Case | Result |
|---|---|
| Baseline | BASELINE_FAIL_REPRODUCED |
| Fixed | PASS |
| No deduplication mutation | MUTANT_KILLED |
| Clear unrelated mappings mutation | MUTANT_KILLED |
| Allow conflict mutation | MUTANT_KILLED |
| Unrelated first-call error control | FIRST_CALL_ERROR_REJECTED |

All six cases matched the frozen protocol. Total in-container probe time was 0.120339 seconds and host invocation time was 1.163444 seconds. Maximum per-process RSS was 18,743,296 bytes; maximum Python allocation peak was 632,373 bytes. Each container had one CPU, a 512 MiB memory limit, 64 PIDs, and no network. Host swap remained at 3,734.44 MiB used and 1,385.56 MiB free. No timeout or OOM occurred.

The v2 artifacts are preserved unchanged, but the independent exact-head review found that v2 could misclassify a first-call error as a reproduced baseline failure. V3 addresses that finding with an exact second-call error assertion and the first-call fault control. V2 is therefore retained as exploratory output only and is not accepted as authoritative baseline evidence.

This remains narrow source-level evidence, not task certification. The harness stubs model discovery and mapping-registry persistence. No candidate patch, hidden grader, complete upstream test suite, model inference, agent trajectory, reward, or optimizer update was run. The task remains in the original frozen heldout allocation; repository-patch isolation and the independent grader boundary remain blockers.
