# Current evidence gaps

This page summarizes what the retained evidence supports today and what would most improve the next evidence tier. It is a technical project assessment, not a claim about model capability or deployment readiness. All proposed work is designed to fit a local CPU and no-paid-service budget.

## What the repository currently demonstrates

- A local evaluation control plane with bounded execution, provenance checks, retained records, offline audits, and same-host recovery.
- Frozen CPU evidence for failure handling, local scheduling, freshness checks, and transactional recovery. The measurements are specific to the tested fixtures and local machine.
- The environment command runner now enforces its per-stream output capture limit while draining child processes and rejects overflow; a frozen baseline/fixed regression covers stdout, stderr, normal output, and POSIX process-group cleanup.
- Two calibrated historical task/grader pairs. The fixed upstream revisions pass and the pre-fix revisions fail under the declared fixtures.
- Adversarial checks that found and corrected several false accepts in the TRL grader. Protocol v7 rejects five frozen mutations while accepting the pinned fixed source across 12 arithmetic conditions.
- Transparent reporting of a negative local agent pilot: three formal runs produced no accepted patch.

These are useful evaluation and reliability engineering results. They do not demonstrate that an agent can solve tasks, that model quality improves, or that the system operates at fleet scale.

## Ranked gaps

| Priority | Gap | Why it matters | No-cost evidence that would close part of it |
| --- | --- | --- | --- |
| 1 | No reproducible successful agent task trajectory | The only formal agent cohort is one task, one small local model, and three unsuccessful runs. This leaves the central agent-workflow claim untested. | Freeze a small, varied CPU-verifiable task pack; improve the patch/tool protocol; run repeated seeds in fresh workspaces; retain every trace, patch, grade, and budget. Require at least one independently replayable accepted patch before claiming a positive result. |
| 2 | Narrow task and grader coverage | Two historically fixed tasks calibrate specific contracts; they do not establish broad verifier precision, recall, or resistance to plausible wrong patches. | Add tasks from distinct code paths or repositories with independently justified expected outcomes, positive and negative controls, meaningful mutations, and a held-out task set. Report false accepts and false rejects without extrapolating beyond the tested set. |
| 3 | No independent reproduction or upstream contribution | Local tests and CI establish project-internal reproducibility, not usefulness to another maintainer or user. | Ask an independent person to reproduce a frozen bundle from a clean clone, or submit a narrow, useful upstream fix and retain review/CI outcomes. AI review alone is not independent corroboration. |
| 4 | Single-host operational evidence | Current scheduling, freshness, and recovery results are local CPU measurements; they say little about workload heterogeneity, OS differences, or fleet behavior. | Run the same frozen correctness suite on Linux and another supported Python version; profile mixed-duration CPU jobs and record latency, peak memory, storage, and recovery behavior. Keep distributed-scale claims out unless actually measured. |
| 5 | Grader semantic and runtime limits | TRL v7 uses source-structure checks plus scalar arithmetic fixtures, and RVL uses deterministic model/tokenizer doubles. Neither proves arbitrary semantic correctness or full production-model parity. | Add held-out refactors and adversarial mutations; where feasible, compare a small dependency-backed CPU fixture against upstream runtime behavior. Document every accepted/rejected pattern and keep a full-trainer claim out of scope. |
| 6 | Hostile-code and resource-isolation boundary | Candidate code runs with the current user's permissions; process groups and before/after hashes are not a security sandbox or hard CPU/RAM/network quota. | Keep tasks trusted, or move candidate execution to a disposable local container/VM with read-only evaluator assets and explicit CPU, memory, disk, and network controls; test escape and timeout behavior before describing isolation. |
| 7 | End-to-end improvement loop not measured | Components for failure signals, selection, evaluation, and promotion exist, but a complete intervention has not shown an independently measured downstream benefit. | Use a preregistered CPU-only mechanism experiment with a matched-budget fixed-selector baseline, held-out decision outcome, and explicit synthetic/mechanism-only labels. |

## Claim ceiling

Current evidence supports describing VARE as a local, CPU-tested evaluation control plane with narrow historical grader calibration, adversarial grader revisions, and same-host reliability measurements. It does not support claims of successful agent solving, general grader validity, model improvement, generalization, hostile-code isolation, independent adoption, or distributed/production scale.

## Recommended order

1. Build a small diverse frozen task pack and establish a valid repeated-run harness.
2. Produce and independently replay at least one successful task trajectory; preserve failures and budget accounting.
3. Expand held-out grader mutations/refactors and seek one independent reproduction or narrow upstream review.
4. Measure cross-platform and heterogeneous CPU behavior, then revisit the end-to-end intervention only when its outcome can be measured independently.

Do not add more infrastructure surface until a retained failure or task outcome motivates it. Update this page when evidence changes, not when implementation alone changes.
