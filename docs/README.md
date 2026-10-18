# Documentation

**First read:** [project summary and verified results](../README.md) → [architecture](architecture.md) → [engineering evidence](engineering-depth.md). This index is curated; the historical experiment and evidence ledgers deliberately preserve older records and dated decisions.

| Reader question | Reference |
| --- | --- |
| What problem does VARE solve, and what has actually been demonstrated? | [README](../README.md), [reviewer guide](reviewer-guide.md) |
| How do training, orchestration, evaluation and recovery fit together? | [Architecture and ownership](architecture.md), [engineering deep dive](engineering-depth.md) |
| What can I run now? | [Quick start](../README.md#quick-start), [reproduction scripts](../scripts/README.md), [CLI](execution.md) |
| How are repositories and source paths organized? | [Repository layout](repository-layout.md) |
| How are faults detected, candidates discarded and uncertain state recovered? | [Whole-round runtime](runtime-round-ownership-v1.md), [evaluation lifetime](engine-evaluation-lifecycle-v1-report.md), [publication](promotion-publication-recovery-v1.md), [durable jobs](recovery.md) |
| What stops graders when their worker is killed during evaluation? | [Owner-loss containment, resume and cost](grader-owner-loss-v1-report.md) |
| Do updated rollout workers and a SIGKILLed learner resume actual optimizer updates correctly? | [Continuous actual-model benchmark](e2e-rl-continuation-v2-report.md), [retained artifact audit](../README.md#audit-the-retained-model-benchmark) |
| What measured corruption, forged evidence, worker crash and cancellation faults? | [Preserved v1 actual-model CPU benchmark](e2e-rl-real-model-v1-report.md); different timing boundary and baseline from v2 |
| What does the earlier simulated policy measure? | [Four-parameter control-loop benchmark](e2e-rl-system-benchmark-v1.md); a separate workload and baseline |
| Can a resumed evaluator silently use different executor source? | [Durable source pinning and restoration](durable-runtime-source-v1-report.md) |
| How do actual trainer and trajectory boundaries work? | [RVL integration](rvl-evaluation-promotion-e2e-v1-report.md), [direct trainer freshness](rvl-direct-trainer-freshness-v1-report.md), [trajectory reference](trajectory-reference-v1.md), [trajectory safety](trajectory-aware-safety-v1.md) |
| Does trajectory replay match dispatch policy and terminal tool lifetime? | [Capture boundary report](trajectory-capture-boundary-v1-report.md) |
| Where are completed and failed experiments? | [Experiment index](experiments.md), [evidence ledger](evidence.md), [roadmap chronology](roadmap.md) |
| What is not validated, and what would justify another study? | [Current gaps](current-gaps.md), [research qualification](research-candidate-qualification-2026-10-15.md), [agent charter](../AGENTS.md) |
| Did a fresh model/task/method qualify for another learning comparison? | [2026-10-16 candidate screen](research-candidate-screen-2026-10-16.md): NO-GO before base inference; prior-art differences, unused metadata candidates and remaining grader/resource gates |
| How are protocols, artifacts and large blobs preserved? | [Repository layout](repository-layout.md), [artifact retention](artifact-retention.md) |

**Evidence policy:** A regression fixture validates a bounded implementation property; a model-feasibility test does not show learning; an optimizer smoke is not a held-out success result. No retained real-model policy update establishes independently confirmed held-out task-success improvement. Follow each report's pinned source, protocol, raw artifacts, and stated limits. Historical PR/CI status references are snapshots, not live state.
