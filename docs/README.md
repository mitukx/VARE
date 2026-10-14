# Documentation index

Start here for navigation. Reports and frozen evidence retain their original paths and dates; an old PR-status snapshot is historical, not a live status assertion.

| Purpose | Entry point |
| --- | --- |
| Understand the research question, current outcomes, and quick start | [Root README](../README.md) |
| Review a small representative set of contributions | [Reviewer guide](reviewer-guide.md) |
| Locate implementation, experimental components, and raw evidence | [Repository layout](repository-layout.md) |
| Check unresolved evidence and experiment admission | [Current gaps](current-gaps.md), [agent charter](../AGENTS.md) |
| Follow completed studies and negative results | [Experiment index](experiments.md), [evidence ledger](evidence.md), [roadmap chronology](roadmap.md) |
| Reproduce current runtime ownership and recover after a crash | [Whole-round ownership](runtime-round-ownership-v1.md) |
| Reproduce evaluation cleanup, discard, and publication boundaries | [Evaluation integration](evaluation-transaction-integration-v1.md), [discard](candidate-discard-recovery-v1.md), [publication](promotion-publication-recovery-v1.md) |
| Inspect the real trainer integration and inference ownership | [RVL integration](rvl-evaluation-promotion-e2e-v1-report.md), [inference transaction](rvl-inference-transaction-v1-report.md) |
| Inspect replay/provenance and authenticated evidence contracts | [Trajectory reference](trajectory-reference-v1.md), [checkpoint-bound promotion](checkpoint-bound-promotion.md), [attested promotion](attested-promotion.md) |
| Run durable historical evaluations (a different state machine) | [Local evaluation recovery](recovery.md), [CLI](execution.md) |
| Read real-model outcomes without confusing feasibility with learning | [OpenBookQA confirmation](openbookqa-qwen25-grpo-sft-confirmation-v1-report.md), [format/correctness comparison](format-vs-correctness-comparison-v2.md), [one-update smoke](heldout-exact-integer-update-cost-smoke-v1.md) |
| Preserve artifacts and avoid new large tracked files | [Artifact retention](artifact-retention.md) |
| Find reproduction commands by purpose | [Script index](../scripts/README.md) |

No retained real-model update establishes an independently confirmed held-out task-success gain. CPU fixture correctness, historical grader calibration, model feasibility, and task-success comparisons are different evidence levels. Read each report's protocol, raw records, and limits before reusing its numbers.
