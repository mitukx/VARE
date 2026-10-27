# Sprint VI Development task cohort evidence

This bundle contains the selected frozen protocol and raw attempt records for the qualified external Development task families, plus the unqualified/disqualified task ledgers needed to interpret the cohort. It excludes model weights, hidden grader implementations, hidden test source, fixed source files, and task solutions. Some attempt folders retain `candidate.diff`; these are raw model artifacts and are not assertions of validity. The #49900 reconciliation explicitly excludes the legacy `.strip()`-induced false diff and grade.

Four unique task families passed baseline-fail, known-fix-pass, incorrect-mutant-fail controls and were eligible for Development use at launch: Transformers #49696, #49975, #50001, and #49900. Each is now exposed and must not be reused as blind held-out evidence. All pretraining exposure remains unknown. The retained #49772 record is marked qualification-pending and is not counted. #49842 has a valid grader-control record but is disqualified as fresh because the same task family was used in Sprint II.

Use `development-task-cohort-summary.json` for compact qualification receipts and artifact hashes. See the parent Sprint VI report for model outcomes, failures, task-level interpretation, and GO/NO-GO decisions.

The two 1.7 MB #50001 protocol JSON artifacts are stored losslessly as deterministic gzip files to remain below the repository 1 MiB blob limit. Their original byte lengths and SHA-256 digests are retained in `campaign_50001/metadata/compressed-artifacts-v1.json`; the public verifier expands and checks both original byte streams.
