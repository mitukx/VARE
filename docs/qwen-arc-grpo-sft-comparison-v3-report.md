# Qwen ARC-Challenge GRPO versus successful-trace SFT — v3

**Decision: STOP this model/task pairing.** The frozen CPU comparison did not show GRPO outperforming matched successful-trace SFT or the base policy. The result is a narrow negative development outcome, not a general conclusion about GRPO.

## Protocol and data

Protocol [`qwen_arc_grpo_sft_comparison_v3.lock.json`](../protocols/qwen_arc_grpo_sft_comparison_v3.lock.json) was frozen before response generation. It compared Qwen2.5-0.5B-Instruct base, GRPO and successful-trace SFT on one ARC-Challenge validation cohort, with three seeds and matched training-example / update budgets. The primary outcome was exact task success; the uncertainty interval was a task bootstrap stratified by answer key. Earlier v1/v2 harness failures occurred before response generation or optimizer updates and are preserved separately; v3 records their protocol corrections and did not reuse any prior generated responses.

## Results

- Base: `48/115` exact answers (`41.74%`).
- GRPO seed scores: `48/115`, `44/115`, `42/115`; mean `38.84%`.
- Successful-trace SFT seed scores: `45/115`, `45/115`, `48/115`; mean `40.00%`.
- GRPO minus SFT: `−1.159` percentage points; answer-key-stratified task-bootstrap 95% interval `[−5.217, +2.899]` points.
- GRPO minus base: `−2.899` points; task-bootstrap 95% interval `[−8.986, +3.188]` points.
- The frozen positive advancement gate did not pass.

The corrected independent auditor reconstructed the raw answers, group rewards, advantages, rollouts/token integrity, exact-answer scores and bootstrap results. The first auditor attempt failed in its aggregation code after completing earlier checks; that failure is retained in `results/qwen-arc-grpo-sft-comparison-v3/run-3/independent_audit_attempt_v1.failure.json`. The corrected audit passed, and a separate read-only auditor independently recomputed the same metrics with discrepancies below `1e-16`: [corrected audit result](../results/qwen-arc-grpo-sft-comparison-v3/run-3/independent_audit_corrected_v1.json).

Resource limits passed: CPU-only, 756.4 seconds wall time, peak RSS 13,278.5 MiB, no paid services. Candidate checkpoint hashes were recorded during the run, but checkpoint files were removed after the save/reload round-trip and could not be rehashed afterward. This limits checkpoint-level reinspection. All raw generated outputs and the failed audit attempt remain retained under [`results/qwen-arc-grpo-sft-comparison-v3/run-3/`](../results/qwen-arc-grpo-sft-comparison-v3/run-3/).

## Interpretation

This comparison does not support GRPO over successful-trace SFT on this model, task, cohort, and budget. The confidence intervals include no difference and practically relevant effects in either direction. Do not claim an SFT or GRPO capability gain, generalize to other tasks/models, or continue tuning on the consumed cohort. The outcome closes this pairing under the frozen decision rule.

**Next action:** stop additional runs on this ARC pairing. The active technical finding in this iteration is separately documented in [the RVL AdamW report](rvl-grpo-zero-advantage-weight-decay-v1-report.md). Independently measured model task-success improvement remains VARE's primary evidence gap.
