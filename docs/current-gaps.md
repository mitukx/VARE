# Current evidence gaps

This assessment ranks the technical work needed to study post-training signals and policy updates under limited compute. All proposed work fits a local CPU/no-paid-service budget. The project has substantial evaluation and reliability evidence plus one synthetic policy-update diagnostic, but no real-model learner update.

## What current evidence supports

- A local evaluation control plane with bounded execution, provenance checks, retained records, offline audits, and same-host recovery.
- Frozen CPU evidence for failure handling, scheduling, freshness checks, transactional recovery, and environment command output limits. Measurements are specific to the tested fixtures and machine.
- Two calibrated historical task/grader pairs and adversarial grader revisions that corrected specific false accepts.
- A negative local small-model agent pilot: three formal attempts produced no accepted patch.
- Contracts for replay, freshness, curriculum, promotion, and RVL trainer integration.

These demonstrate narrow evaluation and reliability properties. The synthetic run does show a policy update on a toy preference distribution, but it failed its frozen acceptance rule. No evidence demonstrates improvement in language-model preferences or capabilities, RLHF/DPO behavior on a language model, or deployment-scale training.

## Ranked gaps

| Priority | Gap | Why it matters | No-cost work that closes part of it |
| --- | --- | --- | --- |
| 1 | The first synthetic preference run failed its frozen acceptance rule | v1 lowered mean held-out synthetic preference NLL by 0.3523 nats/pair over 10 seeds, but mean KL was 0.5629 against a 0.5 ceiling. The initial no-update accuracy metric also incorrectly treated ties as wrong. | Keep v1 immutable and visible. A new version must fix tie handling, select the update budget from training-only diagnostics, freeze a separate lock, and rerun an independent seed cohort. Do not relax v1's KL ceiling after observing results. |
| 2 | Synthetic evidence does not establish real-model post-training | The current preference result uses a tiny linear policy and synthetic Bradley–Terry preferences. It cannot say anything about language-model DPO, natural-language preferences, reasoning, or truthfulness. | First establish a reliable synthetic control. Only then test whether cached weights and installed dependencies permit a separately frozen, hard-bounded CPU adapter smoke test without downloads or spend. |
| 3 | Robustness evidence is preliminary | v1 contains 20% label-flip and shuffled-label arms, but one run on one synthetic generator does not establish behavior under realistic reward/preference shift. | Retain these arms as exploratory diagnostics; define a new preregistered shift condition, calibration metric, and held-out decision rule before a confirmatory run. |
| 4 | Learner provenance and diagnostics are not demonstrated end to end | A trustworthy result needs exact reference/policy/data versions and evidence that a candidate changed as intended. | Add objective traces, parameter snapshots/deltas, split identity, runtime/memory, source/data/config hashes, and a transactional candidate record to a future small-model run. |
| 5 | Narrow grader/task coverage and no independent reproduction | Historical fixes calibrate specific checks, not general evaluator precision/recall. Local CI is not external reproduction. | Add held-out refactors and meaningful mutations; ask for a clean-clone reproduction or submit a narrow upstream contribution and retain the outcome. |
| 6 | Single-host operational and trust boundaries | CPU timing is machine-local; candidate execution is not hostile-code isolation. | Add another OS/Python correctness run if available and keep arbitrary candidate code out of scope unless a tested disposable sandbox exists. Do not imply distributed, fleet, or production evidence. |
| 7 | No individual skill or collaboration evidence | A repository cannot establish unaided understanding, ownership boundaries, or communication. | Keep attribution accurate; prepare to derive the preference objective/gradient, explain a grader false accept and its limits, and reproduce a small result unaided. This is preparation guidance, not repo evidence. |

## Immediate order

1. Use the retained v1 as a diagnostic non-pass. Freeze a separate protocol after fixing tie scoring and choosing a lower update budget using training-only diagnostics.
2. Run the new seed cohort, reconstruct all metrics from the raw bundle, and report the acceptance outcome without overclaiming.
3. Only then assess whether the already-cached tiny-model environment can support a bounded adapter/DPO smoke test without downloads or spending.
4. Improve end-to-end learner provenance and diagnostics, then seek independent reproduction and broaden grader coverage.

The code-agent trajectory remains a separate supporting question. Its negative pilot should stay visible, but should not displace a learner experiment as the next priority. Update this page when retained evidence changes, not when an integration or plan alone is added.
