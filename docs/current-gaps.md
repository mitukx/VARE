# Current evidence gaps

This assessment ranks the technical work needed to study post-training signals and policy updates under limited compute. All proposed work fits a local CPU/no-paid-service budget. VARE has two accepted synthetic preference-policy studies, a non-pass cached-model update, and narrow historical trainer-grader evidence.

## What current evidence supports

- A local evaluation control plane with bounded execution, provenance checks, retained records, offline audits, and same-host recovery.
- Frozen CPU evidence for failure handling, scheduling, freshness checks, transactional recovery, and environment command output limits. Measurements are specific to the tested fixtures and machine.
- Two calibrated historical task/grader pairs. The TRL GRPO normalizer evaluator was advanced to protocol v8 after real-source mutations showed that v7 accepted both an early `return None` and a zeroed policy-loss numerator. The retained v8 mutation study records v7 accepting both and v8 rejecting both while accepting the unmodified fixed revision.
- A frozen synthetic DPO-style confirmation on 10 independent seeds: mean held-out NLL improvement 0.3074 nats/pair, paired bootstrap 95% interval [0.2928, 0.3226], all 10 seeds improve, and mean KL 0.3874 under its 0.5 ceiling. Raw examples and seed-by-arm metrics are retained and reconstructed by the audit.
- A second, pre-run-committed synthetic study compares clean training, 20%/40% pair-orientation flips, and evaluation on a predefined shifted preference function. Its audited clean arm improves base-teacher held-out NLL by 0.3174 nats/pair (paired 95% seed-bootstrap interval [0.3021, 0.3328]), all 10 seeds improve, and mean KL is 0.3952. Both flip arms have higher mean NLL than clean on each seed; mean clean-arm NLL rises by 0.0793 under the specified shift. The offline audit also passes from a detached clean clone and from Ubuntu/Python 3.11 CI. See the [noise/shift report](synthetic-preference-robustness-v1-report.md).
- A frozen CPU update on a cached 0.5B language model completed three adapter initializations. Mean held-out conditional preference NLL worsened by 0.1120, its paired interval crossed zero, and one seed exceeded the KL ceiling. The preregistered result is a non-pass; the first code-failure attempt is also preserved. See the [model-level report](cpu-lm-dpo-head-v1-report.md).
- A negative local small-model agent pilot: three formal attempts produced no accepted patch.
- Contracts for replay, freshness, curriculum, promotion, and RVL trainer integration.

These demonstrate narrow evaluation/reliability properties, two synthetic policy updates under known generators, and a failed-to-improve small-model update. No evidence demonstrates reliable improvement in language-model preferences or capabilities, general RLHF/DPO behavior, or deployment-scale training.

## Ranked gaps

| Priority | Gap | Why it matters | No-cost work that closes part of it |
| --- | --- | --- | --- |
| 1 | The cached-model update did not improve held-out preferences reliably | The three-seed run's mean NLL change was +0.1120 and the 95% interval crossed zero; seed 101 worsened substantially and seed 211 exceeded the KL ceiling. | Stop tuning this frozen protocol. If resuming backend work later, run a distinct training-only development procedure and commit a new lock before fresh confirmation examples. |
| 2 | v2 synthetic protocol chronology is not independently anchored | The exploratory training-only sweep that informed its budget was not retained, and the separate development replay was generated after confirmation. The primary metrics reconstruct, but v2 should not be described as independently preregistered. | Preserve this provenance limitation in all presentations. |
| 3 | Robustness evidence remains synthetic and narrow | The new locked study tests independent pair-orientation flips and one explicit shifted teacher, but uses a known four-action teacher and a linear softmax policy. It does not model annotator disagreement, learned reward models, adaptive behavior, or language outputs. | Do not tune the completed protocol. Seek independent clean-clone reproduction, then define a separate no-download task with an executable oracle and distinct preference generator if it adds information beyond the existing cached-model non-pass. |
| 4 | Scale and task diversity are absent | Small arithmetic choices do not test long-form language, coding, broad reasoning, truthfulness, or deployment constraints. | Add only a CPU-feasible task with an independent oracle; state what remains unmeasured. |
| 5 | Grader evidence is still narrow | The v8 audit demonstrates detection of two specific false accepts in one pinned historical change; it does not establish arbitrary-code soundness, broad precision/recall, or full trainer correctness. | Expand with held-out refactors and independently designed meaningful mutations. Preserve the claim boundary and pursue an external clean-clone reproduction. |
| 6 | Independent reproduction and provenance review are missing | Local manifests and CI make the runs reconstructible, but no independent person has reproduced the learner results or reviewed the protocol. | Seek a clean-clone reproduction and independent technical review; keep the request and outcome in the record. |
| 7 | Individual ownership and operational scope are not demonstrated | A repository cannot prove unaided understanding, judgment, communication, collaboration, distributed systems experience, or hostile-code isolation. | Re-derive the objective and gradient, explain the v1 model non-pass, the synthetic v1 drift miss, and v2 provenance limits; reproduce an audit unaided. |

## Immediate order

1. Keep v1's KL failure, v2's chronology limit, and the invalid v2 shuffle arm visible; use the new robustness study only within its stated synthetic claim boundary.
2. Seek a clean-clone reproduction and independent protocol review for the completed robustness bundle.
3. Expand grader mutation coverage with held-out refactors while preserving narrow per-case claims.
4. Do not tune the failed cached-model protocol. A future model study needs a distinct train-only development cohort, fresh confirmation data, and a new committed lock.
5. Prepare an unaided technical walkthrough of the objective, gradient check, model non-pass, v1 drift failure, v2 chronology, robustness result and claim boundaries.

The code-agent trajectory remains a separate supporting question. Its negative pilot should stay visible, but should not displace a learner experiment as the next priority. Update this page when retained evidence changes, not when an integration or plan alone is added.
