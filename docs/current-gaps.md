# Current evidence gaps

This assessment ranks the technical work needed to study post-training signals and policy updates under limited compute. All proposed work fits a local CPU/no-paid-service budget. VARE has one accepted synthetic preference-policy confirmation, a mixed/non-pass cached-model update, and narrow historical trainer-grader evidence.

## What current evidence supports

- A local evaluation control plane with bounded execution, provenance checks, retained records, offline audits, and same-host recovery.
- Frozen CPU evidence for failure handling, scheduling, freshness checks, transactional recovery, and environment command output limits. Measurements are specific to the tested fixtures and machine.
- Two calibrated historical task/grader pairs. The TRL GRPO normalizer evaluator was advanced to protocol v8 after real-source mutations showed that v7 accepted both an early `return None` and a zeroed policy-loss numerator. The retained v8 mutation study records v7 accepting both and v8 rejecting both while accepting the unmodified fixed revision.
- A frozen synthetic DPO-style confirmation on 10 independent seeds: mean held-out NLL improvement 0.3074 nats/pair, paired bootstrap 95% interval [0.2928, 0.3226], all 10 seeds improve, and mean KL 0.3874 under its 0.5 ceiling. Raw examples and seed-by-arm metrics are retained and reconstructed by the audit.
- A frozen CPU update on a cached 0.5B language model completed three adapter initializations. Mean held-out conditional preference NLL worsened by 0.1120, its paired interval crossed zero, and one seed exceeded the KL ceiling. The preregistered result is a non-pass; the first code-failure attempt is also preserved. See the [model-level report](cpu-lm-dpo-head-v1-report.md).
- A negative local small-model agent pilot: three formal attempts produced no accepted patch.
- Contracts for replay, freshness, curriculum, promotion, and RVL trainer integration.

These demonstrate narrow evaluation/reliability properties, one accepted synthetic policy update, and a failed-to-improve small-model update. No evidence demonstrates reliable improvement in language-model preferences or capabilities, general RLHF/DPO behavior, or deployment-scale training.

## Ranked gaps

| Priority | Gap | Why it matters | No-cost work that closes part of it |
| --- | --- | --- | --- |
| 1 | The cached-model update did not improve held-out preferences reliably | The three-seed run's mean NLL change was +0.1120 and the 95% interval crossed zero; seed 101 worsened substantially and seed 211 exceeded the KL ceiling. | Stop tuning this frozen protocol. If resuming backend work later, run a distinct training-only development procedure and commit a new lock before fresh confirmation examples. |
| 2 | v2 synthetic protocol chronology is not independently anchored | The exploratory training-only sweep that informed its budget was not retained, and the separate development replay was generated after confirmation. The primary metrics reconstruct, but v2 should not be described as independently preregistered. | Preserve this provenance limitation in all presentations. |
| 3 | Synthetic confirmation covers one known generator | v2 uses one teacher, context distribution, pairwise-label process, and small action set. Its flip arm is diagnostic, and its shuffled-ID arm is invalid because action IDs were shuffled across distinct pairs. | Freeze a distinct falsification study for reward/preference shift, annotator noise, calibration and reward hacking, with disjoint test contexts and matched budgets. Retain null or adverse outcomes. |
| 4 | Scale and task diversity are absent | Small arithmetic choices do not test long-form language, coding, broad reasoning, truthfulness, or deployment constraints. | Add only a CPU-feasible task with an independent oracle; state what remains unmeasured. |
| 5 | Grader evidence is still narrow | The v8 audit demonstrates detection of two specific false accepts in one pinned historical change; it does not establish arbitrary-code soundness, broad precision/recall, or full trainer correctness. | Expand with held-out refactors and independently designed meaningful mutations. Preserve the claim boundary and pursue an external clean-clone reproduction. |
| 6 | Independent reproduction and provenance review are missing | Local manifests and CI make the runs reconstructible, but no independent person has reproduced the learner results or reviewed the protocol. | Seek a clean-clone reproduction and independent technical review; keep the request and outcome in the record. |
| 7 | Individual ownership and operational scope are not demonstrated | A repository cannot prove unaided understanding, judgment, communication, collaboration, distributed systems experience, or hostile-code isolation. | Re-derive the objective and gradient, explain the v1 model non-pass, the synthetic v1 drift miss, and v2 provenance limits; reproduce an audit unaided. |

## Immediate order

1. Keep v1's KL failure and v2's accepted-but-not-independently-preregistered result visible; exclude the invalid v2 shuffle arm.
2. Freeze a new model study only after a train-only development cohort selects the budget; use fresh seeds and held-out examples and preserve this failed v1 as baseline context.
3. Freeze a distinct preference-shift/noise study before examining held-out outcomes; v2's flip arm is diagnostic and its shuffle arm is invalid.
4. Seek an independent clean-clone reproduction and protocol review, then extend mutation coverage with held-out refactors.
5. Prepare an unaided technical walkthrough of the objective, gradient check, model non-pass, v1 drift failure, v2 chronology and claim boundaries.

The code-agent trajectory remains a separate supporting question. Its negative pilot should stay visible, but should not displace a learner experiment as the next priority. Update this page when retained evidence changes, not when an integration or plan alone is added.
