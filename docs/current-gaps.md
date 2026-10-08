# Current evidence gaps

This assessment ranks the technical work needed to study post-training signals and policy updates under limited compute. All proposed work fits a local CPU/no-paid-service budget. VARE now has one accepted, frozen synthetic preference-policy confirmation and narrow historical trainer-grader evidence, but no learner update on a language model.

## What current evidence supports

- A local evaluation control plane with bounded execution, provenance checks, retained records, offline audits, and same-host recovery.
- Frozen CPU evidence for failure handling, scheduling, freshness checks, transactional recovery, and environment command output limits. Measurements are specific to the tested fixtures and machine.
- Two calibrated historical task/grader pairs. The TRL GRPO normalizer evaluator was advanced to protocol v8 after real-source mutations showed that v7 accepted both an early `return None` and a zeroed policy-loss numerator. The retained v8 mutation study records v7 accepting both and v8 rejecting both while accepting the unmodified fixed revision.
- A frozen synthetic DPO-style confirmation on 10 independent seeds: mean held-out NLL improvement 0.3074 nats/pair, paired bootstrap 95% interval [0.2928, 0.3226], all 10 seeds improve, and mean KL 0.3874 under its 0.5 ceiling. Raw examples and seed-by-arm metrics are retained and reconstructed by the audit.
- A negative local small-model agent pilot: three formal attempts produced no accepted patch.
- Contracts for replay, freshness, curriculum, promotion, and RVL trainer integration.

These demonstrate narrow evaluation/reliability properties and one accepted synthetic policy update. No evidence demonstrates improvement in language-model preferences or capabilities, RLHF/DPO behavior on a language model, or deployment-scale training.

## Ranked gaps

| Priority | Gap | Why it matters | No-cost work that closes part of it |
| --- | --- | --- | --- |
| 1 | No measured post-training update on a language model | The accepted DPO-style experiment trains a four-action linear policy on synthetic preferences. The historical TRL task grades selected source-derived arithmetic and mutations; it does not run a trainer or optimizer. | Determine whether existing cached weights and installed CPU dependencies allow one hard-bounded, no-download adapter update with held-out model outputs. If not, preserve that as an unresolved constraint and do not imply synthetic transfer. |
| 2 | v2 protocol chronology is not independently anchored | The exploratory training-only sweep that informed its budget was not retained, and the separate development replay was generated after confirmation. The recorded Git revision predates the uncommitted protocol files. The primary metrics reconstruct, but v2 should not be described as independently preregistered. | Commit a future protocol and lock before generating confirmation data; retain and audit the development run first. |
| 3 | Synthetic confirmation covers one known generator | v2 uses one teacher, context distribution, pairwise-label process, and small action set. Its flip arm is diagnostic, and its shuffled-ID arm is invalid because action IDs were shuffled across distinct pairs. | Freeze a distinct falsification study for reward/preference shift, annotator noise, calibration and reward hacking, with disjoint test contexts and matched budgets. Retain null or adverse outcomes. |
| 4 | Scale and task diversity are absent | One compact synthetic task does not test long-form language, reasoning, coding, truthfulness, or deployment constraints. | Add only a small evaluation that has a defensible CPU path and independent oracle; state what remains unmeasured. No GPU or paid compute is required for the next falsification and reproducibility steps. |
| 5 | Grader evidence is still narrow | The v8 audit demonstrates detection of two specific false accepts in one pinned historical change; it does not establish arbitrary-code soundness, broad precision/recall, or full trainer correctness. | Expand with held-out refactors and independently designed meaningful mutations. Preserve the claim boundary and pursue an external clean-clone reproduction. |
| 6 | Independent reproduction and provenance review are missing | Local manifests and CI make the runs reconstructible, but no independent person has reproduced the learner results or reviewed the protocol. | Ask for an external clean-clone replay and a technical review of the frozen protocol, raw reconstruction, and claim limits. Keep the request and outcome in the record. |
| 7 | Individual ownership and operational scope are not demonstrated | A repository cannot prove unaided understanding, judgment, communication, collaboration, distributed systems experience, or hostile-code isolation. | Re-derive the objective and gradient, explain the v1 drift miss and v2 provenance limits, and reproduce an audit unaided. |

## Immediate order

1. Keep v1's KL failure and v2's accepted-but-not-independently-preregistered result visible; exclude the invalid v2 shuffle arm.
2. Commit a new frozen protocol before a training-only development cohort, then confirm it on separate seeds with separate raw data. Complete a bounded feasibility inventory of already-cached model weights, tokenizer/runtime dependencies, and CPU memory/time; run an adapter only if no download, paid service, or unbounded execution is required.
3. Freeze a distinct preference-shift/noise study before examining held-out outcomes; v2's flip arm is diagnostic and its shuffle arm is invalid.
4. Seek an independent clean-clone reproduction and protocol review, then extend mutation coverage with held-out refactors.
5. Prepare an unaided technical walkthrough of the objective, gradient check, drift failure in v1, v2's chronology and synthetic claim boundary.

The code-agent trajectory remains a separate supporting question. Its negative pilot should stay visible, but should not displace a learner experiment as the next priority. Update this page when retained evidence changes, not when an integration or plan alone is added.
