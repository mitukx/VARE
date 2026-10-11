# Research candidate screen — 2026-10-11 follow-up

## Decision

No new learning or GPU experiment is admitted. The repository's frozen Stage 0 decision remains in force; this screen supplements it with a structured comparison of three concrete questions. The source revision audited was VARE `b3204926b3cae291f7c13e3134444c71a3a63172`, and the frozen Stage 0 report, current gaps, experiment history, and evidence ledger were read before selecting work. Prior-art links below identify the closest retrieved primary work; they do not establish an exhaustive literature review.

The highest-value admissible work was the concrete RVL rollback defect documented in the [engineering follow-up](rvl-module-mode-shared-alias-followup-v2.md). No research hypothesis met both a defensible distinction and a usable fresh data/evaluation gate.

## Candidate 1 — rollout age, policy drift, and task-contract failure

1. **Exact research question:** On a fresh task with a functioning base policy, does increasing the number of learner updates per fixed rollout group cause a measurable loss in independent task success or output-contract validity, after controlling optimizer updates and generated-token budget?
2. **Closest existing work:** [Revisiting GRPO: Insights into On-Policy and Off-Policy Training](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b4008025c2182bfe16fcc8566ee14d64-Abstract-Conference.html), [Mu-GRPO](https://arxiv.org/abs/2605.17570), [Scaling Laws for Collapse in Asynchronous GRPO](https://arxiv.org/abs/2607.01083), and [Prefix-Normalized Policy Optimization](https://arxiv.org/abs/2608.01418) directly address off-policy reuse, staleness/collapse, or correction.
3. **What is genuinely different?** The only plausible distinction is a preregistered, independent task-contract outcome under a matched compute budget, rather than a new general stale-rollout algorithm. VARE's observed OpenBookQA drift is from a retired pairing and is confounded; it cannot supply a fresh confirmation result.
4. **Falsifiable hypothesis:** At fixed learner-update count, a prespecified refresh cadence improves held-out executable task success and/or parse validity over fixed-rollout reuse, with uncertainty excluding the frozen minimum effect; a change only in KL or clipping does not support a capability claim.
5. **Smallest decisive experiment:** First pass a new base feasibility gate. Then compare unchanged base and matched SFT controls with two GRPO refresh cadences on new train/dev rows and untouched confirmation rows, using multiple seeds and a fixed independent grader. Record token and rollout costs separately because refresh cadence changes generation cost.
6. **Independent outcome measure:** Task success from a held-out evaluator whose assets and scoring implementation are outside the optimizer path; report parse failures, per-seed results, and paired uncertainty.
7. **Necessary compute and data:** A legally usable fresh dataset, successful base-model interaction, reward variation, independent checker, frozen split, and CPU/GPU training capacity sufficient for matched arms. Current local candidate set does not meet this gate.
8. **Failure criteria:** Base feasibility or reward variance fails; control matching fails; independent checker disagrees; or the only measured change is training reward, KL, or clipping without the frozen held-out outcome effect.
9. **Expected contribution:** A positive result could quantify a narrow task-specific refresh trade-off; a negative result could rule out the prespecified effect on that task. Neither result alone would be a new general GRPO method.

**Decision: no-go.** The central mechanism is already directly studied in recent GRPO work, and no fresh, passing task/base pair is available. Reusing OpenBookQA confirmation data or altering its old protocol is prohibited.

## Candidate 2 — policy-conditioned verifier audits after verifier change

1. **Exact research question:** At equal audit cost after a verifier change, do audits drawn from the current policy's rollout distribution detect more false accepts on the current task distribution than a static verifier-mutation corpus?
2. **Closest existing work:** [Verifier Errors in RLVR](https://arxiv.org/abs/2609.35677) studies verifier-induced gradient errors and selective audit correction; [Verifier Fuzzing](https://arxiv.org/abs/2606.01066) and [Isomorphic Perturbation Testing](https://arxiv.org/abs/2604.15149) target verifier failures and reward hacking under perturbations. VARE already has verifier freshness/replay studies and synthetic audit-allocation analyses.
3. **What is genuinely different?** A policy-conditioned audit *selection* comparison under explicit version change and equal label cost could be distinct if it measures detection on a fresh, externally adjudicated task distribution. It is not distinct as currently specified enough to support a broad co-evolution or freshness claim.
4. **Falsifiable hypothesis:** Under a held-out verifier shift, current-policy-conditioned audits increase false-accept detection recall over static mutation audits at the same number of trusted labels, while remaining below a frozen false-alarm ceiling.
5. **Smallest decisive experiment:** Use a new deterministic task contract with independently trusted semantic labels, inject preregistered verifier changes, and compare policy-conditioned samples against a matched-size static perturbation set on untouched cases. Freeze the sampler and threshold before the held-out cases are scored.
6. **Independent outcome measure:** Recall and false alarms against the separate trusted semantic oracle on held-out policy outputs, with confidence intervals and exact call counts.
7. **Necessary compute and data:** A fresh policy/task distribution, versioned verifier changes, protected held-out cases, and enough independently labeled false-accept examples. Small deterministic CPU models could test the harness; a model-general claim needs substantially broader data.
8. **Failure criteria:** No improvement at equal cost; gains disappear on held-out outputs; false alarms exceed the frozen ceiling; or labels/selection leak into the verifier or optimizer.
9. **Expected contribution:** A positive result could motivate a task-scoped audit-selection rule. A negative result could show that a static audit set is equally effective under the tested shift. Both require independent labels and do not imply policy capability improvement.

**Decision: no-go.** The candidate is close to verifier fuzzing, perturbation testing, and existing RLVR verifier-error audits. VARE has no fresh workload with trusted false-accept labels, so a small synthetic result would not separate the proposal adequately.

## Candidate 3 — budget-aware audit allocation for GRPO update decisions

1. **Exact research question:** Under a fixed clean-label budget and conditional verifier shift, can a prespecified group-level audit allocation reduce the probability of choosing the wrong GRPO update direction versus uniform or full-group audits?
2. **Closest existing work:** Classical cost-adjusted Neyman allocation and sequential testing cover the sampling objective. VARE's existing [non-oracle audit-allocation analysis](grpo-audit-cost-reduction-v1-report.md), [partial-label estimator study](grpo-partial-audit-estimator-v1-report.md), and [policy-shift sign-risk study](grpo-policy-shift-audit-risk-v1-report.md) already analyze closely related costs, MSE, and direction risk.
3. **What is genuinely different?** A candidate distinction would require a new estimand or a prospective, independently labeled deployment distribution with an operational decision boundary. Re-running the existing finite synthetic comparisons or changing their allocation after seeing results is not new.
4. **Falsifiable hypothesis:** On a preregistered set of fresh conditional reward laws at matched labeling cost, the allocation reduces update-direction error while satisfying a frozen worst-case error ceiling against uniform audit.
5. **Smallest decisive experiment:** Freeze a new law generator and independent enumerator, tune allocation only on development laws, then compare methods on disjoint confirmation laws at identical label budgets; later confirm on a real task with trusted labels before making a practical claim.
6. **Independent outcome measure:** Exact sign error and estimator MSE from an independently implemented enumerator; on a real task, sign agreement against fully audited groups.
7. **Necessary compute and data:** CPU is sufficient for finite enumeration. A practical claim additionally needs fresh reward laws or a legally usable real task with full trusted group labels.
8. **Failure criteria:** No frozen sign-error gain at equal cost, any violation of the worst-case ceiling, an independent enumerator mismatch, or an effect confined to the old generator.
9. **Expected contribution:** Positive finite results could establish a narrow sampling property; negative results could reject the allocation rule. Neither can be described as a new method without a genuinely different estimand and independent real-task confirmation.

**Decision: no-go.** The topic is already covered by retained VARE analyses and standard allocation theory. A new synthetic sweep would add little; the missing real-task labels cannot be created without a new, gated task study.

## Work boundary and remaining uncertainty

No training, model inference, GPU, MPS, paid API, cloud compute, or new dataset use was performed in this candidate screen. It is a literature-and-evidence triage, not an experimental result or exhaustive review. Primary-paper scope and current publication status can change; the linked papers should be rechecked before any future protocol is frozen. The next research gate requires a materially distinct question, a legal unused task dataset, an independent checker, and a base model that clears a prespecified feasibility screen.
