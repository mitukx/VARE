# Action-conditioned GRPO label omission — exploratory screen

## Question

Can a structured choice of which reward label to omit recover the four-member group-relative update with lower clean-label cost error than uniform 3-of-4 auditing or full-group auditing?

This was an **exploratory development screen**, not a preregistered confirmation. The candidate rule was fixed from the GRPO update's known Walsh coefficients, without using clean labels:

\[
q_j(a)=\frac1{16}+\frac34\frac{|\alpha_{[4]\setminus\{j\}}(a)|}{\sum_k|\alpha_{[4]\setminus\{k\}}(a)|},
\]

where \(J=j\) is the response omitted from a three-label group audit. If all coefficients vanish, omissions are uniform. The minimum probability is 1/16. The inverse-inclusion estimator is pointwise unbiased over the omission randomization for every fixed action/reward group.

## Screen and result

The development generator produced 128 synthetic conditional reward laws under four families: arbitrary random joint laws, conditionally independent labels, common-shock mixtures, and a single action-stratum parity tilt. The finite action/reward/subset outcomes were exhaustively enumerated in double precision; each law's probabilities are retained as exact rational numerators and denominators. At each arm, the total budget was 12 clean labels:

- 3 complete groups for full-group audit;
- 4 groups with uniform three-of-four omission;
- 4 groups with coefficient-weighted omission;
- one- and two-label Walsh projections as biased low-order references.

All 128 laws had nonzero full-audit MSE. The structured design beat full-group MSE in **0/128** laws and was at least 10% better in **0/128**. Its median structured/full fixed-budget MSE ratio was **1.159** and mean ratio **1.334**. It beat uniform 3-of-4 in **0/128**; the median structured/uniform MSE ratio was **1.140**. Uniform 3-of-4 itself had a median full-audit ratio of **1.013** on this new panel.

A preliminary single-law check on a separate development seed had already pointed the same way (structured/full 1.161; uniform/full 1.017); it was not used as confirmation and is not part of the 128-law summary. The full panel's outcomes and generating source snapshot are retained under [`development-seed-20261016`](../results/grpo-action-conditioned-omission-v1/development-seed-20261016/).

## Interpretation

The structured rule preserved unbiasedness but its inverse-probability weighting increased variance. In this frozen synthetic screen it was worse than both uniform subset selection and full groups at matched label cost. This rejects this particular coefficient-weighted omission rule; it does not prove that every action- or history-conditioned design is inferior under every verifier law.

The result is not confirmatory: the rule, generator, and decision thresholds were not committed together before calculation, and the family mix is synthetic. A post-hoc same-host auditor now independently reconstructs the metrics from the retained exact-rational panel and matches all 128 per-law rows within (2.7\times10^{-15}). It checks pointwise unbiasedness over 32,768 action/reward/law cases. This audit does **not** establish the historical source-to-summary provenance or constitute external replication. No direction-error simulation, verifier accuracy on natural tasks, optimizer step, policy improvement, or capability outcome was measured. Do not claim a deployable estimator or a general sample-efficiency result.

## Prior-art and novelty assessment

Generic adaptive verifier allocation is already covered by [VStress](https://arxiv.org/abs/2609.36958), which selects repeated verifier views using conditional information and cost under fixed budgets. Its object is additional verifier channels per item, not the clean-label coordinates required by a group-normalized GRPO update. [Audit-First VAPO](https://arxiv.org/abs/2609.33662) selects whether to admit a proposal direction under finite audit budgets and certifies a finite-stage selected-risk target; it explicitly leaves nonlinear parameter trajectories and long-horizon utility to separate evaluation. [Verifier Errors in RLVR](https://arxiv.org/abs/2609.35677) studies partial correctness audits and projected correction under a no-false-negative assumption, while warning that finite-sample gradients, Adam, and finite steps can invalidate continuous-flow guarantees. Earlier [noise-corrected GRPO](https://arxiv.org/abs/2510.18924) and [group-relative advantage bias](https://arxiv.org/abs/2601.08521) work also make broad claims about noise correction or finite-group bias unsuitable here.

The only narrow differentiator tested here is the marginal-variance effect of an action-conditioned omission rule for this fixed four-member GRPO estimand. Its exploratory result is negative. **No algorithmic novelty is established.**

VStress's arXiv preprint is relevant prior art but its reported numerical results have not been reproduced here. Two table-level items merit checking against its underlying artifact before relying on them: the symmetric-35 majority row reports different coverage across tables despite the same balanced accuracy, and the adaptive row's rounded items-times-calls figure is above the stated exact budget. These are reproduction questions, not claims about the paper's validity.

## Decision

**STOP** this coefficient-weighted omission rule and the generic partial-label efficiency line. Do not search more weights against this or the prior v1 grid. The existing v1 exact result and this exploratory screen together show that the tested unbiased subset designs did not improve fixed-budget MSE. The next justified work is the mandated pivot: establish a materially distinct cached base-model/task/update path with independent task-success evaluation. The latest local candidate scan found no such pair clearing the CPU feasibility gates; do not open a scored cohort until that changes. A controlled finite-step risk-to-clean-outcome study remains a separate possible theory task, not evidence of real-model improvement.

## Reproduction and provenance

The [raw summary](../results/grpo-action-conditioned-omission-v1/development-seed-20261016/summary.json) includes all per-law moments and ratios. The [law panel](../results/grpo-action-conditioned-omission-v1/development-seed-20261016/law_panel.json) stores exact rational probabilities. `runner-as-run.py` retains the calculation helpers used during development, but the one-off inline driver that assembled the historical summary was not saved; therefore, byte-identical source-to-summary provenance cannot be established. The [development entry point](../scripts/dev_grpo_action_conditioned_omission_v1.py) is a later retained helper, not a replay claim. The [post-hoc auditor](../scripts/audit_grpo_action_conditioned_omission_development_v1.py) recomputes metrics independently from the panel and checks them against the summary, without importing the development runner.

No GPU, model, paid API, third-party data, or additional dependency was used. The same-host post-hoc audit passed; no external replication or CI run is claimed for this exploratory analysis.
