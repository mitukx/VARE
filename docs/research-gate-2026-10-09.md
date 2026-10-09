# Research gate — post-training contribution screen (2026-10-09)

## Decision

**STOP both screened mechanism proposals before implementation.** Neither survives a novelty gate, and the repository has no currently eligible model/task/update path for a new capability experiment. This is a negative novelty and feasibility result, not evidence that the underlying problems are solved. No training data, sealed evaluation labels, or new model outputs were consumed for this decision.

The next action is to keep the learner gate closed until either (a) an unused model/task pair has an independently justified path through base success, CPU update, save/reload, and fresh held-out evaluation, or (b) an external reviewer identifies a concrete uncovered claim in a retained evidence packet. This report does not authorize a nearby prompt or parser variant on any retired pairing.

## Scope and evidence inspected

The current portfolio decision, `AGENTS.md`, current gaps, roadmap, and the StrategyQA feasibility report were reviewed. Relevant retained results include the negative three-seed ARC-Challenge GRPO/SFT comparison, its post-hoc outcome-signal diagnostic, the failed StrategyQA base gate, and the stopped GRPO partial-audit allocation studies. The cache inventory and the retirement decisions in the active decision record were also checked. No repository code or experiment protocol was changed.

The retained model evidence does not provide a fresh candidate: ARC-Challenge/Qwen2.5-0.5B-Instruct, StrategyQA/Qwen2.5-0.5B-Instruct, Qwen Math/TIR, calendar, code repair, ticket-tool, and earlier task cohorts are retired or failed their frozen gate. Cached Gemma 2 2B has no measured task/update path; cached One-Shot-RLVR Qwen2.5-Math-1.5B is a third-party math checkpoint with training/task overlap risk; neither is a justified new pair. This is consistent with the repository's rule against another base-only screen without a distinct feasibility basis.

## Candidate 1 — recover learning signal from homogeneous GRPO groups

### Precise question

For binary correctness reward and group size \(G\), can an outcome-level advantage that sends signal through all-correct and all-incorrect groups improve policy learning without introducing an unacceptable verifier-error bias?

### Exact mechanism

Let \(R_i\in\{0,1\}\) be correctness for a completion in a group and \(K=\sum_i R_i\). With the usual centered advantage, \(A_i=R_i-K/G\). If \(K=0\) or \(K=G\), then every \(A_i=0\). Under conditionally i.i.d. Bernoulli outcomes with success probability \(p\), the probability of a zero-signal group is

\[
  \Pr(K\in\{0,G\}) = (1-p)^G+p^G.
\]

That algebra matches the signal-coverage pattern in VARE's retained ARC diagnostic; it does not establish why GRPO underperformed SFT there. A fixed-reference sign advantage \(A_i=2R_i-1\) makes homogeneous groups nonzero, but changes the objective from within-group comparison toward absolute success/failure information.

### Prior-art result and novelty boundary

This exact mechanism and intervention are already directly treated in [Gradient Starvation in Binary-Reward GRPO](https://arxiv.org/abs/2605.07689). The paper proves the degeneracy phenomenon, derives the pass@G interpretation of fixed-reference Sign advantages, and reports a multi-seed Qwen3.5-9B/GSM8K comparison. VARE's ARC diagnostic is therefore a small retrospective consistency check, not a distinct contribution. The same family of issues is also adjacent to GRPO finite-group estimator analyses, including the U-statistic treatment in [Demystifying GRPO](https://arxiv.org/abs/2603.01162), and baseline/normalization analyses such as [Shrinking the Variance](https://proceedings.mlr.press/v306/zeng26z.html).

Adding verifier noise to this already-published intervention would not alone establish novelty. Existing work directly analyzes noisy-verifier correction and selective control, and VARE's audit-allocation results already reduce to established design/testing theory. No distinct theorem or falsifiable advantage over those methods was identified in this bounded review.

**Decision: reject as a new VARE algorithm.** Do not rerun ARC or tune its consumed cohort.

## Candidate 2 — non-oracle audit allocation under policy-induced verifier shift

### Precise question

Can group-level higher-order clean-label audits, selected without an oracle, reduce wrong-sign GRPO updates at a matched total budget that counts rollout generation and audit labels, when behavior and target policies differ?

### Reduction and prior-art result

The retained VARE estimator is Horvitz–Thompson for a selected joint-label event. Under known inclusion probabilities its variance allocation reduces to cost-adjusted Neyman/optimal design; sequentially choosing a sign after observations reduces to sequential testing. The frozen VARE sensitivity study already reports matched-budget sign risks and an independent multinomial audit, and found no deployable non-oracle design that creates a distinct method. Its two-world witness also shows why behavior-policy importance weights cannot restore a clean-label component absent from observations.

For policy shift in the conditional verifier-error law, action importance weights alone transport \(P(A)\), not \(P(R\mid A)\). If that conditional law is unrestricted and no clean labels cover target-relevant strata, the target clean-gradient sign is not identified. VARE's parity-shift counterexample demonstrates this limit. It is a diagnostic impossibility boundary, not a new audit algorithm.

This direction also sits directly beside [Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers](https://arxiv.org/abs/2510.00915), whose revised analysis includes trajectory-dependent error residuals and group-centered updates; [Audit-First VAPO](https://arxiv.org/abs/2609.33662), which studies risk-certified selective updates; [Verifier Errors in RLVR](https://arxiv.org/abs/2609.35677), which analyzes reward hacking and selective control; and [VStress](https://arxiv.org/abs/2609.36958), which studies correlation-aware verification cost. The screening did not find an assumption regime or guarantee not already covered by those works or VARE's existing negative results.

**Decision: stop this branch.** Do not produce another variant of the existing Walsh witness or call a synthetic audit-risk change a policy-learning improvement.

## Real-model feasibility and portfolio impact

The open evidence gap remains held-out task success after a real model update. The nearest completed learning comparison used the now-retired ARC-Challenge pairing: GRPO mean exact match was 38.84%, successful-trace SFT was 40.00%, and the paired GRPO-minus-SFT 95% task-bootstrap interval was \([-5.22,+2.90]\) percentage points. Its advancement gate failed. StrategyQA's frozen Qwen screen parsed 0/200 outputs, so it stopped before training; this is a format/task-interface failure, not evidence of zero semantic accuracy. Existing reports preserve both outcomes.

The cache contains model files, but file presence does not satisfy the required task-skill, CPU update, save/reload, and independent held-out gates. Under the current no-spend constraint and frozen retirement rules, a new learning run would either be a nearby repeat or lack a defensible capability metric. No model update was attempted.

## Research decision

**STOP the two screened mechanisms; continue only on a materially distinct path.** This is not a claim that VARE has produced an original frontier-RL method. It records that the most plausible nearby ideas are now known prior art or reduce to the existing theory, and that the current model/task portfolio does not justify another learning run.

The cheapest decisive next input is a new, unused task/model pairing with a frozen independent evaluator and a demonstrated CPU update/save/reload path, or an external technical review of a retained substantive packet. Do not spend compute merely to keep the research loop active.

## Reproduction and source links

This decision uses retained repository artifacts only; it has no new experiment runner. To inspect the quantitative model outcomes, see [`qwen-arc-grpo-sft-comparison-v3-report.md`](qwen-arc-grpo-sft-comparison-v3-report.md), [`cpu-strategyqa-grpo-shift-feasibility-v2-report.md`](cpu-strategyqa-grpo-shift-feasibility-v2-report.md), [`grpo-audit-cost-reduction-v1-report.md`](grpo-audit-cost-reduction-v1-report.md), and [`grpo-policy-shift-audit-risk-v1-report.md`](grpo-policy-shift-audit-risk-v1-report.md). The analysis above is a novelty/portfolio decision, not a new empirical result.
