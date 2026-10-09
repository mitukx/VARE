# RLVR and verifier research scan — 2026-10-09

## Scope

This scan focused on primary papers from 2024–2026 about GRPO, noisy verifiers, verifier exploitation, policy/verifier drift, reward modeling, and adaptive audit allocation. It was used to select one CPU-only question. It is a focused scan, not a systematic review, and it does not establish novelty.

## Findings and overlap

| Area | Relevant work | Implication for VARE |
| --- | --- | --- |
| GRPO | [DeepSeekMath](https://arxiv.org/abs/2402.03300) introduced group-relative policy optimization for mathematical reasoning. | A GRPO reward group is the comparison unit; member provenance alone does not guarantee its joint reward signal is observable. |
| Noisy and correlated verifier rewards | [Cai et al.](https://arxiv.org/abs/2510.00915) derive corrections for asymmetric false-positive/false-negative reward channels. [Plesner et al.](https://arxiv.org/abs/2604.07666) test several noise structures, including errors shared across an entire rollout group, and derive conditional advantage distributions. | Generic asymmetric-noise correction and generic group-correlated reward noise are already studied. VARE should not claim either as new. Plesner et al. analyze training under specified corruption; the present question instead concerns whether a limited partial-label audit can identify the latent joint clean reward signal. This is a narrow distinction, not proof of novelty. |
| GRPO estimator behavior | [Your Group-Relative Advantage Is Biased](https://arxiv.org/abs/2601.08521) analyzes difficulty-dependent bias in group-relative advantages. | A new claim about GRPO advantage bias would overlap. The current study concerns missing joint audit observations, not difficulty reweighting. |
| Verifier exploitation | [LLMs Gaming Verifiers](https://arxiv.org/abs/2604.15149) use isomorphic perturbation testing to catch extensional shortcuts. [Reward Hacking in Rubric-Based RL](https://arxiv.org/abs/2605.12474) separates verifier errors from underspecified rubrics and uses cross-family evaluators. | “Use another verifier” or “perturb the task” is not a new contribution. The current study makes no model-behavior or hack-detection claim. |
| Adaptive reward audits | [Adversarial Reward Auditing](https://arxiv.org/abs/2602.01750) frames reward hacking as an auditor–hacker process. [VStress](https://arxiv.org/abs/2609.36958) allocates repeated verifier calls using conditional information, cost, correlation, and a shift fallback, with matched-budget comparisons. | A general adaptive-audit allocator is already crowded and VStress is close. VARE's candidate is narrower: which *joint clean-label observations* are necessary to reconstruct the nonlinear normalized signal of one GRPO rollout group. |
| Capability versus proxy reward | [RLVR can Lead to Reward Hacking](https://arxiv.org/abs/2604.15149) and the rubric-based study above show that verifier-reward improvement can diverge from external evaluation. | VARE's repeated NLL/reward positives with task-success non-passes remain important negative evidence. None of the synthetic results in this repository establishes language-model capability. |

## Selected question

Can one clean-label audit per rollout group identify the distribution of the clean group-normalized GRPO advantage when the proxy reward is fixed but clean labels have different within-group dependence?

The question was selected because VARE recently added atomic replay handling for GRPO/RLOO groups. That protects training from accidentally incomplete groups, but it does not answer whether an *audit* that samples only individual group members can reconstruct the joint clean signal.

The exact two-world construction is in the [frozen protocol](../protocols/grpo_group_audit_identifiability_v1.lock.json). In both worlds, each group has proxy rewards `(1,0)`. In World A, clean rewards are `(Z,Z)` for a fair Bernoulli bit `Z`; in World B they are `(Z,1-Z)`. Either member's clean reward is Bernoulli(1/2) in both worlds. Therefore, a single randomly selected member per group has the same observation law under both worlds, even with known per-item propensities. But the clean normalized group advantage is always `(0,0)` in A and is `(1,-1)` or `(-1,1)` in B.

This is an identifiability counterexample about the *per-group normalized signal*. It does not show that the expected parameter gradient is unidentifiable in every policy/task, and does not show that group audits are generally cost-optimal. A future extension would need actual policy-gradient updates, larger groups, variable costs, and a stronger novelty review.

## Decision

The original frozen per-group signal counterexample and matched-label-budget simulation are complete. Do not pursue a broad adaptive audit allocation method: VStress covers much of that framing. Do not present the scan or either synthetic result as evidence of a new algorithm or a model improvement. Keep verifier/future-version provenance validation as a separate concrete correctness issue.

## Follow-up exact update witness

After the original two-member result, the next open question was whether partial labels fail to identify an actual policy update, rather than only the realized normalized reward vector. A three-member Bernoulli/noise proposal was rejected after exact enumeration showed its expected gradients were equal. A subsequent finite LP search found a four-member construction; its selection after search is disclosed in the protocol. Exact enumeration and a separate same-host implementation verify that the single-item observation law remains identical even conditioned on all group actions, while expected local clipped-GRPO gradients differ by about 0.0131. See the [exact-update report](grpo-expected-update-audit-identifiability-v1-report.md).

This sharpens the constructed estimand, not the novelty claim. Plesner et al. already study group-structured verifier noise and conditional advantages, while recent paired-rollout work distinguishes reward-contrast behavior from gradient variance. The present finite witness is about partial-audit identifiability under complete action context; whether this separation is sufficiently distinct for a research contribution remains unresolved. No general audit method or model-level result follows.

The wider scan also checked current work on: reward-bias substitution and policy-induced evaluation distributions ([RBS](https://arxiv.org/abs/2605.27996)); cross-family judging and rubric hacking ([rubric-RL](https://arxiv.org/abs/2605.12474)); RLVR verifier gaming and isomorphic perturbations ([IPT](https://arxiv.org/abs/2604.15149)); a controllable coding reward-hacking testbed ([CATCH](https://arxiv.org/abs/2609.39533)); and correlation-aware, cost-normalized verifier-call allocation ([VStress](https://arxiv.org/abs/2609.36958)). These make generic proxy-vs-quality audits, generic verifier exploits, and general adaptive audit allocation poor novelty targets without a substantially different estimand or independent empirical effect.

## Update — audit-order characterization and closer overlap

A deterministic follow-up now studies the minimum *marginal order* needed to identify one frozen four-member local clipped-GRPO update. The finite Boolean/Fourier criterion is general mathematics; the application finds degree three for all nonconstant actions in that specific estimator, so two-way marginals can fail while complete three-way marginals suffice. This is narrower than group-atomic auditing and does not yet establish cheaper estimation or a useful policy.

The new scan also adds [Audit-First VAPO](https://arxiv.org/abs/2609.33662), which uses risk-certified selective updates under imperfect verification, and [Verifier Errors in RLVR](https://arxiv.org/abs/2609.35677), which analyzes limits of verifier-only feedback and selective control. These make broad claims about partial-audit correction or selective update admission especially close. Plesner et al.'s revised paper explicitly analyzes group-noise structures and conditional advantages; Yang et al. study finite-group advantage behavior; VStress covers correlation-aware cost-normalized auditing. The audit-order theorem should therefore be described as a finite characterization applied to one estimator, not as a new general RLVR theory or audit method. Novelty is unresolved and no capability result follows.
