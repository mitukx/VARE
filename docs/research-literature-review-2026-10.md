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

Proceed with the frozen finite counterexample and matched-label-budget CPU experiment. Do not pursue a broad adaptive audit allocation method: VStress covers much of that framing. Do not present this scan as evidence of a new algorithm or a model improvement. Keep verifier/future-version provenance validation as a separate concrete correctness issue for later work.
