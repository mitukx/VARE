# Fresh research-candidate qualification — 2026-10-15

## Decision

**No-go for a model screen, training run, or GPU request.** No candidate currently
has both a defensible distinction from recent work and a fresh task with an
independent evaluator. This continues the frozen Stage 0 decision; it does not
mean the research problem is solved. The existing real-model learning gap is
held-out task success after a post-training update.

No checkpoint was loaded, and no inference, optimizer update, or training run
was performed. A broad read-only search incidentally printed a few archived
output rows from a retired exact-integer run; they were not re-scored or used
as evidence in this candidate decision. That cohort remains retired. The
qualification itself used the repository's experiment/retirement ledger and
relevant primary-paper abstracts as of 2026-10-15.

## Objective used for this decision

VARE's technical objective is to identify a post-training signal or
intervention that improves an independently measured held-out outcome under a
fixed compute budget, while making the update, reward provenance, and
evaluation trustworthy. A new wrapper, an E0 harness fix, training reward,
lower KL, or an implementation of a recent paper is not by itself a research
result. Any future small-model experiment must clear a fresh-task base gate
before a separate update/save/reload smoke and a multi-seed comparison.

## Candidate review

| Candidate line | Current distinction and evidence | Decision |
| --- | --- | --- |
| Fixed-rollout reuse, policy drift, and GRPO collapse | Existing VARE evidence is confounded and its OpenBookQA pairing is retired. Recent work directly studies off-policy reuse, stale-rollout stability, and correction; see the [Stage 0 review](stage0-research-gate-2026-10-11.md). | Do not reopen on the old cohort or claim a new general method. |
| Verifier error and audit correction | VARE's previous synthetic audit results do not provide fresh independently labeled errors. *Verifier Errors in RLVR* studies fixed-verifier reward hacking and audit-based selective control; *VStress* studies correlation-aware adaptive allocation of repeated verifier calls; *Where the Verifier Fails* uses metamorphic transformations to audit category-specific verifier false negatives. A new VARE claim needs a distinct estimand and fresh labeled workload, beyond verifier auditing or generic selective control. | No candidate selected. |
| RSI through skill libraries and sequential task chains | *Reinforcement Learning for Self-Improving Agent with Skill Library* (SAGE) studies skill accumulation across sequential tasks. Newer *Skill-V* represents skills as versioned falsifiable contracts, revises boundaries from environment outcomes, and checks historical non-regression. Reimplementing skill accumulation, contract-based revision, or replay-based regression checks would be a reproduction, not differentiation. | Do not present a skill-library or contract-revision port as a novel VARE method. A future candidate must identify a different, falsifiable failure mode under sequential selection. |
| Long-horizon branching and trajectory processing | *QwenGyre* addresses elastic GPU allocation and long-horizon trajectory processing. *SPROUT* adds filesystem/context snapshots, verifier-guided backtracking, and resumed branching, reporting results with Qwen3.8-27B. These author-reported systems results are not a CPU-feasible VARE learning question; this is a scope judgment, not an independent reproduction. | Treat as related systems work; do not port GPU scheduling or snapshot infrastructure to M1 without a distinct measured bottleneck. |
| Language drift and reasoning-trace monitorability | *On Language Drift during RLVR Post-Training* analyzes drift theoretically and reports multi-model RLVR experiments. Its abstract warns that constraining drift can also constrain expected reward, so a simple legibility penalty is not an established free improvement. The reported study scale is outside the local CPU budget, and VARE has no qualifying fresh model/task pair. | Do not add a trace-legibility reward or launch a local replication on existing retired cohorts. |
| New base model on a previously used task family | Changing the checkpoint alone does not create a fresh research question or task composition. Existing negative and retired cohorts remain unavailable for prompt tuning, training, or confirmation. | No-go absent a separately justified fresh task and method. |

The SAGE and verifier papers close two tempting RSI/RL directions more directly
than the prior 2026-10-11 screen recorded. QwenGyre is relevant as a systems
reference, but its GPU scheduling contribution is outside the current local
resource envelope. These comparisons are bounded; they are not an exhaustive
literature review.

## Literature update — 2026-10-15

The candidate table was refreshed against primary arXiv abstracts for four
closely adjacent papers that were missing from the initial screen:
[Skill-V](https://arxiv.org/abs/2610.11781) (submitted 2026-10-08) tests and
revises existing skills as versioned, falsifiable contracts with historical
non-regression; [SPROUT](https://arxiv.org/abs/2610.14405) (2026-10-09)
checkpoints filesystem and agent context to support verifier-guided resumption
and branching; [VStress](https://arxiv.org/abs/2609.36958) (2026-09-29)
allocates repeated-verifier calls using conditional information and a cost
ledger; and [Where the Verifier Fails](https://arxiv.org/abs/2609.01354)
(2026-09-01) audits category-level verifier false negatives with certified
metamorphic answer transformations. The abstracts report promising results,
but those author claims are not independent reproductions. Their overlap
narrows the simple VARE ports that can be presented as distinct contributions.

The same refresh includes [On Language Drift during RLVR Post-Training]
(https://arxiv.org/abs/2610.02015, submitted 2026-10-01). Its abstract reports
a theoretical trade-off between constraining language drift and expected
reward. That result argues against assuming a monitorability reward is a free
improvement; it does not authorize a local replication or override the
fresh-task gate. The no-go decision above is unchanged.

## What must qualify before the next model run

The next research action is candidate qualification, not inference. A proposed
pair may enter a base-only feasibility protocol only after its record states:

1. A falsifiable intervention and the exact claim that differs from the
   closest methods above and the repository's retired lines.
2. A legally usable, unused task composition with a known-failing baseline,
   an independent protected evaluator, and a separate confirmation pool.
3. A base-success and reward-variation threshold fixed before inference, with
   an abort rule and no prompt/parser tuning after seeing outcomes.
4. A measured local inference/update/save/reload budget that fits the M1
   resource ceiling without paid services.
5. A later matched comparison against unchanged base and SFT controls, with
   multiple seeds and independent held-out task success as the primary
   outcome; reward, KL, and parser rate remain secondary diagnostics.

Passing the base gate would authorize only a distinct update-cost smoke. It
would not authorize a training comparison or establish capability. If no
candidate satisfies these conditions, the charter's fallback remains an
independent reproduction of retained evidence or a concrete newly reproduced
trainer defect; do not add generic infrastructure to fill the gap.

## Primary literature checked

- [Verifier Errors in RLVR: Reward Hacking, Limits of Feedback, and Selective Control](https://arxiv.org/abs/2609.35677), submitted 2026-09-28.
- [VStress: Correlation-Aware Auditing and Adaptive Budget Allocation for Repeated Verifiers](https://arxiv.org/abs/2609.36958), submitted 2026-09-29.
- [Where the Verifier Fails: A Category-Level Audit of Reward Signals in RLVR](https://arxiv.org/abs/2609.01354), submitted 2026-09-01.
- [Skill-V: Verifiable Self-Evolving Skill Library for Interactive Agents](https://arxiv.org/abs/2610.11781), submitted 2026-10-08.
- [SPROUT: Scaling Agentic Reinforcement Learning for Long-Horizon Software Engineering](https://arxiv.org/abs/2610.14405), submitted 2026-10-09.
- [On Language Drift during RLVR Post-Training](https://arxiv.org/abs/2610.02015), submitted 2026-10-01.
- [QwenGyre: An Elastic Reinforcement Learning Framework for Training xLong-Horizon Agents](https://arxiv.org/abs/2609.33848), submitted 2026-09-27.
- [Reinforcement Learning for Self-Improving Agent with Skill Library](https://arxiv.org/abs/2512.17102), revised 2026-03-10.
- The related stale-rollout and off-policy papers recorded in the [2026-10-11 Stage 0 gate](stage0-research-gate-2026-10-11.md).
