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
| Verifier error and audit correction | VARE's previous synthetic audit results do not provide fresh independently labeled errors. *Verifier Errors in RLVR* characterizes fixed-verifier error and proposes audit-based selective control with model experiments. A new claim would need a distinct estimand and a fresh labeled workload. | No candidate selected. |
| RSI through skill libraries and sequential task chains | *Reinforcement Learning for Self-Improving Agent with Skill Library* (SAGE) directly studies skill accumulation across sequential tasks and adds skill-integrated reward, including AppWorld experiments. Reimplementing this in VARE would be a reproduction, not differentiation. | Do not present a port as a novel VARE method. |
| Long-horizon branching and trajectory processing | *QwenGyre* addresses elastic GPU allocation, branching-trajectory reconstruction, partial-progress scoring, and deduplication for xLong-horizon RL. Its reported setup targets very large GPU-trained agents and does not furnish a CPU-feasible VARE learning question by itself. This is a scope-based feasibility judgment, not an independent reproduction of the paper. | Treat as related systems work; do not port the GPU scheduler to M1. |
| New base model on a previously used task family | Changing the checkpoint alone does not create a fresh research question or task composition. Existing negative and retired cohorts remain unavailable for prompt tuning, training, or confirmation. | No-go absent a separately justified fresh task and method. |

The SAGE and verifier papers close two tempting RSI/RL directions more directly
than the prior 2026-10-11 screen recorded. QwenGyre is relevant as a systems
reference, but its GPU scheduling contribution is outside the current local
resource envelope. These comparisons are bounded; they are not an exhaustive
literature review.

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
- [QwenGyre: An Elastic Reinforcement Learning Framework for Training xLong-Horizon Agents](https://arxiv.org/abs/2609.33848), submitted 2026-09-27.
- [Reinforcement Learning for Self-Improving Agent with Skill Library](https://arxiv.org/abs/2512.17102), revised 2026-03-10.
- The related stale-rollout and off-policy papers recorded in the [2026-10-11 Stage 0 gate](stage0-research-gate-2026-10-11.md).
