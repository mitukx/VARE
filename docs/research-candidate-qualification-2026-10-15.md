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
| Failure reflection, counterfactual memory, and trajectory-quality learning | *COUNTERMEM* executes and verifies counterfactual action alternatives, stores corrected actions with reuse conditions, and learns when to use or skip memory. *Training Language Agents to Learn from Experience* RL-trains reflectors for future unseen task families. *HeRL* turns failed trajectories and unmet rubrics into guided exploration. *TQG-Exp* adds process-quality feedback and paired rollouts with or without retrieved experience to revise experience utility. These works substantially overlap naive failure-reflection, counterfactual-memory, and trajectory-quality reuse proposals. | Do not claim a memory/reflection port as novel. A distinct estimand and unused task families would be required; no such CPU-qualified candidate is selected. |
| Self-verifiable task transformation and synthetic app worlds | *RLSVR/SpyRL* transforms open-ended tasks into self-play environments with verifiable rewards. *Terrarium* synthesizes executable app worlds and evaluator-side task rubrics, reporting model updates at 4B, 8B, and 14B scales. Task transformation, automatic world generation, or rubric synthesis alone is therefore not a distinct contribution, and the reported model scale is outside the local M1 budget. | No-go for a task-transformation or synthetic-world training port. Reconsider only with a clearly different estimand and an independently protected evaluator that fits the resource budget. |
| RSI through skill libraries and sequential task chains | *Reinforcement Learning for Self-Improving Agent with Skill Library* (SAGE) studies skill accumulation across sequential tasks. Newer *Skill-V* represents skills as versioned falsifiable contracts, revises boundaries from environment outcomes, and checks historical non-regression. Reimplementing skill accumulation, contract-based revision, or replay-based regression checks would be a reproduction, not differentiation. | Do not present a skill-library or contract-revision port as a novel VARE method. A future candidate must identify a different, falsifiable failure mode under sequential selection. |
| Long-horizon branching and trajectory processing | *QwenGyre* addresses elastic GPU allocation and long-horizon trajectory processing. *SPROUT* adds filesystem/context snapshots, verifier-guided backtracking, and resumed branching, reporting results with Qwen3.8-27B. *TQG-Exp* also makes trajectory quality part of experience selection and utility estimation. These author-reported systems and agent-learning results do not establish a CPU-feasible VARE learning question; this is a scope judgment, not an independent reproduction. | Treat as related systems work; do not port GPU scheduling, snapshot infrastructure, or trajectory-quality memory updates to M1 without a distinct measured bottleneck. |
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

The follow-up screen also checked six papers that close additional post-training
and RSI-shaped ports. [COUNTERMEM](https://arxiv.org/abs/2609.31874), submitted
2026-09-25, verifies counterfactual alternatives with executable world models,
stores corrected actions and reuse conditions, and trains a memory-use policy
while keeping the base LLM fixed. [Training Language Agents to Learn from
Experience](https://arxiv.org/abs/2605.20477), submitted 2026-05-19, RL-trains a
reflector to improve an actor on unseen task families. [HeRL](https://arxiv.org/abs/2603.20046),
submitted 2026-03-20, uses failures with unmet rubric items as exploration
guidance. The 2026-10-10 [TQG-Exp](https://arxiv.org/abs/2610.14877) paper
scores process quality, compares paired rollouts with and without retrieved
experience, and updates experience utility. Together these make a generic
"learn from failures" or trajectory-memory proposal too close to existing
work absent a different estimand and task family.

For task construction, [RLSVR/SpyRL](https://arxiv.org/abs/2607.23802),
submitted 2026-07-26, transforms open-ended tasks into self-play tasks with
verifiable rewards. [Terrarium](https://arxiv.org/abs/2610.14860), submitted
2026-10-10, synthesizes executable app worlds and evaluator-side rubrics; its
abstract reports training at 4B, 8B, and 14B model scales. These papers further
narrow the available task-transformation novelty and the latter's reported
scale is beyond the local compute envelope. All results above are author
reports from primary-paper abstracts; none was reproduced here. The no-go
decision remains unchanged.

An external task-pool screen identified [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench)
as a possible future source of post-checkpoint software tasks, not an approved
evaluation set. The examined Hub revision is pinned to
[`89cdfbab4ab1bd8f5a658bb212d1b63624f4f881`](https://huggingface.co/datasets/nebius/SWE-rebench/tree/89cdfbab4ab1bd8f5a658bb212d1b63624f4f881).
At that revision, the test split's recorded `created_at` values range from
2014-04-30 to 2025-04-30 20:14:11. The dataset card labels the collection CC
BY 4.0 and exposes base commits, task timestamps, per-repository license
fields, hidden-test patches, and hints. Any use must verify each source
repository's license and creation date against a fixed checkpoint, keep tests,
patches, and hints outside the model context, and pass an M1 base-only
feasibility gate first. Its execution cost and repository setup have not been
measured locally.
The [metadata-only snapshot](../results/research-candidate-qualification-2026-10-15/swe-rebench-snapshot-89cdfba/manifest.json)
retains revision-specific Hub API metadata, split/size metadata, test and filtered
statistics, endpoint URLs, and SHA-256 hashes. No parquet or task rows were
downloaded, and no base inference was run.

## Autonomous research planning and resource allocation — 2026-10-15

The mission also targets an outer control plane for autonomous RL research
engineering, so the novelty screen includes systems that choose research actions
and allocate experiment budgets. [PrimeScientist](https://arxiv.org/abs/2609.17846)
uses an executable plan tree and adaptive MCTS to select research directions and
resource investment across attempts, including AI research, systems, and code
optimization. Its paper frames the main budget in tokens and leaves separate
hardware-time constraints as future work. [MARS](https://proceedings.mlr.press/v306/chen26r.html)
already combines cost-constrained MCTS, modular ML-engineering work, and
cross-branch reflective memory. [Anytime Verified Agents (AVA)](https://openreview.net/pdf?id=JMDCMf7mlF)
allocates a bounded budget across search, sampling, verification, and tool use.
[RepairAgent](https://arxiv.org/abs/2403.17134) already lets a repair agent
interleave information gathering, repair-material search, and validation based
on prior attempts.

The evaluation target is also established territory. [RE-Bench](https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/)
measures AI research-engineering outcomes under wall-clock budgets.
[PostTrainBench](https://arxiv.org/abs/2603.08640) studies autonomous post-training
under bounded compute and reports research-integrity failures. [MMPostTrainBench](https://arxiv.org/abs/2610.05398)
explicitly evaluates iterative candidate selection and non-target regressions.
Within RL, [Knapsack RL](https://proceedings.mlr.press/v306/li26v.html) allocates
heterogeneous GRPO rollout budgets by expected learning value and cost, while
the [IsoCompute Playbook](https://proceedings.mlr.press/v306/cheng26n.html)
studies compute-optimal allocation among rollouts, problems, and update steps.
These are author-reported results; VARE has not reproduced them.

Consequently, generic UCB/MCTS selection, cost-aware research planning,
interleaving inspect/patch/test actions, time-bounded repair scoring, or uneven
GRPO rollout allocation are not VARE novelty claims. Merely adding CPU time to a
token budget is also insufficient because execution expense is already central
to MARS and wall-clock-bounded evaluation is established by RE-Bench. The
[QwenGyre](https://arxiv.org/abs/2609.33848) systems contribution centers on
elastic GPU scheduling and long-horizon trajectories, so it is a reference for
trajectory identity and accounting rather than a local M1 implementation target.

A possible **empirical diagnostic**, not a new-algorithm claim, is whether
strategy rankings change on fresh RL repair tasks when budgets charge measured
CPU time, tool calls, and protected non-regression confirmation. This remains
**no-go for execution** until VARE has an unused task pack with independently
verified defect, correct-fix, regression, and grader-tampering cases; genuinely
heterogeneous measured costs rather than padded costs; a fixed no-additional-fee
agent path; and a confirmation evaluator hidden from selection. Compare against
relevant existing cost-aware baselines under the same locked budget. If any
entry condition fails, do not implement a generic selector or report a campaign
result; continue with an independent reproduction of a concrete RL defect.

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
- [COUNTERMEM: World-Model Verified Counter-Factual Memory for Language Agents](https://arxiv.org/abs/2609.31874), submitted 2026-09-25.
- [Training Language Agents to Learn from Experience](https://arxiv.org/abs/2605.20477), submitted 2026-05-19.
- [From RLVR to RLSVR: Task Transformation Induces Self-Verifiable Rewards for Open-Ended LLM Self-Improvement](https://arxiv.org/abs/2607.23802), submitted 2026-07-26, revised 2026-07-31.
- [Experience is the Best Teacher: Motivating Effective Exploration in Reinforcement Learning for LLMs](https://arxiv.org/abs/2603.20046), submitted 2026-03-20.
- [Beyond Outcome Feedback: Trajectory-Quality-Guided Experience-based Learning](https://arxiv.org/abs/2610.14877), submitted 2026-10-10.
- [Terrarium: Growing Personal Assistants via Scalable Synthesis of App Worlds](https://arxiv.org/abs/2610.14860), submitted 2026-10-10.
- [PrimeScientist: Strategic Allocation of Research Effort in Autonomous Research](https://arxiv.org/abs/2609.17846), submitted 2026-09-15.
- [MARS: Modular Agent with Reflective Search for Automated AI Research](https://proceedings.mlr.press/v306/chen26r.html), ICML 2026.
- [Anytime Verified Agents: Adaptive Compute Allocation for Reliable LLM Reasoning under Budget Constraints](https://openreview.net/forum?id=JMDCMF7mLF), TMLR 2026.
- [RepairAgent: An Autonomous, LLM-Based Agent for Program Repair](https://arxiv.org/abs/2403.17134), submitted 2024-03-25.
- [RE-Bench: Evaluating Frontier ML Research Capabilities of Language Model Agents Against Human Experts](https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/), 2024-11-22.
- [PostTrainBench: Can LLM Agents Automate LLM Post-Training?](https://arxiv.org/abs/2603.08640), submitted 2026-03-09.
- [MMPostTrainBench: Benchmarking Autonomous Research for Multimodal Post-Training](https://arxiv.org/abs/2610.05398), submitted 2026-10-04.
- [Knapsack RL: Compute-Efficient Reinforcement Learning via Heterogeneous Rollout Allocation](https://proceedings.mlr.press/v306/li26v.html), ICML 2026.
- [IsoCompute Playbook: Optimally Scaling Sampling Compute for LLM RL](https://proceedings.mlr.press/v306/cheng26n.html), ICML 2026.
- [SWE-rebench dataset card](https://huggingface.co/datasets/nebius/SWE-rebench/tree/89cdfbab4ab1bd8f5a658bb212d1b63624f4f881), inspected 2026-10-15; candidate task source only, not approved or executed. The metadata/statistics capture and hashes are in [`swe-rebench-snapshot-89cdfba`](../results/research-candidate-qualification-2026-10-15/swe-rebench-snapshot-89cdfba/manifest.json).
- [QwenGyre: An Elastic Reinforcement Learning Framework for Training xLong-Horizon Agents](https://arxiv.org/abs/2609.33848), submitted 2026-09-27.
- [Reinforcement Learning for Self-Improving Agent with Skill Library](https://arxiv.org/abs/2512.17102), revised 2026-03-10.
- The related stale-rollout and off-policy papers recorded in the [2026-10-11 Stage 0 gate](stage0-research-gate-2026-10-11.md).
