# VARE-TRACE progress report — 2026-11-02

## Decision

This change set is an engineering milestone for source-grounded navigation, safe edit recovery, and stricter long-horizon success accounting. It does **not** establish a new coding-capability gain, a new real-repository model result, or a VARE-TRACE algorithmic contribution.

The research run is **NO-GO** at the task-eligibility gate. No new task was source-qualified in this pass, so no new model inference, training update, held-out evaluation, or real-workload verifier-drift experiment was launched. This preserves the existing exposure and grading boundaries.

## Starting evidence retained

PR #180 remains the prior scientific result and is unchanged by this report. Its baseline is recorded in [`docs/vare-ig-flagship-research-2026-11-02.md`](vare-ig-flagship-research-2026-11-02.md) and its evidence tree:

- The coding pack contained 26 synthetic contracts; all 26 reference controls passed and all 26 incorrect mutants were rejected.
- Uniform and adaptive scheduling each found 3 informative groups in 18. The adaptive arm used more tokens and measured generation and verifier cost per informative group.
- Each arm made three real optimizer updates across two seeds. Matched seeds produced identical adapter digests.
- Development performance was 2/4. The base and trained policies both had 0/4 full long-horizon successes under the corrected success rule.
- Prefix reuse measured about 1.0337× on a narrow two-prompt workload. This does not establish a general inference speedup.

The saved Qwen3-4B Development attempt on vLLM #64575 remains a consumed, unsuccessful task exposure ([saved trace](../results/mlsys-fresh-vllm-64575-qwen3-4b-development-agent-v2/attempt-001/agent-attempt-002.json)). Its trace contains seven related empty literal searches and a malformed patch (`error: corrupt patch at <stdin>:17`). The new regressions replay that evidence locally; they are not a second model attempt or a capability result.

## Research scope and prior art

The current adaptive scheduler does not support a novelty claim. DAPO already filters zero-variance groups and changes the GRPO objective; Online Difficulty Filtering, VADE, and CoDaPO study online or competence-aware task selection and update weighting. Agent-search work on Lazy Likelihood Displacement also reports policy collapse mechanisms in tool-integrated RL. These papers make another synthetic difficulty-scheduler comparison a weak next experiment.

Asynchronous rollout throughput and staleness control are also established topics. AReaL, StaleFlow, DORA, and PACE address asynchronous execution, stale trajectories, or throughput coordination. The plausible VARE research gap is narrower: independently graded repository-repair trajectories whose task, candidate source, behavior policy, verifier version/configuration, and final reward remain bound across delayed grading, policy changes, and recovery. This is a research framing, not a validated novelty claim; it needs real eligible tasks and measured learning/system outcomes.

## Task eligibility and safety gate

The retained [deep-qualification report](mlsys-task-deep-qualification-v1-report.md) contains 15 source-level regression tasks split 3 Train / 3 Development / 9 Held-out. A retrospective exposure audit found eight prior inference attempts across all three Development task families, leaving zero fresh Development tasks in that split. The nine Held-out source tasks remain unused, but the prior qualification report says VARE's narrow function grader does not yet provide the required hidden-upstream-pytest boundary for those tasks.

The [metadata-only candidate screen v4](../results/mlsys-fresh-development-candidate-screen-v4/evidence-manifest.json) contains 3,305 candidates, including 1,278 provisional rows. Its frozen record explicitly sets qualification, model inference, and training authorization to false. Candidate rows are therefore leads for a new append-only qualification protocol, not eligible tasks.

One read-only lead surfaced during follow-up: [vLLM #64674](https://github.com/vllm-project/vllm/pull/64674), an incremental decoder for Responses custom-tool streaming, merged on 2026-10-31. It is unqualified here; no source patch, test tree, or model run was used, and its tool-API family overlap with the consumed #64575 Development task still needs review. **New source-qualified tasks this pass: 0.** No Held-Out identity was accessed.

The current attempt therefore stops before model inference. It does not relabel an exposed task as fresh, weaken an upstream test, or claim that an unqualified candidate is a task.

## Engineering changes

- Added an AST-only Python symbol index and a repeated-empty-search guard. After three related empty literal searches, the controller offers symbol and bounded-range navigation; later related searches are intercepted until new source context is read or changed. The symbol index parses text and never imports candidate modules.
- Added optional path-scoped literal search and SHA-256-bound exact replacement for new navigation-enabled runs. Replacement checks the bytes read by the agent and verifies the exact postimage.
- Added a bounded malformed-edit retry intervention that preserves the parser's concrete error, verifies whether source bytes changed, and does not fabricate a successful tool call or run a grader for a rejected edit.
- Saved attempts now record whether navigation tools were enabled. Retrospective context-compaction replay reconstructs the old frozen tool schemas for older attempts and the expanded schemas only for attempts that recorded the new flag. The legacy default schemas remain unchanged for frozen campaign protocols.
- Long-horizon episode schema v2 now binds trusted final success to a terminal episode, all expected committed steps, the trajectory head, registered verifier configuration, the graded candidate's final source hash, and the task/source/workspace identity. Configured public tests must also pass. Legacy v1 success retains its prior checks.

These changes improve protocol integrity and address a failure mode observed in a consumed Development trace. They have not yet shown that a model uses the navigation affordances or repairs a real repository task.

## Validation

Passed locally:

- 18 relevant pytest modules covering the agent tool format, long-horizon evaluator, frozen Development campaign protocols, trajectory store/provenance/admission/learning/promotion/safety, rollout task binding/evaluation identity, verified rollout replay, and verifier refresh.
- Repository CI Ruff target set: all checks passed.
- `mypy src/vare`: no issues in 99 source files.
- `git diff --check`: clean.

The passing tests validate code behavior and protocol compatibility. They are not model inference, task qualification, or evidence of policy improvement.

## Separate estimands

| Estimand | Current evidence | Status |
| --- | --- | --- |
| Capability | Prior #180 Development 2/4; prior full long-horizon success 0/4 in both arms; no new eligible tasks | No new estimate |
| Training systems | Prior matched updates and recovery checks only; no new trajectory-derived update | No new estimate |
| Long horizon | Controller and final-success accounting regressions only; no fresh repository cohort | No new estimate |

## Next evidence gate

Before another model run, create and freeze a new append-only source-task protocol. It must include prior-exposure review, exact source/test/license identities, baseline failure, known-fix success, wrong-mutant rejection, family deduplication, and a grader boundary that keeps upstream hidden tests and golden fixes outside the candidate workspace. Only then assign tasks to disjoint splits and run a bounded Development feasibility study. Training remains gated on complete behavior-policy data (including per-decision probabilities), trusted rewards bound to exact trajectory and verifier versions, and independent task success. No Held-Out access is authorized until the Development gate and analysis are frozen.

## References

- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Online Difficulty Filtering for Reasoning-Oriented Reinforcement Learning](https://aclanthology.org/2026.eacl-long.30/)
- [VADE: Variance-Aware Dynamic Sampling via Online Sample-Level Difficulty Estimation](https://arxiv.org/abs/2511.18902)
- [CoDaPO: Confidence and Difficulty-Adaptive Policy Optimization](https://arxiv.org/abs/2606.07950)
- [On GRPO Collapse in Agent Search: Lazy Likelihood Displacement](https://arxiv.org/abs/2512.04220)
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System](https://arxiv.org/abs/2505.24298)
- [StaleFlow: Staleness-Constrained Rollout Coordination](https://arxiv.org/abs/2601.12784)
- [DORA: A Scalable Asynchronous Reinforcement Learning System](https://arxiv.org/abs/2604.26256)
- [PACE: Pool-Aware Control of Effective Staleness](https://arxiv.org/abs/2609.36830)
