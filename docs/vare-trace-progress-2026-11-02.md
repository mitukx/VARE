# VARE-TRACE progress report — 2026-11-02

## Decision

This change set is an engineering milestone for source-grounded navigation, hash-bound edits, source-read handles, and stricter long-horizon success accounting. It does **not** establish a coding-capability gain, policy improvement, held-out performance, or a VARE-TRACE algorithmic contribution.

The bounded #64674 Development diagnostic ran once with source-read handles enabled. The model read the target source and referenced issued handles, but all three exact replacements were rejected because the submitted old text did not match the source. No source edit was applied, the public test was not invoked, and the independent grader was not run on the unchanged candidate. The result is an engineering diagnostic on an exposed Development task, not held-out or fresh-task evidence.

## Starting evidence retained

PR #180 remains the prior scientific result and is unchanged by this report. Its baseline is recorded in [`docs/vare-ig-flagship-research-2026-11-02.md`](vare-ig-flagship-research-2026-11-02.md) and its evidence tree:

- The coding pack contained 26 synthetic contracts; all 26 reference controls passed and all 26 incorrect mutants were rejected.
- Uniform and adaptive scheduling each found 3 informative groups in 18. The adaptive arm used more tokens and measured generation and verifier cost per informative group.
- Each arm made three real optimizer updates across two seeds. Matched seeds produced identical adapter digests.
- Development performance was 2/4. The base and trained policies both had 0/4 full long-horizon successes under the corrected success rule.
- Prefix reuse measured about 1.0337× on a narrow two-prompt workload. This does not establish a general inference speedup.

The saved Qwen3-4B Development attempt on vLLM #64575 remains a consumed, unsuccessful task exposure ([saved trace](../results/mlsys-fresh-vllm-64575-qwen3-4b-development-agent-v2/attempt-001/agent-attempt-002.json)). Its trace contains seven related empty literal searches and a malformed patch (`error: corrupt patch at <stdin>:17`). The new regressions replay that evidence locally; they are not a second model attempt or a capability result.

The first saved #64674 attempt remains unchanged. It used 1,434 generated tokens over 12 tool turns, omitted `expected_source_sha256` in three `replace_text` calls, and applied no patch. Its independent grade is `INDEPENDENT_GRADE_COMPLETE_NO_REPAIR`: the unchanged candidate failed the public and hidden reparse-invariance checks. The source-handle diagnostic is recorded separately in [attempt 002](../results/mlsys-fresh-vllm-64674-agent-attempt-v1/attempt-002/attempt-summary-redacted.json).

## Research scope and prior art

The current adaptive scheduler does not support a novelty claim. DAPO already filters zero-variance groups and changes the GRPO objective; Online Difficulty Filtering, VADE, and CoDaPO study online or competence-aware task selection and update weighting. Agent-search work on Lazy Likelihood Displacement also reports policy collapse mechanisms in tool-integrated RL. These papers make another synthetic difficulty-scheduler comparison a weak next experiment.

Asynchronous rollout throughput and staleness control are also established topics. AReaL, StaleFlow, DORA, and PACE address asynchronous execution, stale trajectories, or throughput coordination. The plausible VARE research gap is narrower: independently graded repository-repair trajectories whose task, candidate source, behavior policy, verifier version/configuration, and final reward remain bound across delayed grading, policy changes, and recovery. This is a research framing, not a validated novelty claim; it needs real eligible tasks and measured learning/system outcomes.

## Task eligibility and safety gate

The retained [deep-qualification report](mlsys-task-deep-qualification-v1-report.md) contains 15 source-level regression tasks split 3 Train / 3 Development / 9 Held-out. A retrospective exposure audit found eight prior inference attempts across all three Development task families, leaving zero fresh Development tasks in that split. The nine Held-out source tasks remain unused, but the prior qualification report says VARE's narrow function grader does not yet provide the required hidden-upstream-pytest boundary for those tasks.

The [metadata-only candidate screen v4](../results/mlsys-fresh-development-candidate-screen-v4/evidence-manifest.json) contains 3,305 candidates, including 1,278 provisional rows. Its frozen record explicitly sets qualification, model inference, and training authorization to false. Candidate rows are therefore leads for a new append-only qualification protocol, not eligible tasks.

[vLLM #64674](https://github.com/vllm-project/vllm/pull/64674), an incremental decoder for Responses custom-tool streaming, was source-qualified for a narrow Development contract. Baseline and incorrect-mutant controls fail, while the exact upstream fix passes the public regression and independent hidden checks. The task was assigned to Development and has since been exposed to two Qwen3-4B attempts; it is not fresh or held-out evidence. No Held-Out task identity was accessed.

The latest attempt stopped after its fixed 12 tool turns. It issued five source handles, referenced handles in three `replace_from_read` calls, and received an exact-match rejection each time (`match_count: 0`). The candidate diff stayed empty; neither the public test nor the independent grader ran for this attempt. See the [redacted outcome](../results/mlsys-fresh-vllm-64674-agent-attempt-v1/attempt-002/attempt-summary-redacted.json) and [frozen diagnostic protocol](../results/mlsys-fresh-vllm-64674-agent-attempt-v1/attempt-002/protocol.json).

## Engineering changes

- Added an AST-only Python symbol index and a repeated-empty-search guard. After three related empty literal searches, the controller offers symbol and bounded-range navigation; later related searches are intercepted until new source context is read or changed. The symbol index parses text and never imports candidate modules.
- Added optional path-scoped literal search and SHA-256-bound exact replacement for new navigation-enabled runs. Replacement checks the bytes read by the agent and verifies the exact postimage.
- Added a bounded malformed-edit retry intervention that preserves the parser's concrete error, verifies whether source bytes changed, and does not fabricate a successful tool call or run a grader for a rejected edit.
- Added opt-in short source-read handles. Each handle binds a permitted read to its exact path and full-file SHA-256; `replace_from_read` accepts only a unique exact snippet from that read and verifies the postimage. The interface rejects stale, wrong-path, and nonmatching edits. The lifecycle fixture covers an applied source edit and public/independent-grader handoff using explicit test doubles; it is not model evidence.
- Saved attempts now record whether navigation tools were enabled. Retrospective context-compaction replay reconstructs the old frozen tool schemas for older attempts and the expanded schemas only for attempts that recorded the new flag. The legacy default schemas remain unchanged for frozen campaign protocols.
- Long-horizon episode schema v2 now binds trusted final success to a terminal episode, all expected committed steps, the trajectory head, registered verifier configuration, the graded candidate's final source hash, and the task/source/workspace identity. Configured public tests must also pass. Legacy v1 success retains its prior checks.

The real-model diagnostic confirmed that the source-handle tool was exposed, issued handles for the files actually read, and rejected malformed exact text without changing source. It did not demonstrate a model-applied repair. No additional model attempt is authorized in this closure cycle.

## Validation

Passed for the current PR #181 head `414904e333d53b733abc26c3f4c21125a8e2fe3a`:

- Focused local pytest coverage for agent tool formatting, exact edit handling, separated grading, and the full runner lifecycle fixture passed with the pinned Colima test image.
- GitHub checks passed for Python 3.11, 3.12, and 3.13, lint, typecheck, blob budget, CPU evidence, and optimizer/recovery.
- `ruff`, `py_compile`, and `git diff --check` passed locally.
- The bounded source-handle diagnostic passed frozen-input and resource preflight. It ran for 211.5 seconds, generated 1,535 tokens over 12 tool turns, stayed below its wall and RSS caps, and made no source change.

The engineering tests validate controller behavior and protocol compatibility. The model diagnostic is a negative edit result, not a coding-capability or policy-improvement result.

## Separate estimands

| Estimand | Current evidence | Status |
| --- | --- | --- |
| Capability | Prior #180 Development 2/4; prior full long-horizon success 0/4 in both arms; latest exposed #64674 diagnostic applied no edit | No new estimate |
| Training systems | Prior matched updates and recovery checks only; no new trajectory-derived update | No new estimate |
| Long horizon | Controller and final-success accounting regressions only; no fresh repository cohort | No new estimate |

## Next evidence gate

Do not run another model attempt on this exposed task in the current closure cycle. Finish the engineering review and integration decision for PR #181, then review the already developed #183 and #182 work. Any later flagship must start from clean main with one focused hypothesis, an append-only protocol, and task and grader boundaries frozen before inference. Training remains gated on complete behavior-policy data, trusted rewards bound to exact trajectory and verifier versions, and independent task success. No Held-Out access is authorized by this diagnostic.

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
