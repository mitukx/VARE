# VARE research and engineering checkpoint

**Updated:** 2026-10-31 (JST)

**Milestone:** P2 — first independently graded base-model trajectory on the frozen Train split

**Decision:** PR #151 is merged on `main` at `818fd8c802277ad360ee9b5250d0315cfb8dad81`. The 24-entry queue audit found no fresh Development candidates. One preregistered Qwen3-4B attempt on the already assigned Train task MLX-LM #1904 passed its two-case independent grader. This is a narrow Train-task trajectory, not Development generalization or model-learning evidence. No Held-out task or test was opened.

## GitHub state

- Last verified remote `main`: `818fd8c802277ad360ee9b5250d0315cfb8dad81`.
- Last integration: [PR #151 — Harden repository-task source staging against symlinked parents](https://github.com/mitukx/VARE/pull/151), merged at the SHA above. Exact-head Python 3.11/3.12/3.13, lint, typecheck, CPU evidence, and E2E recovery checks passed.
- Current branch: `codex/mlsys-candidate-exposure-audit-v1`, based on `main`.
- Active PR: none yet. The queue audit, frozen model protocol, and run-19 evidence are staged for a focused review PR.
- PR #149 remains open on a pre-#150 base with no recorded review; it is unrelated to this checkpoint.

## Work completed

- Audited all 24 entries of the frozen manual qualification queue against the retained qualification and Stage 2/3 exposure records. The output verifies 1,120 Stage 2 and 2,194 Stage 3 evidence-file hashes.
- Queue disposition: 2 previously model-exposed Development tasks, 8 historical Held-out tasks still reserved, 3 Train tasks, 11 excluded/collapsed candidates, and **0 fresh Development-eligible tasks**. The frozen queue and historical splits were not changed. The raw survey screen is absent from this checkout, so changed-path and merge-SHA metadata could not be independently revalidated; pretraining exposure also remains unknown.
- Froze protocol v1 before inference for one deterministic local Qwen3-4B attempt on Train task `mlxlm-1904-tool-arguments`; the exact 4-bit snapshot and source/runtime/grader inputs are hash-bound.
- Run 19 passed the independent v12 grader: structured object arguments and the JSON-string control both passed (2/2). The agent used 302 generated tokens, five tool turns, and 31.679 seconds on Apple Metal. The grader was called once and returned no feedback to the agent.
- Recorded a pre-inference launch failure caused by the runner requiring a real `.git` directory; no prompt was rendered and no tokens were generated in that launch. The workspace was corrected at the same pinned source SHA before the one authorized inference.
- Appended the new Train exposure without modifying prior Stage 2/3 ledgers or the earlier queue audit.

## Validation and experiment state

- Run-19 evidence bundle checksum verification: passed.
- Independent grader: **PASSED, 2/2 cases** (`dict_arguments`, `json_string_control`).
- Model policy update, SFT, GRPO, RL comparison, Development generalization, and Held-out evaluation: **NOT RUN**.
- Coding-capability milestone: **NO-GO** until a prospective Development cohort exists and is evaluated independently. RL update: **NO-GO**.
- No paid compute, paid API, or external inference was used. Apple Metal inference was local.

## Immediate next action

Create and push a focused PR with the audit protocol/script/output, the frozen one-attempt task protocol, run-19 raw evidence, and durable mission updates. Inspect exact-head required checks and review, fix any actionable issues, and integrate only after repository requirements are met. Then prospectively qualify a new, non-overlapping Development cohort without opening any Held-out files or tests.

## External blockers

- The audited 24-entry manual queue contains no task currently assigned and qualified as fresh Development.
- The raw survey screen artifact was not present, limiting retrospective revalidation of its source changed-path and merge-SHA metadata.
- Public-task pretraining exposure is unknown; the single Train-task pass cannot establish generalization.
- No real policy update or independent post-update task-success comparison has been executed.
- PR #149 is an unrelated legacy open PR based on a pre-#150 integration point.
