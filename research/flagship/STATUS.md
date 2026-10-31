# VARE Flagship Mission Status

## Current checkpoint

- Milestone: p12-rvl-grpo-training-generation-task-identity-pending-ci.
- Last verified `origin/main`: `96b3cb86046ccde579caa1c65ccde9f8a2c76e65`, merge of PR #157. PR #156 remains the last code integration.
- Active branch: `codex/rvl-training-generation-task-binding`; PR #158 is open and contains implementation commit `af8508d42c431a54373497d15e53085252202f41`. This checkpoint update advances the PR head, so CI must be refreshed before merge.
- A frozen ungrouped training-boundary regression reproduced two cases on main where RVL `prompt_id` or prompt text differed from the attached VARE task and still reached `train_step`. The adapter now rejects the mismatch before constructing the trainer input.
- Local validation: 79 tests passed across 11 RVL-focused files; Ruff, `py_compile`, protocol hashes, and result hashes passed. PR #158 CI started for the implementation commit; its latest checkpoint head still needs exact-head validation.

## Evidence and limits

- The training-generation identity defect is a deterministic VARE adapter-boundary finding. A clean base worktree reproduced both failures; the fake trainer does not establish historical incidence or a real optimizer/task-success effect.
- The #49458 Qwen3-4B attempt emitted three incomplete edit calls at 256, 255, and 256 tokens under a 256-token per-turn cap. No edit was dispatched and the candidate diff was empty. The family is consumed and must not be retried.
- Coding capability remains NO-GO. SFT/RL update remains NO-GO. Held-Out remains untouched. Public pretraining exposure remains unknown. The current frozen task inventory has zero fresh Development candidates.

## Immediate next action

Refresh PR #158's head and CI after this checkpoint update. Review every required check and review thread, fix failures, and merge only when branch protection permits; then verify the merge on `origin/main`. Keep inference, SFT/RL, and Held-Out gated until a distinct fresh Development cohort and frozen protocol exist.
