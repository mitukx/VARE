# VARE research and engineering checkpoint

**Updated:** 2026-10-31 (JST)

**Milestone:** P3 — prospectively qualify a fresh Development task cohort

**Decision:** PR #150 is merged at `cb5c9d91ce269014b4166644248aa2b71a5c3dae`; its recorded required checks passed. PR #152 is merged on `main` at `ce097a5d2c695d4fa7a2e2119c3e25bc3a460578`, and the latest remote `main` matches. The candidate pipeline found 1,278 metadata-review rows and produced a deterministic 71-task sample. Three repository-diverse fixes are selected, but **zero tasks are qualified so far**. The exact source/test hashes, selectors, runtimes, fixture, and budgets are frozen in qualification addendum 1. No regression assertion has executed. Transformers baseline attempt 1 failed at collection due to a Hub/tokenizer dependency mismatch (zero cases); the raw log is retained and addendum 2 pins a corrected runtime before retry. No model or agent saw candidate content.

## GitHub state

- Last verified remote `main`: `ce097a5d2c695d4fa7a2e2119c3e25bc3a460578`.
- Last integration: [PR #152 — Audit candidate exposure and record a bounded Train-task model pass](https://github.com/mitukx/VARE/pull/152), merged at that SHA with all recorded required checks successful.
- PR #150’s CI regressions are resolved in the merged PR; recorded exact-head checks passed.
- Current branch: `codex/fresh-dev-candidate-discovery-v1`, based on the verified `main` SHA.
- Active PR: none yet; qualification evidence is being prepared before PR creation.
- PR #149 and other older branches remain open; they are unrelated and were not modified.

## Work completed

- Completed the fresh public-PR metadata scan across six upstream repositories. The corrected weekly-partition scan captured 3,305 merged PR metadata rows without reading source diffs, PR bodies/comments, test contents, or Held-out assets.
- Corrected the GraphQL `MERGED` state screen; 1,278 rows remain for metadata review. All remain `UNQUALIFIED`, `UNASSIGNED`, and `NOT_ASSESSED` until source-level calibration.
- Generated the fixed-seed 71-row stratified metadata sample. Selected Transformers #49418, TRL #7668, and MLX #4778 for source/test qualification.
- After freezing task selection, inspected only the selected source/test paths and recorded base/fix/test hashes. Prepared pinned CPU-only runtime images and an offline 20.8 MB test fixture. Frozen addendum 1 explicitly permits only the two TRL regression test cases’ tiny forward/loss/autograd checks; it forbids generation, optimizer updates, model/agent prompts, policy inference/training, Development assignment, and Held-out access.
- The preparation log retains the MLX build-environment setup failures and their fixes. The MLX package was built with CPU enabled and Metal disabled; fixed and baseline runtimes are separately pinned.

## Validation and experiment state

- PR #150 and PR #152 required-check records: **PASSED**.
- Candidate discovery, corrected metadata screen, deterministic sample, source/test hashes, and offline fixture manifest: **verified**.
- Selected task regression assertions: **NOT RUN**. Transformers baseline collection attempt 1: **INCONCLUSIVE — environment failure**, zero cases; no fixed arm started.
- Qualified fresh Development tasks: **0**.
- Coding capability milestone: **NO-GO** pending a qualified prospective Development cohort and independent model evaluation.
- Model policy update, SFT, GRPO, RL comparison, Development generalization, and Held-out evaluation: **NOT RUN**.
- No paid compute or API spend; no accelerator was used.

## Immediate next action

Rerun the Transformers baseline under addendum 2, then its fixed arm if the targeted assertions execute. Next run TRL and MLX under addendum 1, one task at a time, in no-network containers. A task qualifies only if the baseline fails the targeted assertions, the exact fix passes every expected parameter case, and no cases skip or fail for environmental reasons. Do not assign any task to Development unless all three qualify.

## External blockers

- The historical 24-entry queue has no fresh Development-eligible task; this new cohort is still in qualification.
- Public pretraining exposure of these public upstream patches remains unknown.
- No genuine policy update or independent post-update task-success comparison has been run.
