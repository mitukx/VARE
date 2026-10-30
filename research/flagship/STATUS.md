# VARE research and engineering checkpoint

**Updated:** 2026-10-31 (JST)

**Milestone:** P1 — harden task grading and reconcile Development exposure

**Decision:** PR #150 is integrated. The narrow #1904 Train-task grader calibration remains the current E0/E1 engineering check. Do not run a model on the frozen deep-qualification Development split: all three allocated tasks have prior Stage 2 model exposure, and #1912 also has three Stage 3 attempts. No held-out task was touched by this audit.

## GitHub state

- Last verified `main`: `cb5c9d91ce269014b4166644248aa2b71a5c3dae`
- Last integration: [PR #150 — Bound coding-agent tools and grader receipts](https://github.com/mitukx/VARE/pull/150), merged at `cb5c9d91ce269014b4166644248aa2b71a5c3dae`.
- PR #150's corrected head `e3b48a9cc84a877ba84fc1ae8b35fecfc86e613f` passed the required Python 3.11/3.12/3.13 CI matrix and companion recovery/CPU checks. Recorded Actions runs: `41129971394`, `41129977978`, `41129977929`, `41129977995`.
- Current branch: `codex/mlsys-qualification-gates` from the verified merge commit. No PR for this branch has been opened yet.
- PR #149 is still open with no review submissions or inline review threads recorded at inspection; it has 296 changed files / 79,741 additions and is based on pre-#150 `main`. Its reported 0/4 verified repairs are not an integrated result.

## Work completed in this checkpoint

- Fetched the latest remote state and verified PR #150's merge commit on `origin/main`.
- Reproduced a concrete #1904 grader staging defect: a `mlx_lm` parent symlink caused `grade_workspace` to copy an external `server.py` into its subject snapshot. The new path reader opens each component with `O_NOFOLLOW`, validates the file descriptor as regular, and grades the exact bytes it hashed. A regression test requires rejection before the grader call.
- Audited the deep-qualification Development task IDs against the retained Stage 2/3 exposure ledger. All three are previously exposed: Transformers #45852 (1 Stage 2 attempt), MLX-LM #1912 (2 Stage 2 + 3 Stage 3 attempts), MLX-LM #1971 (2 Stage 2 attempts). None produced an independently verified repair; the #1912 edit failed all 14 hidden cases.
- Preserved the original frozen split and qualification catalog. Added a hash-bound, append-only exposure disposition: fresh Development tasks in that 3-task allocation = 0/3. The audit did not rerun tasks, use held-out tests, or launch a model.
- Froze #1904 calibration protocol v11. It adds a parent-symlink rejection probe while keeping the task in Train and authorizing no model rollout.

## Validation and experiment state

- Local targeted tests: `tests/test_mlsys_repo_task.py` and `tests/test_mlsys_deep_qualification_exposure.py` passed (23 tests).
- Ruff 0.15.12 passed on the changed source, test, and script files. `py_compile` and `git diff --check` passed.
- Formal #1904 protocol-v11 run 17: **PENDING**. It must verify pinned baseline fail, known-fix pass, source identity, prior red-team controls, and rejection of the parent-symlink workspace before any candidate-grade call.
- Coding capability: narrow prior feasibility evidence exists—Qwen3-4B completed 1/2 independently graded tasks in an authored exploratory cohort. This is one task result, not a generalization or model-learning claim. The frozen deep-qualification Development split cannot supply a fresh evaluation.
- Policy update / coding RL: **NO-GO**. No task-success policy update or held-out improvement is established.
- No paid compute, external inference, or model rollout was used in this checkpoint.

## Immediate next action

Commit the v11 code/protocol checkpoint, run the bounded Train-only calibration as `results/mlsys-repo-eval-v1/run-17/`, retain its raw calibration and red-team records, then create a focused PR. After integration, audit the 24-entry manual candidate queue against the full prior exposure history and assemble a replacement Development cohort without opening Held-out tasks.

## External blockers

- The source-level 15-task pool's original Development allocation is fully exposed; it cannot support a fresh pass@k experiment.
- The remaining manual queue has not yet been reconciled against all Stage 2/3 task-family exposure.
- No multi-family fresh Development run or post-training coding capability comparison has been executed.
