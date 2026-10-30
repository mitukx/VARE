# Reproduction and resumption

Run commands from the VARE repository root. The current local work is limited to task-grader integrity and exposure reconciliation. No model inference, SFT, GRPO, or held-out evaluation is authorized by this checkpoint.

## Test current implementation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q \
  tests/test_development_repository_repair_campaign_v1.py \
  tests/test_development_repository_repair_campaign_v2.py \
  tests/test_isolated_candidate_expected_exception.py

for py in 311 312 313; do
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
    "work/ci-py${py}/bin/python" -m pytest -q \
    tests/test_mlsys_repo_task.py \
    tests/test_mlsys_deep_qualification_exposure.py
done

ruff check \
  src/vare/mlsys_repo_task.py \
  tests/test_mlsys_repo_task.py \
  scripts/run_mlxlm_1904_calibration.py \
  scripts/red_team_mlxlm_1904.py \
  scripts/audit_mlsys_deep_qualification_exposure_v1.py \
  tests/test_mlsys_deep_qualification_exposure.py

python -m py_compile \
  src/vare/mlsys_repo_task.py \
  scripts/run_mlxlm_1904_calibration.py \
  scripts/red_team_mlxlm_1904.py \
  scripts/audit_mlsys_deep_qualification_exposure_v1.py
git diff --check
```

## Reconcile prior task exposure

The auditor verifies both pinned input hashes, maps task IDs to source PR and family, checks per-stage attempt totals, and refuses to overwrite its output.

```bash
PYTHONPATH=src python scripts/audit_mlsys_deep_qualification_exposure_v1.py \
  --stage3-ledger ../../outputs/VARE-stage3-2026-10-31/task-exposure-ledger.json \
  --output work/exposure-audit-copy.json

shasum -a 256 results/mlsys-task-deep-qualification-v1-exposure-append-v1.json
```

Do not relaunch #45852, #1912, or #1971 as fresh Development tasks. This audit is retrospective and does not reallocate them to Train or Held-out.

## Reproduce the Train-only #1904 grader calibration

Requires a clean VARE commit containing protocol v11, Colima with the pinned image already cached, and a clean `mlx-lm` checkout containing both source revisions. The run script refuses to execute from a dirty VARE worktree, overwrites no evidence directory, and is restricted to the frozen Train task. It authorizes no model rollout.

```bash
git -C ../mlx-lm-source cat-file -e \
  2af5f7ce8620115bf37fbbddf9817f083118df59^{commit}
git -C ../mlx-lm-source worktree add --detach \
  "$PWD/../mlx-lm-baseline-1904-replay" \
  2af5f7ce8620115bf37fbbddf9817f083118df59

mkdir -p ../colima-tmp
TMPDIR="$PWD/../colima-tmp" PYTHONPATH=src DOCKER_CONTEXT=colima \
  work/ci-py311/bin/python scripts/run_mlxlm_1904_calibration.py \
  --source-root ../mlx-lm-baseline-1904-replay \
  --output results/mlsys-repo-eval-v1/run-17-replay/calibration.json

shasum -a 256 \
  results/mlsys-repo-eval-v1/run-17-replay/calibration.json \
  results/mlsys-repo-eval-v1/run-17-replay/calibration.red-team.json
```

Run 17 passed on implementation commit `36af6f17908b50cb53bb159013eecfa8b0263da0`: baseline failed, known fix passed with exact source identity, source swap was rejected, all nine red-team controls passed, and the symlinked parent was rejected before any grader call. Its manifest SHA-256 is `921e1260451bf94d9a7d8f19e3f62fed4b5b2c3dfec6d09aeca71ab0756c9ed6`. Verify run evidence with `(cd results/mlsys-repo-eval-v1/run-17 && shasum -a 256 -c SHA256SUMS)`, and verify the cumulative root index with `shasum -a 256 -c results/mlsys-repo-eval-v1/SHA256SUMS` from the VARE root. Run 17's affected 23-test suite passed on Python 3.11, 3.12, and 3.13. The v11 protocol and evidence remain immutable.

The PR #151 code review found that opening an unwritten FIFO with blocking `O_RDONLY` could stall host-side source validation before the Docker timeout. Protocol v12 adds nonblocking leaf open and a tenth probe, `workspace_fifo_probe`; it is frozen at SHA-256 `36c5932ab46e00573535f1d6826396cc24cc5021703566f49b9169cad45460c4`. Run 18 **PASSED** on implementation commit `a7d39bda1da662c18bc7b6738b95f605b679ee30`: both task arms matched expectations, the source swap was rejected, and all 10 probes passed. The FIFO was rejected in `0.00012s` without invoking the grader. Manifest SHA-256: `24e4ed3e7e5749a3bd19bdf721cc54104fae913d8902352487a9f3f8801b7d4a`. The focused 25-test suite passed on Python 3.11, 3.12, and 3.13. The run-18 evidence is separate from and does not modify run 17.

```bash
git -C ../mlx-lm-source worktree add --detach \
  "$PWD/../mlx-lm-baseline-1904-v12" \
  2af5f7ce8620115bf37fbbddf9817f083118df59
mkdir -p ../colima-tmp
TMPDIR="$PWD/../colima-tmp" PYTHONPATH=src DOCKER_CONTEXT=colima \
  work/ci-py311/bin/python scripts/run_mlxlm_1904_calibration.py \
  --source-root ../mlx-lm-baseline-1904-v12 \
  --output results/mlsys-repo-eval-v1/run-18/calibration.json
shasum -a 256 \
  results/mlsys-repo-eval-v1/run-18/calibration.json \
  results/mlsys-repo-eval-v1/run-18/calibration.red-team.json
```

The 25-test focused suite passed locally on Python 3.11, 3.12, and 3.13. The v12 acceptance gate preserves all run-17 conditions and requires the FIFO to be rejected in under one second without invoking the candidate grader.

## Full CI and PR state

PR #150 is integrated at `cb5c9d91ce269014b4166644248aa2b71a5c3dae`. Its exact-head CI and companion recovery/CPU workflows are recorded in `state.json`. For any new PR, push the branch, inspect all required GitHub Actions, read failures, review the complete diff and unresolved review threads, then merge only if checks and repository rules permit. Verify the merge SHA on remote `main`.

## Model, receipt, and resume status

- The prior Qwen3-4B authored-task cohort has one verified success in two tasks; it produced no policy-update checkpoint and establishes no RL or generalization gain.
- The deep-qualification Development split has zero fresh tasks after the append-only exposure correction. No model checkpoint or optimizer state for that source-task cohort exists.
- The Stage 3 candidate receipt remains at `../../outputs/VARE-stage3-2026-10-31/capability-diagnostic-exact-replace-v4/grader-receipts/stage3_qwen3_4b_exact_replace_v4__mlxlm-1912-top-p-min-token.terminal-lf-regrade.json`; it is `wrong_output` on 0/14 correctness cases.
- No resumable model or optimizer run exists. Do not relaunch the consumed task/model pairs.

After this checkpoint is integrated, the highest-priority research work is a full family-overlap audit of the 24-entry manual candidate queue, followed by source qualification of a new Development cohort that does not consume any Held-out task.
