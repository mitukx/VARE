# Reproduction and resumption

Run commands from the VARE repository root. The current checkpoint includes a frozen, one-attempt Train-only model trajectory on MLX-LM #1904. That attempt is consumed; do not rerun it. No model update, SFT, GRPO, Development evaluation, or Held-out evaluation was authorized or executed.

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

PR #151 is integrated at `818fd8c802277ad360ee9b5250d0315cfb8dad81`; PR #150 is in its history at `cb5c9d91ce269014b4166644248aa2b71a5c3dae`. PR #151 exact-head Python 3.11/3.12/3.13, lint, typecheck, CPU evidence, and E2E recovery checks passed. PR #152 contains the current audit and run-19 evidence; its exact-head checks are pending. See `state.json` for the verified merge and current PR.

## Run 19 — one Train-only model trajectory

Protocol `protocols/mlsys_task_model_attempt_mlxlm_1904_v1.json` was committed before inference. It pins one greedy Qwen3-4B-Instruct-2507-4bit MLX attempt on the existing Train task `mlxlm-1904-tool-arguments`, base `2af5f7ce8620115bf37fbbddf9817f083118df59`, one allowed file, 2,048 total generated tokens, 12 turns, 300 seconds, and one independent v12 grader call. The exact model snapshot, candidate prompt/spec, controller, and grader hashes are recorded in the protocol and run manifest.

The run completed in 31.679 seconds using 302 generated tokens over five tool turns. The independent grader passed `dict_arguments` and `json_string_control` (2/2); no grader feedback was returned to the agent. Evidence is retained in `results/mlsys-repo-eval-v1/run-19/`. A first infrastructure launch loaded weights but failed before prompt rendering or token generation because a linked worktree had a `.git` pointer file. That launch error is preserved separately and did not consume the single inference attempt. The successful attempt is consumed; do not rerun it or use grader feedback for a second attempt.

Verify the immutable run bundle and cumulative checksum index:

```bash
(cd results/mlsys-repo-eval-v1/run-19 && shasum -a 256 -c SHA256SUMS)
shasum -a 256 -c results/mlsys-repo-eval-v1/SHA256SUMS
python - <<'PY'
import json
from pathlib import Path
attempt = json.loads(Path('results/mlsys-repo-eval-v1/run-19/agent-attempt-1.json').read_text())
grade = json.loads(Path('results/mlsys-repo-eval-v1/run-19/independent-grade.json').read_text())
assert attempt['new_tokens'] == 302 and attempt['stop_reason'] == 'agent_finished'
assert grade['grader_call_count'] == 1 and grade['grader']['passed'] is True
assert [case['passed'] for case in grade['grader']['cases']] == [True, True]
PY
```

This is one independently graded task from Train. It is not Development/generalization evidence, a policy update, SFT, GRPO, or RL. The public task's pretraining exposure is unknown. The grader covers the selected function contract, not the full upstream HTTP/model-serving test. The appended exposure event is in `run-19/exposure-append.json`; the prior Stage 2/3 ledgers and frozen queue audit remain unchanged.

## Model, receipt, and resume status

- The prior Qwen3-4B authored-task cohort has one verified success in two tasks; it produced no policy-update checkpoint.
- Run 19 adds one narrow Train-task pass (2/2 cases). Its transcript, edit, independent receipt, failed pre-inference launch, model snapshot hashes, and resource telemetry are retained.
- The queue audit finds zero fresh Development candidates; the historical Held-out allocation remains untouched.
- The Stage 3 candidate receipt remains at `../../outputs/VARE-stage3-2026-10-31/capability-diagnostic-exact-replace-v4/grader-receipts/stage3_qwen3_4b_exact_replace_v4__mlxlm-1912-top-p-min-token.terminal-lf-regrade.json`; it is `wrong_output` on 0/14 correctness cases.
- No resumable model or optimizer run exists. Do not relaunch the consumed task/model pairs.

Next, integrate the audit and run-19 evidence by focused PR. Then prospectively source-qualify a diverse Development cohort without opening Held-out files or tests.

## Fresh Development task qualification — Transformers #49418

The selected task and six parameter cases are frozen in `results/mlsys-fresh-development-task-qualification-v1/protocol-addendum-4.json` (SHA-256 `6be83fbf944ee60467ed87d30e5eebeb2f8ce4af8096235908a160e17d8b2598`). Attempt 3 is an inconclusive runtime failure, not a baseline result. Runtime v3 is pinned by image ID `sha256:d93a01090a15c8c9ce47d7e08c6bc935f1abb83683f6318dff866a07f08f8ff2`; its Dockerfile and build log are under `results/mlsys-fresh-development-task-qualification-v1/environment/`.

Prepare two separate `/source` trees from the frozen base (`bd3f926f52b55a0a59e812376e7624ca9d471ac2`) and fixed (`1734d8d9d6a4c5331ba87dd4a1a36d3a3cb25ac1`) commits. Mount the fixed commit's `tests/utils/test_cache_utils.py` at `/trusted/test_cache_utils.py` in both runs. Verify the frozen source/test input manifest before launch. Use the six exact node IDs in addendum 4, the resource/security flags in addenda 1 and 4, `MAX_JOBS=1`, and read-only source/test mounts. The only additional writable mount is:

```text
/usr/local/lib/python3.12/site-packages/optimum/quanto/library/extensions/cpp/build:rw,exec,nosuid,nodev,size=256m,mode=1777
```

Run the baseline first. Do not run the fixed arm unless the intended residual-boundary correctness assertion executes and fails on baseline. Retain each raw log and a separate JSON attempt record. No candidate prompt, agent/model inference, optimizer update, Development assignment, or Held-out access is included in this qualification protocol.

The baseline attempt under addendum 4 is a valid negative control: 4/6 expected assertions failed and 2/6 passed. A first fixed-arm invocation was rejected before case execution due to a shell selector typo; its log/record are retained as inconclusive. Addendum 5 freezes the exact selector strings and authorizes one fixed-source retry. Build the pytest argv from that `exact_test_node_ids` list rather than retyping it. The retry uses the same runtime v3 image and isolation flags listed above.
