# VARE Flagship Mission Status

## Current checkpoint

- Last verified `origin/main`: `416f35c465712898e8e5764a8e43193ef86e7afb`; PR #159 is integrated.
- PR #150 is confirmed merged with its exact-head CI, CPU-evidence, and E2E checks passing. Its historical three regressions are closed.
- PR #160 is open and ready for review at `71aa74ab6a8f6f1625505b68f7f332c49fabff93`. This update fixes both newly posted automated-review findings. The targeted local suite passed **19 tests**; exact-head GitHub CI and CPU evidence are running. Normal review and branch protection remain required.
- PR #161 is open as a draft at `ade8a0ac3dfe609e67698638faf6eb826101dbc5`, stacked on #160. It now merges the #160 review fix; exact-head CI and CPU evidence are running. No review or reviewer request is recorded.
- Active branch: `codex/freshdev-qualification-vllm64830-oct31`, locally based on updated PR #161 at `78d6a0a3cfcd8a9865a688f93046d7b7aa298139`. The qualification checkpoint is to be pushed before running its candidate tests; PR #162 is not open yet.
- Milestone: `p15-fresh-vllm-64830-source-qualification`.

## Completed in this checkpoint

- Corrected `_check_artifact_hashes` so `frozen_inputs.task` is validated before generic artifact filtering. Missing task entries, paths, hashes, files, or content mismatches now fail closed.
- Corrected bounded-attempt cleanup to send `SIGKILL` to surviving process-group members even if the group leader already exited.
- Added regressions for changed/deleted task JSON and prompt files, malformed manifests, and a SIGTERM-ignoring child. Ruff and `py_compile` passed; the focused tests passed **19/19**. Result record: `results/pr160-review-fixes-v1/run-001.json`.
- Verified the exact #64830 source identity and expected three-file changes. The metadata-only overlap review found no direct identity/family overlap at moderate confidence; pretraining exposure remains unknown.
- Froze the execution addendum before any candidate test: `results/mlsys-fresh-vllm-64830-source-qualification-v1/protocol-addendum-2-execution.json`, SHA-256 `96fbf7a9c05373eab6dae43f5ce057f6f686e3766eab3aaae8e360d38da0a856`.
- Prepared a hash-bound overlay with unchanged upstream assertions for HiSparse and watermarking, exact baseline/fixed method extracts, and a mutant that removes those two checks. The modified custom-speculator behavior is explicitly outside this scoped qualification.

## Capability and research status

- PR #64830 is **UNQUALIFIED / UNASSIGNED** until its baseline, mutant, fixed, and isolation gates pass.
- No #64830 candidate test has executed. No model inference, training, or Held-Out access has occurred in this source qualification.
- Coding capability and RL update remain **NO-GO**. Prior vLLM #64575 model attempt remains **NOT RUN** at its failed memory preflight; do not retry it.

## Immediate next action

Push this frozen checkpoint. Then run the baseline arm exactly once using its `runner_argv` in the execution addendum. Continue to the wrong-mutant arm only if both cases fail at the intended post-override assertion; continue to upstream-fixed only if that gate also passes. Keep #64830 unqualified and unassigned on any unexpected outcome.
