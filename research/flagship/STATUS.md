# VARE Flagship Mission Status

## Current checkpoint

- Last verified `origin/main`: `416f35c465712898e8e5764a8e43193ef86e7afb`; PR #159 is integrated.
- PR #150 is confirmed merged with its exact-head CI, CPU-evidence, and E2E checks passing.
- PR #160 is open at `71aa74ab6a8f6f1625505b68f7f332c49fabff93`. Its Python 3.11/3.12/3.13 CI, lint, typecheck, seeded-summary, RVL inference, Math500 grader, and CPU-evidence checks pass. GitHub reports merge state BLOCKED; normal review and branch protection remain.
- PR #161 is open as a draft at `ade8a0ac3dfe609e67698638faf6eb826101dbc5`, stacked on #160. Its exact-head CI and CPU evidence pass. No review is recorded.
- Active branch: `codex/freshdev-qualification-vllm64830-oct31`; draft PR #162 is open at `ba25adfe015272455c4baac959cd6de927a54dd5`, based on PR #161. CI is running after the runner correction.
- Milestone: `p15-fresh-vllm-64830-source-qualification`.

## Completed in this checkpoint

- Corrected `_check_artifact_hashes` so `frozen_inputs.task` is validated before generic artifact filtering. Missing task entries, paths, hashes, files, or content mismatches now fail closed.
- Corrected bounded-attempt cleanup to send `SIGKILL` to surviving process-group members even if the group leader already exited.
- Added regressions for changed/deleted task JSON and prompt files, malformed manifests, and a SIGTERM-ignoring child. Ruff and `py_compile` passed; the focused suite passed **19/19**. Result record: `results/pr160-review-fixes-v1/run-001.json`.
- Verified the exact #64830 source identity and expected three-file changes. The metadata-only overlap review found no direct identity/family overlap at moderate confidence; pretraining exposure remains unknown.
- Preserved baseline attempt-001: pytest setup stopped before test collection because pytest's logging plugin tried to open `/dev/null`, which Seatbelt denied. No candidate assertions executed and no later arm ran. The receipt, log, classification, and checksums are committed.
- Pushed commit `ba25adfe` with the runner logging correction and regression. Ruff, `py_compile`, and the runner test (**1 passed**) succeeded.
- Found and preserved a pre-execution error in Addendum 3: it targeted attempt-002 output directories while its Seatbelt profiles still allowed writes only to attempt-001. Addendum 3 was never invoked. Addendum 4 creates arm-specific profiles with the same default-deny/network-deny policy and only relocates each allowed write subtree to its matching attempt-002 directory. All three profile/output pairs, command arrays, hashes, and unused output roots were validated locally.
- The modified custom-speculator behavior remains excluded because its full registration and VllmConfig validation fixtures are outside the overlay.

## Capability and research status

- PR #64830 is **UNQUALIFIED / UNASSIGNED**. Attempt-001 is an environmental setup failure, not a baseline behavior result. No corrected attempt has run.
- No model inference, training, or Held-Out access has occurred in this source qualification. Coding capability and RL update remain **NO-GO**.

## Immediate next action

Commit and push Addendum 4 and the corrected profiles. Verify the frozen addendum and profiles are on PR #162, then run baseline attempt-002 exactly once using its stored `runner_argv`. Proceed to wrong-mutant and upstream-fixed only through the addendum's gates; stop on any unexpected result.
